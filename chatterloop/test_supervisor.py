"""Leases, and the task that actually answers.

The lease tests run against a fake Redis that implements only what
`LeaseManager` uses - including the two Lua scripts, whose compare-and-act
behaviour is the whole point of them being scripts.
"""

import time
import uuid
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from chatterloop.leases import LeaseManager, lease_key
from chatterloop.models import ChatterloopBot
from chatterloop.policy import AddressedOnlyPolicy, InMemoryPolicyStore
from chatterloop.provisioning import mint_bot
from chatterloop.runtime import BotIdentity
from chatterloop.tasks import answer_trigger, footprint_for
from chatterloop.testing import create_chatterloop_schema
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource
from llm.models import Agent, Model, Role, Service
from messenger.models import Conversation, Message
from neon.testing import make_account, make_organization
from neon.utils import crypto
from organization.models import ProviderCredential


class FakeRedis:
    """Only what LeaseManager uses. The two scripts are matched by shape
    rather than executed - what matters is that both compare the owner before
    acting, which is what stops a supervisor extending or deleting somebody
    else's lease."""

    def __init__(self):
        self.store = {}
        self.expiry = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        if ex is not None:
            self.expiry[key] = ex
        return True

    def get(self, key):
        return self.store.get(key)

    def eval(self, script, numkeys, key, *args):
        owner = str(args[0])
        if self.store.get(key) != owner:
            return 0
        if "expire" in script:
            self.expiry[key] = int(args[1])
            return 1
        if "del" in script:
            self.store.pop(key, None)
            self.expiry.pop(key, None)
            return 1
        return 0

    def expire_now(self, key):
        """Simulate the TTL elapsing."""
        self.store.pop(key, None)


class LeaseTests(SimpleTestCase):

    def setUp(self):
        self.redis = FakeRedis()

    def test_only_one_supervisor_wins_a_bot(self):
        """The guarantee the whole mechanism exists for: two supervisors each
        answering the same mention is what happened before there was a lease."""
        first = LeaseManager(self.redis, owner="host-a")
        second = LeaseManager(self.redis, owner="host-b")

        self.assertTrue(first.acquire("bot-1"))
        self.assertFalse(second.acquire("bot-1"))
        self.assertEqual(first.held, frozenset({"bot-1"}))
        self.assertEqual(second.held, frozenset())

    def test_renewal_extends_a_lease_we_hold(self):
        manager = LeaseManager(self.redis, owner="host-a")
        manager.acquire("bot-1")
        self.assertTrue(manager.renew("bot-1"))

    def test_renewal_fails_once_the_lease_is_gone(self):
        """Not a retryable error: a peer holds the bot and may already be
        answering for it, so the caller has to stop consuming."""
        manager = LeaseManager(self.redis, owner="host-a")
        manager.acquire("bot-1")
        self.redis.expire_now(lease_key("bot-1"))

        self.assertFalse(manager.renew("bot-1"))
        self.assertEqual(manager.held, frozenset())

    def test_a_supervisor_cannot_renew_somebody_elses_lease(self):
        """A plain EXPIRE would let a paused supervisor extend the new
        owner's lease. The script compares first."""
        first = LeaseManager(self.redis, owner="host-a")
        first.acquire("bot-1")
        self.redis.expire_now(lease_key("bot-1"))

        second = LeaseManager(self.redis, owner="host-b")
        self.assertTrue(second.acquire("bot-1"))

        self.assertFalse(first.renew("bot-1"))
        self.assertEqual(self.redis.get(lease_key("bot-1")), "host-b")

    def test_a_supervisor_cannot_release_somebody_elses_lease(self):
        """Without the comparison, a supervisor shutting down after its lease
        had expired and been taken would delete the new owner's key."""
        first = LeaseManager(self.redis, owner="host-a")
        first.acquire("bot-1")
        self.redis.expire_now(lease_key("bot-1"))

        second = LeaseManager(self.redis, owner="host-b")
        second.acquire("bot-1")

        first.release("bot-1")
        self.assertEqual(self.redis.get(lease_key("bot-1")), "host-b")

    def test_releasing_lets_a_peer_take_over_immediately(self):
        """Without this every deploy leaves the bot unheld for a whole TTL."""
        first = LeaseManager(self.redis, owner="host-a")
        second = LeaseManager(self.redis, owner="host-b")
        first.acquire("bot-1")

        first.release("bot-1")

        self.assertTrue(second.acquire("bot-1"))

    def test_re_acquiring_a_held_bot_refreshes_rather_than_fails(self):
        manager = LeaseManager(self.redis, owner="host-a")
        manager.acquire("bot-1")
        self.assertTrue(manager.acquire("bot-1"))

    def test_renew_all_reports_what_was_lost(self):
        manager = LeaseManager(self.redis, owner="host-a")
        manager.acquire("bot-1")
        manager.acquire("bot-2")
        self.redis.expire_now(lease_key("bot-2"))

        self.assertEqual(manager.renew_all(), {"bot-2"})
        self.assertEqual(manager.held, frozenset({"bot-1"}))


class FakeLLM:
    def __init__(self, reply="Here is what we decided."):
        self.reply = reply
        self.calls = []

    def stream_chat_completion(self, history, system_prompt, content, tools):
        self.calls.append(
            {"history": history, "system_prompt": system_prompt, "content": content}
        )
        for chunk in self.reply.split(" "):
            yield chunk + " "


class AnswerTaskTests(TestCase):
    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()

        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")

        self.service = Service.objects.create(name="OpenAI")
        self.model = Model.objects.create(service=self.service, model="gpt-4o-mini")
        credential = ProviderCredential(
            organization=self.org, service=self.service, is_embedding_default=True
        )
        credential.api_key = "sk-test"
        credential.save()

        role = Role.objects.create(
            organization=self.org, name="Support", system_prompt="Be helpful."
        )
        self.agent = Agent.objects.create(
            organization=self.org, name="Helper", slug="helper", role=role
        )

        self.bot, self.credential = mint_bot(
            organization=self.org,
            created_by=self.alice,
            name="Helper",
            handle="helper",
            owner_entity_id=self.alice.entity_id,
            agent=self.agent,
        )
        self.bot.model = self.model
        self.bot.save(update_fields=["model"])

        self.llm = FakeLLM()

    def tearDown(self):
        crypto._cipher = None

    def trigger(self, **kwargs):
        kwargs.setdefault("source", TriggerSource.MESSAGE)
        kwargs.setdefault("reason", TriggerReason.MENTION)
        kwargs.setdefault("author_entity_id", "entity-human")
        kwargs.setdefault("conversation_id", "conv-1")
        kwargs.setdefault("message_id", "m1")
        kwargs.setdefault("text", "@helper what did we decide?")
        kwargs.setdefault("query", "what did we decide?")
        kwargs.setdefault("dedupe_key", f"msg:{uuid.uuid4().hex}")
        return Trigger(**kwargs)

    def run_task(self, trigger=None, **patches):
        trigger = trigger or self.trigger()
        defaults = {
            "chatterloop.tasks.send_message": None,
            "chatterloop.tasks.post_comment": None,
        }
        with patch("chatterloop.tasks.LLMFactory") as factory, patch(
            "chatterloop.tasks._retrieve", return_value=[]
        ), patch("chatterloop.tasks.send_message") as send, patch(
            "chatterloop.tasks.post_comment"
        ) as comment:
            factory.return_value.create.return_value = patches.get("llm", self.llm)
            if "send_side_effect" in patches:
                send.side_effect = patches["send_side_effect"]
            answer_trigger(str(self.bot.pk), trigger.to_payload())
            return send, comment

    def test_a_reply_is_generated_and_sent_threaded(self):
        send, _ = self.run_task()

        send.assert_called_once()
        args = send.call_args[0]
        self.assertEqual(args[1], "conv-1")
        self.assertIn("decided", args[2])
        # Threaded under the message that asked, so the answer stays attached
        # to the question.
        self.assertEqual(args[3], "m1")

    def test_the_agents_system_prompt_is_used(self):
        self.run_task()
        self.assertEqual(self.llm.calls[0]["system_prompt"], "Be helpful.")
        # The stripped query, not the raw text with the handle in it.
        self.assertEqual(self.llm.calls[0]["content"], "what did we decide?")

    def test_both_sides_are_mirrored_into_neon(self):
        self.run_task()

        conversation = Conversation.objects.get()
        self.assertTrue(conversation.footprint.startswith("chatterloop:"))
        self.assertEqual(conversation.organization_id, self.org.id)

        kinds = list(
            Message.objects.filter(conversation=conversation)
            .order_by("created_at")
            .values_list("message_type", flat=True)
        )
        self.assertEqual(kinds, ["text", "ai_reply"])

    def test_the_footprint_is_namespaced_per_organization(self):
        """Conversation.footprint is globally unique, and two organizations'
        bots can be in the same chatterloop conversation - without the
        namespace the second collides onto the first and leaks its history."""
        trigger = self.trigger()
        mine = footprint_for(self.bot, trigger)

        other_org = make_organization(make_account("bob"), "Beta", "beta")
        self.bot.organization = other_org
        theirs = footprint_for(self.bot, trigger)

        self.assertNotEqual(mine, theirs)

    def test_a_comment_trigger_posts_a_comment_not_a_message(self):
        trigger = self.trigger(
            source=TriggerSource.COMMENT,
            conversation_id="",
            post_id="p1",
            comment_id="c1",
        )
        send, comment = self.run_task(trigger)

        send.assert_not_called()
        comment.assert_called_once()
        args = comment.call_args[0]
        self.assertEqual(args[1], "p1")
        # parentID is what the bot is ANSWERING; the route re-parents it.
        self.assertEqual(args[3], "c1")

    def test_a_deactivated_bot_does_not_get_one_last_word_in(self):
        """A queued task is exactly how a stopped bot speaks again. Re-checked
        here, not only at dispatch."""
        self.bot.status = ChatterloopBot.STATUS_DEACTIVATED
        self.bot.save(update_fields=["status"])

        send, _ = self.run_task()
        send.assert_not_called()

    def test_a_bot_with_no_model_does_not_answer(self):
        self.bot.model = None
        self.bot.save(update_fields=["model"])
        send, _ = self.run_task()
        send.assert_not_called()

    def test_a_bot_whose_token_was_revoked_does_not_answer(self):
        self.credential.revoked_at = self.bot.created_at
        self.credential.save(update_fields=["revoked_at"])
        send, _ = self.run_task()
        send.assert_not_called()

    def test_an_empty_generation_sends_nothing(self):
        send, _ = self.run_task(llm=FakeLLM(reply="   "))
        send.assert_not_called()

    def test_an_over_long_reply_is_trimmed_rather_than_rejected(self):
        """The API rejects anything over 5000 characters, and a rejected send
        is worse than a trimmed one - the user gets nothing at all."""
        send, _ = self.run_task(llm=FakeLLM(reply="x" * 6000))
        sent = send.call_args[0][2]
        self.assertLessEqual(len(sent), 5000)
        self.assertTrue(sent.endswith("…"))

    def test_the_reply_is_recorded_only_after_a_successful_send(self):
        """A broken outbound path must not rate-limit the bot into silence."""
        from chatterloop.client import ChatterloopAPIError

        identity = BotIdentity(self.bot.entity_id, self.bot.handle)
        store = InMemoryPolicyStore()
        policy = AddressedOnlyPolicy(identity, store=store)

        trigger = self.trigger()
        with patch("chatterloop.tasks.build_store", return_value=store):
            self.run_task(trigger, send_side_effect=ChatterloopAPIError("down"))

        self.assertIsNone(store.last_reply_at(trigger.scope))

        with patch("chatterloop.tasks.build_store", return_value=store):
            self.run_task(trigger)

        self.assertIsNotNone(store.last_reply_at(trigger.scope))
        self.assertEqual(policy.store, store)

    def test_a_failed_send_does_not_mirror_a_reply(self):
        from chatterloop.client import ChatterloopAPIError

        self.run_task(send_side_effect=ChatterloopAPIError("down"))

        # The inbound message is still mirrored - it did happen - but nothing
        # claims the bot answered.
        self.assertFalse(
            Message.objects.filter(message_type="ai_reply").exists()
        )

    def test_retrieval_failing_does_not_stop_the_answer(self):
        """An unconfigured embedding key or a Pinecone outage should cost the
        bot its memory for that turn, not its ability to answer."""
        with patch("chatterloop.tasks.LLMFactory") as factory, patch(
            "chatterloop.tasks.embedding_api_key", side_effect=RuntimeError("boom")
        ), patch("chatterloop.tasks.send_message") as send, patch(
            "chatterloop.tasks.post_comment"
        ):
            factory.return_value.create.return_value = self.llm
            answer_trigger(str(self.bot.pk), self.trigger().to_payload())

        send.assert_called_once()

    def test_an_unknown_bot_is_a_no_op(self):
        answer_trigger("no-such-bot", self.trigger().to_payload())
