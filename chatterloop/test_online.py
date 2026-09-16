"""Switching a bot's event stream on and off.

The distinction these pin down is between the two "off" switches:

  OFFLINE       stops the event stream. Nothing else changes - the chatterloop
                identity, the token and the agent binding all survive, and
                turning it back on needs no new credential.
  DEACTIVATED   revokes every token. Coming back needs a fresh one.

Conflating them is the mistake worth guarding against, because one is trivially
reversible and the other is not.
"""

import asyncio
import uuid
from unittest.mock import patch

from asgiref.sync import async_to_sync, sync_to_async
from django.test import TestCase
from django.urls import reverse

from chatterloop.leases import lease_key, running_bot_ids
from chatterloop.models import ChatterloopBot
from chatterloop.provisioning import mint_bot
from chatterloop.tasks import answer_trigger
from chatterloop.testing import create_chatterloop_schema
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource
from llm.models import Agent, Model, Role, Service
from neon.testing import client_for, make_account, make_organization
from neon.utils import crypto
from organization.models import ProviderCredential


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.published = []

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        return True

    def get(self, key):
        return self.store.get(key)

    def mget(self, keys):
        return [self.store.get(key) for key in keys]

    def eval(self, script, numkeys, key, *args):
        if self.store.get(key) != str(args[0]):
            return 0
        if "del" in script:
            self.store.pop(key, None)
        return 1

    def publish(self, channel, message):
        self.published.append((channel, message))
        return 1

    def exists(self, key):
        return 1 if key in self.store else 0

    def zremrangebyscore(self, *a, **k):
        return 0

    def zcard(self, key):
        return 0

    def zadd(self, *a, **k):
        return 1

    def expire(self, *a, **k):
        return 1


class OnlineTestCase(TestCase):
    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()

        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")
        self.client_acme = client_for(self.alice, self.org)

        service = Service.objects.create(name="OpenAI")
        self.model = Model.objects.create(service=service, model="gpt-4o-mini")
        credential = ProviderCredential(
            organization=self.org, service=service, is_embedding_default=True
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
            model=self.model,
        )

    def tearDown(self):
        crypto._cipher = None

    def url(self, bot=None):
        return reverse("api-chatterloop:bot-online", args=[str((bot or self.bot).pk)])


class DefaultsTests(OnlineTestCase):

    def test_a_new_bot_starts_online(self):
        """Minting a bot is an explicit act; making it then require a second
        one to do anything would be a step with no decision in it."""
        self.assertTrue(self.bot.is_online)
        self.assertTrue(self.bot.should_run)


class SwitchingTests(OnlineTestCase):

    def test_switching_off_changes_nothing_but_the_switch(self):
        """The whole point: no deletion, no deactivation, no revocation."""
        from chatterloop.external_models import Bot, Token

        response = self.client_acme.delete(self.url())
        self.assertEqual(response.status_code, 200)

        self.bot.refresh_from_db()
        self.assertFalse(self.bot.is_online)
        # Still active, still bound, still credentialled.
        self.assertEqual(self.bot.status, ChatterloopBot.STATUS_ACTIVE)
        self.assertEqual(self.bot.agent_id, self.agent.id)
        self.assertEqual(self.bot.model_id, self.model.id)

        self.credential.refresh_from_db()
        self.assertIsNone(self.credential.revoked_at)
        self.assertTrue(self.credential.is_live)

        # And chatterloop's own row is untouched - the bot still EXISTS there,
        # it is just not being listened for.
        self.assertTrue(Bot.objects.get(id=self.bot.bot_id).is_active)
        self.assertTrue(Token.objects.get(id=self.credential.token_id).is_active)

    def test_switching_back_on_needs_no_new_token(self):
        self.client_acme.delete(self.url())
        response = self.client_acme.post(self.url())

        self.assertEqual(response.status_code, 200)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)
        self.assertEqual(self.bot.tokens.filter(revoked_at__isnull=True).count(), 1)

    def test_the_switch_records_when_it_was_flipped(self):
        self.assertIsNone(self.bot.online_changed_at)
        self.client_acme.delete(self.url())
        self.bot.refresh_from_db()
        self.assertIsNotNone(self.bot.online_changed_at)

    def test_switching_to_the_state_it_is_already_in_is_harmless(self):
        response = self.client_acme.post(self.url())
        self.assertEqual(response.status_code, 200)
        self.assertIn("Already online", response.data["message"])

    def test_switching_on_a_bot_that_cannot_answer_says_so(self):
        """Otherwise somebody enables a bot and waits for a reply that was
        never possible."""
        self.bot.agent = None
        self.bot.is_online = False
        self.bot.save(update_fields=["agent", "is_online"])

        response = self.client_acme.post(self.url())
        self.assertIn("no agent", response.data["message"])

    def test_another_organizations_bot_cannot_be_switched(self):
        bob = make_account("bob", entity_id="entity-bob")
        other = make_organization(bob, "Beta", "beta")
        response = client_for(bob, other).delete(self.url())
        self.assertEqual(response.status_code, 404)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_switching_nudges_running_supervisors(self):
        """So the change takes effect now rather than at the next sweep."""
        redis = FakeRedis()
        with patch("chatterloop.control._client", return_value=redis):
            self.client_acme.delete(self.url())

        self.assertEqual(len(redis.published), 1)
        channel, message = redis.published[0]
        self.assertEqual(channel, "neon:bots:control")
        self.assertIn("offline", message)

    def test_a_failed_nudge_does_not_fail_the_request(self):
        """The database row is the source of truth and is already written; the
        scheduled sweep converges either way."""
        with patch(
            "chatterloop.control._client", side_effect=RuntimeError("redis down")
        ):
            response = self.client_acme.delete(self.url())

        self.assertEqual(response.status_code, 200)
        self.bot.refresh_from_db()
        self.assertFalse(self.bot.is_online)


class SupervisorHonoursTheSwitchTests(OnlineTestCase):
    """Offline means "not leased", which means no event stream."""

    def _supervisor(self, redis):
        from chatterloop.supervisor import Supervisor

        return Supervisor(redis, sweep_interval=1)

    @staticmethod
    def _inline_threads():
        """Replace `asyncio.to_thread` with a thread-sensitive equivalent.

        The sweep hands its database work to a thread so it never blocks the
        event loop. Under Django's TestCase a plain `to_thread` deadlocks - the
        test's transaction is held by the main thread and SQLite will not admit
        another - while calling it inline raises SynchronousOnlyOperation,
        because Django refuses ORM access from an async context.

        `sync_to_async(thread_sensitive=True)` is the combination that works:
        Django is satisfied it is a sync context, and the work runs in the
        thread that holds the transaction. The hop itself is an implementation
        detail of not blocking the loop, not the behaviour under test.
        """

        async def immediately(func, *args, **kwargs):
            return await sync_to_async(func, thread_sensitive=True)(*args, **kwargs)

        return patch("asyncio.to_thread", new=immediately)

    def test_an_offline_bot_is_not_a_candidate(self):
        supervisor = self._supervisor(FakeRedis())
        self.assertEqual(len(supervisor._candidates()), 1)

        self.bot.is_online = False
        self.bot.save(update_fields=["is_online"])

        self.assertEqual(supervisor._candidates(), [])

    def test_switching_off_stops_a_running_bot_and_frees_its_lease(self):
        """Not a separate disconnect step: the sweep drops anything that has
        stopped being a candidate, and that IS going offline."""
        redis = FakeRedis()
        supervisor = self._supervisor(redis)
        started = []

        with self._inline_threads(), patch(
            "chatterloop.supervisor.BotWorker"
        ) as worker_class:
            worker = worker_class.return_value
            worker.fatal = ""
            worker.start.side_effect = lambda: started.append(1)

            async def stop():
                return None

            worker.stop.side_effect = stop

            async_to_sync(supervisor.sweep)()
            self.assertEqual(len(supervisor.workers), 1)
            self.assertIn(lease_key(str(self.bot.pk)), redis.store)

            self.bot.is_online = False
            self.bot.save(update_fields=["is_online"])

            async_to_sync(supervisor.sweep)()

        self.assertEqual(supervisor.workers, {})
        # Released, so switching back on is picked up immediately rather than
        # after a lease TTL.
        self.assertNotIn(lease_key(str(self.bot.pk)), redis.store)

    def test_switching_back_on_makes_it_a_candidate_again(self):
        supervisor = self._supervisor(FakeRedis())
        self.bot.is_online = False
        self.bot.save(update_fields=["is_online"])
        self.assertEqual(supervisor._candidates(), [])

        self.bot.is_online = True
        self.bot.save(update_fields=["is_online"])
        self.assertEqual(len(supervisor._candidates()), 1)


class QueuedWorkTests(OnlineTestCase):

    def trigger(self):
        return Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.MENTION,
            author_entity_id="entity-human",
            conversation_id="conv-1",
            message_id="m1",
            text="@helper hello",
            query="hello",
            dedupe_key=f"msg:{uuid.uuid4().hex}",
        )

    def _run_task(self):
        """Run the answering task with a model that WOULD produce a reply.

        The fake has to actually generate something, or the test proves
        nothing: a task that fails earlier for an unrelated reason sends
        nothing either, and would pass whether or not the switch is honoured.
        """

        class FakeLLM:
            def stream_chat_completion(self, history, system_prompt, content, tools):
                yield "a reply"

        with patch("chatterloop.tasks.LLMFactory") as factory, patch(
            "chatterloop.tasks.send_message"
        ) as send, patch("chatterloop.tasks._retrieve", return_value=[]), patch("chatterloop.tasks.fetch_messages", return_value=([], "group")), patch("chatterloop.tasks.fetch_thread", return_value=([], False)):
            factory.return_value.create.return_value = FakeLLM()
            answer_trigger(str(self.bot.pk), self.trigger().to_payload())
        return send

    def test_an_online_bot_does_answer(self):
        """The control for the test below - without it, "did not send" could
        mean the task was broken rather than the switch working."""
        self._run_task().assert_called_once()

    def test_a_bot_switched_off_does_not_get_one_last_word_in(self):
        """A task queued just before the switch is exactly how an offline bot
        speaks anyway."""
        self.bot.is_online = False
        self.bot.save(update_fields=["is_online"])

        self._run_task().assert_not_called()


class RunningReportingTests(OnlineTestCase):
    """"Should be running" and "is running" are different questions."""

    def test_running_is_read_from_the_lease(self):
        redis = FakeRedis()
        self.assertEqual(running_bot_ids([self.bot.pk], client=redis), set())

        redis.set(lease_key(str(self.bot.pk)), "host-a")
        self.assertEqual(
            running_bot_ids([self.bot.pk], client=redis), {str(self.bot.pk)}
        )

    def test_an_unreachable_redis_reports_nothing_running(self):
        """The honest answer when the thing that would know cannot be reached
        is not "yes"."""

        class Broken:
            def mget(self, keys):
                raise RuntimeError("redis down")

        self.assertEqual(running_bot_ids([self.bot.pk], client=Broken()), set())

    def test_the_listing_reports_both_states(self):
        redis = FakeRedis()
        with patch("chatterloop.leases.get_redis_connection", create=True):
            with patch("chatterloop.views.running_bot_ids", return_value=set()):
                response = self.client_acme.get(reverse("api-chatterloop:bots"))

        row = response.data["data"][0]
        # Switched on and configured...
        self.assertTrue(row["is_online"])
        self.assertTrue(row["should_run"])
        # ...but nothing is actually holding it, which the UI has to be able to
        # tell apart from "offline".
        self.assertFalse(row["running"])
        _ = redis

    def test_a_bot_being_run_reports_running(self):
        with patch(
            "chatterloop.views.running_bot_ids", return_value={str(self.bot.pk)}
        ):
            response = self.client_acme.get(reverse("api-chatterloop:bots"))

        self.assertTrue(response.data["data"][0]["running"])


class OfflineIsNotDeactivationTests(OnlineTestCase):
    """The two switches must not be confused for each other."""

    def test_deactivating_revokes_but_switching_off_does_not(self):
        self.client_acme.delete(self.url())
        self.credential.refresh_from_db()
        self.assertIsNone(self.credential.revoked_at)

        self.client_acme.delete(
            reverse("api-chatterloop:bot-detail", args=[str(self.bot.pk)])
        )
        self.credential.refresh_from_db()
        self.assertIsNotNone(self.credential.revoked_at)

    def test_a_deactivated_bot_is_not_run_even_if_it_is_marked_online(self):
        """Offline and deactivated are independent, and either one is enough to
        stop it."""
        from chatterloop.supervisor import Supervisor

        self.bot.status = ChatterloopBot.STATUS_DEACTIVATED
        self.bot.save(update_fields=["status"])

        self.assertTrue(self.bot.is_online)
        self.assertFalse(self.bot.should_run)
        self.assertEqual(Supervisor(FakeRedis(), sweep_interval=1)._candidates(), [])
