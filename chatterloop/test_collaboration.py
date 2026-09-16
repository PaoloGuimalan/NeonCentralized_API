"""Two bots working through a task together.

The behaviour being protected is a long exchange that ENDS - on the task being
finished, or on a budget, but not on a rate limit that silently makes the
conversation unrecoverable. Most of these run the real policy over a simulated
back-and-forth, because the bug they replace was only visible after sixty turns
and no single-decision test would have caught it.
"""

import time

from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from chatterloop.models import ChatterloopBot
from chatterloop.provisioning import mint_bot
from chatterloop.testing import create_chatterloop_schema
from llm.models import Agent, Model, Role, Service
from neon.testing import client_for, make_account, make_organization
from neon.utils import crypto

from chatterloop.policy import (
    BOT_TURN_BUDGET,
    AddressedOnlyPolicy,
    InMemoryPolicyStore,
    Verdict,
)
from chatterloop.runtime import BotIdentity
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource

HUMAN = "entity-human"
BOT_A = "entity-a"
BOT_B = "entity-b"
BOTS = {BOT_A, BOT_B}


def is_bot(entity_id):
    return str(entity_id) in BOTS


def policy(allow=True, **kwargs):
    return AddressedOnlyPolicy(
        BotIdentity(BOT_B, "beta"),
        store=InMemoryPolicyStore(),
        is_bot_author=is_bot,
        allow_bot_conversations=allow,
        **kwargs,
    )


def trigger(author=BOT_A, n=1, conversation="conv-1"):
    return Trigger(
        source=TriggerSource.MESSAGE,
        reason=TriggerReason.REPLY,
        author_entity_id=author,
        conversation_id=conversation,
        message_id=f"m{n}",
        text="hi",
        query="hi",
        dedupe_key=f"msg:m{n}",
    )


class ToggleTests(SimpleTestCase):

    def test_off_means_a_bot_is_not_answered(self):
        decision = policy(allow=False).evaluate(trigger())
        self.assertIs(decision.verdict, Verdict.IGNORE)
        self.assertIn("off for this bot", decision.reason)

    def test_off_is_terminal_so_the_message_is_not_reconsidered(self):
        """Nothing about waiting changes a toggle. Leaving it retryable would
        re-offer the same message on every later frame, forever."""
        self.assertFalse(policy(allow=False).evaluate(trigger()).transient)

    def test_on_means_a_bot_is_answered(self):
        self.assertTrue(policy(allow=True).evaluate(trigger()).should_respond)

    def test_a_person_is_answered_either_way(self):
        for allow in (True, False):
            self.assertTrue(
                policy(allow=allow).evaluate(trigger(author=HUMAN)).should_respond,
                f"allow={allow}",
            )

    def test_without_a_detector_everything_counts_as_a_person(self):
        """The conservative direction: a person is subject to the ordinary
        hourly cap, not the collaboration budget."""
        plain = AddressedOnlyPolicy(
            BotIdentity(BOT_B, "beta"), store=InMemoryPolicyStore()
        )
        self.assertTrue(plain.evaluate(trigger()).should_respond)


class CooldownTests(SimpleTestCase):
    """For a person it collapses repeats. Between bots it is pacing."""

    def test_a_collaborating_bot_waits_instead_of_being_dropped(self):
        p = policy()
        now = time.time()
        p.record_reply(trigger(n=1), now=now)

        decision = p.evaluate(trigger(n=2), now=now + 1.0)

        self.assertTrue(decision.should_respond)
        self.assertAlmostEqual(decision.delay, 4.0, places=1)

    def test_a_person_repeating_themselves_still_gets_one_answer(self):
        p = policy()
        now = time.time()
        p.record_reply(trigger(author=HUMAN, n=1), now=now)

        decision = p.evaluate(trigger(author=HUMAN, n=2), now=now + 1.0)

        self.assertIs(decision.verdict, Verdict.IGNORE)
        self.assertIn("cooldown", decision.reason)

    def test_no_wait_once_the_cooldown_has_passed(self):
        p = policy()
        now = time.time()
        p.record_reply(trigger(n=1), now=now)

        decision = p.evaluate(trigger(n=2), now=now + 30.0)

        self.assertTrue(decision.should_respond)
        self.assertEqual(decision.delay, 0.0)


class BudgetTests(SimpleTestCase):

    def _spend(self, p, turns, conversation="conv-1"):
        now = time.time()
        for i in range(turns):
            p.record_reply(trigger(n=i, conversation=conversation), now=now + i * 10)

    def test_a_collaboration_stops_when_the_budget_is_spent(self):
        p = policy()
        self._spend(p, BOT_TURN_BUDGET)

        decision = p.evaluate(trigger(n=999), now=time.time() + 10_000)

        self.assertIs(decision.verdict, Verdict.IGNORE)
        self.assertIn("budget spent", decision.reason)

    def test_the_budget_ceiling_is_terminal(self):
        """A ceiling that waiting lifts is not a ceiling."""
        p = policy()
        self._spend(p, BOT_TURN_BUDGET)
        self.assertFalse(p.evaluate(trigger(n=999)).transient)

    def test_answering_a_person_hands_the_budget_back(self):
        """Somebody asking for something new is the clearest signal available
        that the previous collaboration is over."""
        p = policy()
        self._spend(p, BOT_TURN_BUDGET)
        self.assertFalse(p.evaluate(trigger(n=998)).should_respond)

        p.record_reply(trigger(author=HUMAN, n=500), now=time.time() + 20_000)

        self.assertTrue(
            p.evaluate(trigger(n=999), now=time.time() + 30_000).should_respond
        )

    def test_the_budget_is_per_conversation(self):
        p = policy()
        self._spend(p, BOT_TURN_BUDGET, conversation="conv-1")
        decision = p.evaluate(
            trigger(n=999, conversation="conv-2"), now=time.time() + 10_000
        )
        self.assertTrue(decision.should_respond)

    def test_a_persons_replies_do_not_consume_it(self):
        p = policy()
        now = time.time()
        for i in range(BOT_TURN_BUDGET + 5):
            p.record_reply(trigger(author=HUMAN, n=i), now=now + i * 10)
        self.assertTrue(
            p.evaluate(trigger(n=999), now=now + 10_000).should_respond
        )


class HourlyCapTests(SimpleTestCase):

    def test_the_cap_is_transient_so_the_conversation_can_resume(self):
        """This is the bug that made "stopped for an hour" mean "stopped".

        The caller records a dedupe key for terminal refusals only; recording
        this one marked the message handled forever, so when the hour rolled
        off there was nothing left unanswered and nobody spoke again.
        """
        p = policy(allow=False, max_replies_per_hour=2)
        now = time.time()
        for i in range(2):
            p.record_reply(trigger(author=HUMAN, n=i), now=now + i * 10)

        decision = p.evaluate(trigger(author=HUMAN, n=99), now=now + 60)

        self.assertIs(decision.verdict, Verdict.IGNORE)
        self.assertTrue(decision.transient)

    def test_collaborating_bots_are_bounded_by_the_budget_not_the_cap(self):
        p = policy(allow=True, max_replies_per_hour=2)
        now = time.time()
        for i in range(2):
            p.record_reply(trigger(n=i), now=now + i * 10)

        self.assertTrue(p.evaluate(trigger(n=99), now=now + 60).should_respond)


class ExchangeTests(SimpleTestCase):
    """The whole point, simulated end to end."""

    def _run(self, allow, latency, limit=300):
        """Returns (turns completed, why it stopped)."""
        pols = {
            BOT_A: AddressedOnlyPolicy(
                BotIdentity(BOT_A, "alpha"),
                store=InMemoryPolicyStore(),
                is_bot_author=is_bot,
                allow_bot_conversations=allow,
            ),
            BOT_B: AddressedOnlyPolicy(
                BotIdentity(BOT_B, "beta"),
                store=InMemoryPolicyStore(),
                is_bot_author=is_bot,
                allow_bot_conversations=allow,
            ),
        }
        now, speaker, listener, turns, n = time.time(), BOT_A, BOT_B, 0, 0
        while turns < limit:
            n += 1
            p = pols[listener]
            t = trigger(author=speaker, n=n)
            decision = p.evaluate(t, now=now)
            if not decision.transient:
                p.record_seen(t.dedupe_key)
            if not decision.should_respond:
                return turns, decision.reason
            # The delay is a wait, not a skip - the turn still happens.
            now += decision.delay
            p.record_reply(t, now=now)
            turns += 1
            now += latency
            speaker, listener = listener, speaker
        return turns, "still going"

    def test_the_old_behaviour_died_in_minutes(self):
        turns, reason = self._run(allow=False, latency=3.0)
        self.assertIn("off for this bot", reason)
        self.assertEqual(turns, 0)

    def test_a_collaboration_runs_far_past_the_old_hourly_cap(self):
        """Sixty turns was where this used to stop, about three minutes in."""
        turns, reason = self._run(allow=True, latency=3.0)
        self.assertGreater(turns, 60)
        self.assertIn("budget spent", reason)

    def test_a_fast_exchange_is_paced_rather_than_killed(self):
        """At one second apart the cooldown used to end it after two turns."""
        turns, reason = self._run(allow=True, latency=1.0)
        self.assertGreater(turns, 60)
        self.assertIn("budget spent", reason)

    def test_it_ends_on_the_budget_rather_than_running_forever(self):
        turns, reason = self._run(allow=True, latency=3.0)
        self.assertIn("budget spent", reason)
        # Two bots, each with its own budget.
        self.assertLessEqual(turns, BOT_TURN_BUDGET * 2 + 1)


class ApiTests(TestCase):
    """The toggle is reachable, defaults off, and survives a round trip."""

    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()
        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")
        self.client_acme = client_for(self.alice, self.org)

        service = Service.objects.create(name="OpenAI")
        self.model = Model.objects.create(service=service, model="gpt-4o-mini")
        role = Role.objects.create(
            organization=self.org, name="S", system_prompt="Be helpful."
        )
        self.agent = Agent.objects.create(
            organization=self.org, name="H", slug="h", role=role
        )

    def tearDown(self):
        crypto._cipher = None

    def _bot(self, handle="helper"):
        bot, _ = mint_bot(
            organization=self.org,
            created_by=self.alice,
            name="Helper",
            handle=handle,
            owner_entity_id=self.alice.entity_id,
            agent=self.agent,
            model=self.model,
        )
        return bot

    def test_a_new_bot_will_not_talk_to_bots(self):
        """Off by default. Two bots that answer each other never stop on their
        own, so this is not a default anybody should get by accident."""
        bot = self._bot()
        self.assertFalse(bot.allow_bot_conversations)

    def test_it_can_be_switched_on(self):
        bot = self._bot()
        response = self.client_acme.patch(
            reverse("api-chatterloop:bot-detail", args=[str(bot.pk)]),
            {"allow_bot_conversations": True},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["data"]["allow_bot_conversations"])
        bot.refresh_from_db()
        self.assertTrue(bot.allow_bot_conversations)

    def test_it_can_be_switched_off_again(self):
        """How you stop a collaboration that is running."""
        bot = self._bot()
        bot.allow_bot_conversations = True
        bot.save(update_fields=["allow_bot_conversations"])

        self.client_acme.patch(
            reverse("api-chatterloop:bot-detail", args=[str(bot.pk)]),
            {"allow_bot_conversations": False},
            format="json",
        )
        bot.refresh_from_db()
        self.assertFalse(bot.allow_bot_conversations)

    def test_changing_it_leaves_everything_else_alone(self):
        bot = self._bot()
        self.client_acme.patch(
            reverse("api-chatterloop:bot-detail", args=[str(bot.pk)]),
            {"allow_bot_conversations": True},
            format="json",
        )
        bot.refresh_from_db()
        self.assertEqual(bot.agent_id, self.agent.pk)
        self.assertEqual(bot.model_id, self.model.pk)
        self.assertTrue(bot.is_online)
        self.assertEqual(bot.status, ChatterloopBot.STATUS_ACTIVE)
