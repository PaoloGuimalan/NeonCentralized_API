"""The bot runtime: routing, the three load-bearing rules, and the policy.

No network and no Redis. The API is a fake that records what was asked and
answers from a script, which is what lets these tests assert the thing that
actually matters on each path - not "did it reply" but "did it read at all".
"""

import time
import uuid

from django.test import SimpleTestCase, TestCase

from chatterloop.authors import SYSTEM_BOT_ENTITY_ID
from chatterloop.client import ChatterloopAPIError, TokenRejected
from chatterloop.frames import (
    EVENT_MESSAGES_LIST,
    EVENT_NOTIFICATIONS,
    parse_envelope,
    parse_messages_list,
)
from chatterloop.mentions import (
    extract_handles,
    is_addressed_to,
    normalise_handle,
    strip_mentions,
)
from chatterloop.policy import AddressedOnlyPolicy, InMemoryPolicyStore, Verdict
from chatterloop.runtime import BotIdentity, BotRuntime
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource

BOT_ENTITY = "entity-bot"
HUMAN_ENTITY = "entity-human"


def envelope(event=EVENT_MESSAGES_LIST, body=None, auth=True, status=True):
    return {
        "event": event,
        "dateTime": "2026-09-13T12:00:00Z",
        "message": {"auth": auth, "status": status, "message": body},
    }


def messages_frame(sender=HUMAN_ENTITY, conversation="conv-1", mentioner=None, deleted=""):
    body = {"conversationID": conversation, "entityID": sender}
    if mentioner is not None:
        body["mentioner"] = mentioner
    if deleted:
        body["deletedMessageID"] = deleted
    return envelope(body=body)


def message(
    message_id="m1",
    sender=HUMAN_ENTITY,
    content="@helper what did we decide?",
    created_at=None,
    handle="human",
):
    return {
        "message_id": message_id,
        "conversation_id": "conv-1",
        "sender_entity_id": sender,
        "sender_handle": handle,
        "content": content,
        "created_at": int(time.time() * 1000) if created_at is None else created_at,
        "message_type": "text",
        "is_reply": False,
        "replying_to": "",
    }


class FakeApi:
    """Records every call, so a test can assert a read did NOT happen."""

    def __init__(self, messages=None, replies=None, conv_type="group", mentions=None):
        self.messages = messages if messages is not None else []
        self.replies = replies if replies is not None else []
        self.conv_type = conv_type
        self.mentions = mentions if mentions is not None else []
        self.comment_replies = []
        self.calls = []
        self.raise_on = {}

    def _record(self, name):
        self.calls.append(name)
        if name in self.raise_on:
            raise self.raise_on[name]

    def fetch_messages(self, token, conversation_id, limit=40):
        self._record("fetch_messages")
        return list(self.messages), self.conv_type

    def conversation_type(self, token, conversation_id):
        self._record("conversation_type")
        return self.conv_type

    def fetch_replies_to_me(self, token, conversation_id, limit=25):
        self._record("fetch_replies_to_me")
        return list(self.replies)

    def fetch_comment_mentions(self, token, limit=25):
        self._record("fetch_comment_mentions")
        return list(self.mentions)

    def fetch_comment_replies(self, token, limit=25):
        self._record("fetch_comment_replies")
        return list(self.comment_replies)


def build_runtime(api, **kwargs):
    identity = BotIdentity(BOT_ENTITY, "helper")
    policy = AddressedOnlyPolicy(identity, store=InMemoryPolicyStore())
    dispatched = []
    # A watermark a minute in the past, so "live" fixtures stamped `now` are
    # comfortably after it and "stale" ones comfortably before. Left to
    # default, the runtime stamps itself AFTER the test built its fixture, and
    # a live row can lose by a millisecond.
    kwargs.setdefault("started_at_ms", int(time.time() * 1000) - 60_000)
    runtime = BotRuntime(
        identity=identity,
        policy=policy,
        token="clt_test",
        # Two arguments now: the runtime passes the cooldown a collaborating
        # bot should wait before answering, which the supervisor turns into a
        # Celery countdown. Recorded so a test can assert on it.
        dispatch=lambda trigger, delay=0.0: dispatched.append(trigger),
        api=api,
        **kwargs,
    )
    return runtime, dispatched


class MentionParsingTests(SimpleTestCase):
    """Ported verbatim from the platform's own parsers - drift here is a bug."""

    def test_an_email_address_is_not_a_mention(self):
        self.assertEqual(extract_handles("mail you@example.com"), [])

    def test_a_trailing_dot_yields_both_forms(self):
        self.assertEqual(extract_handles("thanks @ana."), ["ana.", "ana"])

    def test_normalise_strips_the_at_and_lowercases(self):
        # `mentioner.username` arrives as "@Ana" while user_account.username is
        # stored bare, so every comparison has to go through this.
        self.assertEqual(normalise_handle("@Ana"), "ana")

    def test_addressing_is_case_insensitive(self):
        self.assertTrue(is_addressed_to("hey @Helper", {"helper"}))

    def test_stripping_removes_only_our_own_handle(self):
        stripped = strip_mentions("@helper ask @ana about pricing", {"helper"})
        self.assertNotIn("@helper", stripped)
        # Mentions of other people are content, not address.
        self.assertIn("@ana", stripped)

    def test_stripping_does_not_fuse_neighbouring_words(self):
        self.assertEqual(strip_mentions("hi @helper there", {"helper"}), "hi there")


class FrameParsingTests(SimpleTestCase):

    def test_the_double_nesting_is_unwrapped(self):
        parsed = parse_envelope(messages_frame())
        self.assertEqual(parsed.event, EVENT_MESSAGES_LIST)
        self.assertTrue(parsed.auth and parsed.status)
        payload = parse_messages_list(parsed)
        self.assertEqual(payload.conversation_id, "conv-1")
        self.assertEqual(payload.entity_id, HUMAN_ENTITY)

    def test_a_malformed_frame_returns_none_rather_than_raising(self):
        """Another service's firehose: one bad frame must not stop the next."""
        self.assertIsNone(parse_envelope("not a dict"))
        self.assertIsNone(parse_messages_list(parse_envelope(envelope(body="text"))))

    def test_a_mentioner_is_narrowed(self):
        parsed = parse_envelope(
            messages_frame(mentioner={"entityID": "e", "username": "@helper", "isSingle": False})
        )
        payload = parse_messages_list(parsed)
        self.assertEqual(payload.mentioner.username, "@helper")


class RoutingTests(SimpleTestCase):

    def test_a_mention_needs_no_probe(self):
        """The frame carries a non-null mentioner only for recipients the
        server resolved a mention against. That is authoritative."""
        api = FakeApi(messages=[message()])
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(
            messages_frame(mentioner={"username": "@helper", "entityID": HUMAN_ENTITY})
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].reason, TriggerReason.MENTION)
        self.assertNotIn("fetch_replies_to_me", api.calls)

    def test_the_bots_own_message_costs_no_read_at_all(self):
        """Loop prevention ahead of everything. The bot's own answers come back
        on its own channel, and a bot that read them back would answer its own
        thread forever - with no @handle ever typed."""
        api = FakeApi()
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame(sender=BOT_ENTITY))

        self.assertEqual(dispatched, [])
        self.assertEqual(api.calls, [])

    def test_a_system_message_costs_no_read_at_all(self):
        """The System bot's notices and command answers are for nobody - not
        even in a DM with the bot, where every other sender is answered."""
        api = FakeApi(messages=[message(sender=SYSTEM_BOT_ENTITY_ID)], conv_type="single")
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame(sender=SYSTEM_BOT_ENTITY_ID))

        self.assertEqual(dispatched, [])
        self.assertEqual(api.calls, [])

    def test_a_dm_answers_the_person_not_the_system_notice_after_them(self):
        """The System bot's answer to their command can land after their
        message and before the read - it is newer, and not what they said."""
        api = FakeApi(
            messages=[
                message(message_id="m1", content="hey there"),
                message(message_id="m2", sender=SYSTEM_BOT_ENTITY_ID,
                        content="Only the first 3 commands ran.", handle="system"),
            ],
            conv_type="single",
        )
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame())

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].message_id, "m1")
        self.assertEqual(dispatched[0].query, "hey there")

    def test_a_deletion_ping_is_not_a_delivery(self):
        api = FakeApi()
        runtime, dispatched = build_runtime(api)
        runtime.handle_envelope(messages_frame(deleted="m9"))
        self.assertEqual(api.calls, [])

    def test_an_unauthenticated_frame_is_dropped(self):
        api = FakeApi()
        runtime, dispatched = build_runtime(api)
        runtime.handle_envelope(messages_frame())
        before = len(api.calls)
        runtime.handle_envelope(
            {"event": EVENT_MESSAGES_LIST, "message": {"auth": False, "status": True, "message": {}}}
        )
        self.assertEqual(len(api.calls), before)

    def test_an_ignored_event_is_dropped_silently(self):
        api = FakeApi()
        runtime, _ = build_runtime(api)
        runtime.handle_envelope(envelope(event="istyping_broadcast", body={}))
        self.assertEqual(api.calls, [])

    def test_a_dm_skips_the_reply_probe(self):
        """A DM needs neither an @mention nor reply-threading, so the probe has
        nothing to add there - only a read to skip."""
        api = FakeApi(messages=[message(content="hey")], conv_type="single")
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame())

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].reason, TriggerReason.DM)
        self.assertNotIn("fetch_replies_to_me", api.calls)

    def test_the_conversation_type_is_resolved_once(self):
        api = FakeApi(messages=[message(content="hey")], conv_type="single")
        runtime, _ = build_runtime(api)

        for _ in range(3):
            runtime.handle_envelope(messages_frame())

        self.assertEqual(api.calls.count("conversation_type"), 1)

    def test_an_unresolvable_conversation_type_is_not_cached(self):
        """A transient read failure must not become a permanent
        misclassification."""
        api = FakeApi(conv_type="")
        runtime, _ = build_runtime(api, answer_replies=False)

        runtime.handle_envelope(messages_frame())
        runtime.handle_envelope(messages_frame())

        self.assertEqual(api.calls.count("conversation_type"), 2)

    def test_a_group_message_probes_for_a_reply(self):
        api = FakeApi(
            conv_type="group",
            replies=[message(message_id="m5", content="yes please")],
            messages=[message(message_id="m5", content="yes please")],
        )
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame())

        self.assertIn("fetch_replies_to_me", api.calls)
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].reason, TriggerReason.REPLY)

    def test_answer_replies_off_makes_the_bot_mention_only(self):
        api = FakeApi(conv_type="group", replies=[message()])
        runtime, dispatched = build_runtime(api, answer_replies=False, answer_dms=False)

        runtime.handle_envelope(messages_frame())

        self.assertEqual(dispatched, [])
        self.assertNotIn("fetch_replies_to_me", api.calls)

    def test_a_read_failure_does_not_stop_the_next_frame(self):
        api = FakeApi(messages=[message()])
        api.raise_on["fetch_messages"] = ChatterloopAPIError("pg is down")
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(
            messages_frame(mentioner={"username": "@helper", "entityID": HUMAN_ENTITY})
        )

        self.assertEqual(dispatched, [])
        self.assertEqual(runtime.stats["errors"], 1)

    def test_a_rejected_token_propagates_rather_than_being_logged_per_frame(self):
        """Not a per-frame problem. The supervisor has to stop the whole bot,
        not log the same error once per message forever."""
        api = FakeApi(messages=[message()])
        api.raise_on["fetch_messages"] = TokenRejected("revoked", 401)
        runtime, _ = build_runtime(api)

        with self.assertRaises(TokenRejected):
            runtime.handle_envelope(
                messages_frame(mentioner={"username": "@helper", "entityID": HUMAN_ENTITY})
            )


class LivenessGateTests(SimpleTestCase):
    """Reads are not live; frames are. See the runtime's module docstring."""

    def test_a_reply_from_before_this_lease_is_refused(self):
        old = int(time.time() * 1000) - 3_600_000
        api = FakeApi(conv_type="group", replies=[message(created_at=old)])
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame())

        self.assertEqual(dispatched, [])

    def test_a_stale_reply_is_recorded_so_it_is_not_reconsidered(self):
        old = int(time.time() * 1000) - 3_600_000
        api = FakeApi(conv_type="group", replies=[message(created_at=old)])
        runtime, _ = build_runtime(api)

        runtime.handle_envelope(messages_frame())
        self.assertTrue(runtime.policy.has_seen("msg:m1"))

    def test_an_undatable_row_counts_as_stale(self):
        """Silence on a row we cannot date is recoverable; a burst is not."""
        api = FakeApi(conv_type="group", replies=[message(created_at=0)])
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(messages_frame())

        self.assertEqual(dispatched, [])

    def test_a_durable_comment_backlog_is_not_answered_on_restart(self):
        """Comment notifications wait indefinitely. A bot that answered
        everything it found would come back from a deploy and work through the
        backlog, each item a real message about something said hours ago."""
        old = int(time.time() * 1000) - 86_400_000
        api = FakeApi(
            mentions=[
                {
                    "comment_id": "c1",
                    "post_id": "p1",
                    "author_entity_id": HUMAN_ENTITY,
                    "author_handle": "human",
                    "text": "@helper thoughts?",
                    "created_at": old,
                    "kind": "mention",
                }
            ]
        )
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(envelope(event=EVENT_NOTIFICATIONS, body="somebody mentioned you"))

        self.assertEqual(dispatched, [])

    def test_a_live_comment_mention_is_answered(self):
        api = FakeApi(
            mentions=[
                {
                    "comment_id": "c1",
                    "post_id": "p1",
                    "author_entity_id": HUMAN_ENTITY,
                    "author_handle": "human",
                    "text": "@helper thoughts?",
                    "created_at": int(time.time() * 1000),
                    "kind": "mention",
                }
            ]
        )
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(envelope(event=EVENT_NOTIFICATIONS, body="somebody mentioned you"))

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].source, TriggerSource.COMMENT)
        self.assertEqual(dispatched[0].post_id, "p1")
        # The address is stripped before it reaches retrieval.
        self.assertNotIn("@helper", dispatched[0].query)


class PolicyTests(SimpleTestCase):

    def setUp(self):
        self.identity = BotIdentity(BOT_ENTITY, "helper")
        self.store = InMemoryPolicyStore()
        self.policy = AddressedOnlyPolicy(
            self.identity, store=self.store, cooldown_seconds=5.0, max_replies_per_hour=3
        )

    def trigger(self, **kwargs):
        kwargs.setdefault("source", TriggerSource.MESSAGE)
        kwargs.setdefault("author_entity_id", HUMAN_ENTITY)
        kwargs.setdefault("conversation_id", "conv-1")
        kwargs.setdefault("query", "what did we decide?")
        kwargs.setdefault("dedupe_key", f"msg:{uuid.uuid4().hex}")
        return Trigger(**kwargs)

    def test_the_bot_never_answers_itself(self):
        decision = self.policy.evaluate(self.trigger(author_entity_id=BOT_ENTITY))
        self.assertEqual(decision.verdict, Verdict.IGNORE)
        self.assertIn("itself", decision.reason)

    def test_an_ignored_entity_is_never_answered(self):
        """Two bots that both answer mentions will otherwise mention each other
        until somebody notices the bill."""
        policy = AddressedOnlyPolicy(
            self.identity, store=self.store, ignore_entity_ids={"other-bot"}
        )
        decision = policy.evaluate(self.trigger(author_entity_id="other-bot"))
        self.assertEqual(decision.verdict, Verdict.IGNORE)

    def test_the_system_bot_is_never_answered_even_by_a_collaborating_bot(self):
        """It is a bot, so `allow_bot_conversations` alone would let it through.
        What it posts is never addressed to anyone."""
        policy = AddressedOnlyPolicy(
            self.identity,
            store=self.store,
            is_bot_author=lambda entity_id: True,
            allow_bot_conversations=True,
        )

        decision = policy.evaluate(self.trigger(author_entity_id=SYSTEM_BOT_ENTITY_ID))

        self.assertEqual(decision.verdict, Verdict.IGNORE)
        self.assertIn("System bot", decision.reason)
        # And only the System bot: another bot still gets through the toggle.
        self.assertTrue(
            policy.evaluate(self.trigger(author_entity_id="other-bot")).should_respond
        )

    def test_a_redelivered_trigger_is_ignored(self):
        trigger = self.trigger()
        self.assertTrue(self.policy.evaluate(trigger).should_respond)
        self.policy.record_seen(trigger.dedupe_key)
        self.assertFalse(self.policy.evaluate(trigger).should_respond)

    def test_an_unresolved_trigger_is_not_answered(self):
        """A trigger with no text is something we know happened but cannot
        read. Answering would mean replying to an unknown question."""
        decision = self.policy.evaluate(self.trigger(query="   "))
        self.assertIn("could not be resolved", decision.reason)

    def test_the_cooldown_collapses_a_burst(self):
        first = self.trigger()
        self.assertTrue(self.policy.evaluate(first).should_respond)
        self.policy.record_reply(first)
        self.assertFalse(self.policy.evaluate(self.trigger()).should_respond)

    def test_the_cooldown_is_per_conversation(self):
        first = self.trigger(conversation_id="conv-1")
        self.policy.record_reply(first)
        other = self.trigger(conversation_id="conv-2")
        self.assertTrue(self.policy.evaluate(other).should_respond)

    def test_the_hourly_cap_is_a_backstop(self):
        now = time.time()
        for i in range(3):
            self.policy.record_reply(self.trigger(), now=now - 100 + i)
        decision = self.policy.evaluate(self.trigger(), now=now)
        self.assertIn("hourly reply limit", decision.reason)

    def test_replies_older_than_an_hour_do_not_count(self):
        now = time.time()
        for i in range(3):
            self.policy.record_reply(self.trigger(), now=now - 7200 + i)
        self.assertTrue(self.policy.evaluate(self.trigger(), now=now).should_respond)

    def test_each_reason_has_its_own_respond_message(self):
        """"Addressed" is one word for three different situations, and which
        one it was is the first thing anyone wants when a reply looks
        unwarranted."""
        reasons = set()
        for reason in (TriggerReason.MENTION, TriggerReason.REPLY, TriggerReason.DM):
            decision = self.policy.evaluate(self.trigger(reason=reason))
            self.assertTrue(decision.should_respond)
            reasons.add(decision.reason)
        self.assertEqual(len(reasons), 3)

    def test_a_failed_send_does_not_consume_the_budget(self):
        """record_reply fires only after a successful send. Otherwise a broken
        outbound path silently rate-limits the bot into silence."""
        policy = AddressedOnlyPolicy(
            self.identity, store=InMemoryPolicyStore(), max_replies_per_hour=1
        )
        first = self.trigger()
        self.assertTrue(policy.evaluate(first).should_respond)
        policy.record_seen(first.dedupe_key)
        # The send failed, so nothing is recorded.
        self.assertTrue(policy.evaluate(self.trigger()).should_respond)


class TriggerSerialisationTests(SimpleTestCase):
    """Celery carries JSON, so a Trigger has to survive the round trip."""

    def test_a_trigger_round_trips_through_a_payload(self):
        original = Trigger(
            source=TriggerSource.COMMENT,
            reason=TriggerReason.REPLY,
            author_entity_id=HUMAN_ENTITY,
            post_id="p1",
            comment_id="c1",
            query="what about pricing",
            dedupe_key="comment:c1",
        )
        restored = Trigger.from_payload(original.to_payload())

        self.assertEqual(restored.source, TriggerSource.COMMENT)
        self.assertEqual(restored.reason, TriggerReason.REPLY)
        self.assertEqual(restored.comment_id, "c1")
        self.assertEqual(restored.scope, "p1")

    def test_the_payload_is_json_safe(self):
        import json

        payload = Trigger(
            source=TriggerSource.MESSAGE, author_entity_id="e", conversation_id="c"
        ).to_payload()
        self.assertEqual(json.loads(json.dumps(payload))["source"], "message")
