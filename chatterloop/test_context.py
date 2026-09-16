"""Merging the two histories a bot answers from.

The flat window and the reply lineage overlap, contradict each other on order,
and each matters in a case the other cannot cover. These assert the merge, not
the fetching - the reads are one API call each and fail independently.
"""

from django.test import SimpleTestCase

from chatterloop.context import CHAIN_LIMIT, RECENT_LIMIT, build, merge
from chatterloop.runtime import BotIdentity

BOT = "entity-bot"


def message(message_id, at, content="text", sender="entity-human"):
    return {
        "message_id": message_id,
        "created_at": at,
        "content": content,
        "sender_entity_id": sender,
    }


class MergeTests(SimpleTestCase):

    def test_both_histories_appear(self):
        recent = [message("r1", 300), message("r2", 400)]
        chain = [message("c1", 100), message("c2", 200)]

        merged = merge(recent, chain)

        self.assertEqual([m["message_id"] for m in merged], ["c1", "c2", "r1", "r2"])

    def test_an_overlap_is_not_shown_twice(self):
        """A reply to something recent appears in BOTH histories, and a model
        shown the same turn twice treats it as having been said twice."""
        shared = message("m2", 200)
        merged = merge([shared, message("m3", 300)], [message("m1", 100), shared])

        self.assertEqual([m["message_id"] for m in merged], ["m1", "m2", "m3"])

    def test_an_old_ancestor_sorts_to_its_real_position(self):
        """Chronological, not chain-first. An ancestor from far back reads as
        the beginning of the thread because that is where it happened."""
        merged = merge([message("r1", 9000)], [message("ancient", 5)])

        self.assertEqual([m["message_id"] for m in merged], ["ancient", "r1"])

    def test_the_triggering_message_is_left_out(self):
        """It is passed to the model as the question. Included here it is asked
        twice, and the model answers the turn before it."""
        merged = merge([message("m1", 100), message("m2", 200)], [], exclude_id="m2")

        self.assertEqual([m["message_id"] for m in merged], ["m1"])

    def test_a_message_with_no_id_is_dropped_rather_than_fatal(self):
        merged = merge([{"created_at": 1, "content": "x"}, message("m1", 2)], [])

        self.assertEqual([m["message_id"] for m in merged], ["m1"])

    def test_a_missing_timestamp_sorts_to_the_start(self):
        """0 means unparseable, and the platform's own rule is that it sorts to
        the start of history - never letting a parsing gap look like the newest
        thing said."""
        merged = merge([message("m1", 500)], [{"message_id": "m0", "content": "x"}])

        self.assertEqual(merged[0]["message_id"], "m0")


class BuildTests(SimpleTestCase):

    def setUp(self):
        self.identity = BotIdentity(BOT, "neon")

    def test_the_bots_own_turns_are_marked_assistant(self):
        turns = build(
            [message("m1", 100, "a question"),
             message("m2", 200, "an answer", sender=BOT)],
            [],
            identity=self.identity,
        )

        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])

    def test_without_an_identity_nothing_is_claimed_as_the_bots_own(self):
        """A model told it said something it did not will build on the
        invention, so the safe direction is to claim nothing."""
        turns = build([message("m1", 100, "x", sender=BOT)], [])

        self.assertEqual(turns[0]["role"], "user")

    def test_empty_messages_are_skipped(self):
        turns = build([message("m1", 100, "   "), message("m2", 200, "real")], [])

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["content"], "real")

    def test_each_history_is_trimmed_before_merging(self):
        """Trimming after the merge lets whichever history happens to be longer
        crowd the other out entirely."""
        recent = [message(f"r{i}", 1000 + i) for i in range(RECENT_LIMIT * 3)]
        chain = [message(f"c{i}", i) for i in range(CHAIN_LIMIT * 3)]

        turns = build(recent, chain, identity=self.identity)

        self.assertEqual(len(turns), RECENT_LIMIT + CHAIN_LIMIT)

    def test_content_is_sent_unlabelled(self):
        """No "History:" prefix. Prefixing every turn taught the model to copy
        it, and replies went out starting with the word."""
        turns = build([message("m1", 100, "hello")], [])

        self.assertEqual(turns[0]["content"], "hello")

    def test_nothing_at_all_is_not_an_error(self):
        self.assertEqual(build([], []), [])
