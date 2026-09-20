"""Bot-category commands arriving on a `messages_list` frame.

chatterloop runs `system` and `webhook` commands itself and never tells us. A
`bot` one is delivered ONLY as a field on the frame - no queue, no delivery
receipt, no second channel - so everything here is about what the runtime does
with that field.
"""

from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from chatterloop.frames import parse_envelope, parse_messages_list
from chatterloop.mentions import strip_command
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource

from .test_runtime import (
    BOT_ENTITY,
    HUMAN_ENTITY,
    FakeApi,
    build_runtime,
    envelope,
    message,
)


def command_frame(name, target="", sender=HUMAN_ENTITY, mentioner=None):
    body = {
        "conversationID": "conv-1",
        "entityID": sender,
        "command": {"name": name, "target": target or None},
    }
    if mentioner is not None:
        body["mentioner"] = mentioner
    return envelope(body=body)


class CommandFrameParsingTests(SimpleTestCase):
    def test_a_command_is_read_off_the_frame(self):
        payload = parse_messages_list(parse_envelope(command_frame("summarize", "neon")))

        self.assertIsNotNone(payload.command)
        self.assertEqual(payload.command.name, "summarize")
        self.assertEqual(payload.command.target, "neon")

    def test_no_command_field_is_not_a_command(self):
        payload = parse_messages_list(parse_envelope(command_frame("x")))
        self.assertIsNotNone(payload.command)

        plain = parse_messages_list(
            parse_envelope(envelope(body={"conversationID": "c", "entityID": "e"}))
        )
        self.assertIsNone(plain.command)

    def test_a_nameless_command_is_not_a_command(self):
        """The field is another service's, so an empty name is guarded rather
        than trusted - it would otherwise match an arsenal entry by accident."""
        payload = parse_messages_list(
            parse_envelope(
                envelope(body={"conversationID": "c", "entityID": "e", "command": {"name": ""}})
            )
        )
        self.assertIsNone(payload.command)

    def test_the_name_and_target_are_lowercased(self):
        payload = parse_messages_list(parse_envelope(command_frame("Summarize", "Neon")))

        self.assertEqual(payload.command.name, "summarize")
        self.assertEqual(payload.command.target, "neon")


class StripCommandTests(SimpleTestCase):
    def test_the_token_goes_and_the_arguments_stay(self):
        self.assertEqual(
            strip_command("/summarize the pricing thread"), "the pricing thread"
        )

    def test_a_target_is_removed_too(self):
        self.assertEqual(strip_command("/summarize:neon  what about Q4"), "what about Q4")

    def test_a_bare_command_leaves_nothing(self):
        self.assertEqual(strip_command("/members"), "")

    def test_the_escape_hatch_is_left_alone(self):
        """"//summarize" is how somebody writes the word rather than runs it."""
        self.assertEqual(strip_command("//summarize literally"), "//summarize literally")

    def test_a_slash_inside_a_word_is_not_a_command(self):
        """"and/or" has no whitespace before its slash, so it is a word."""
        self.assertEqual(strip_command("see and/or the docs"), "see and/or the docs")
        self.assertEqual(strip_command("/api/v1/users"), "/api/v1/users")

    def test_a_command_after_text_is_still_a_command(self):
        """The grammar is no longer anchored to the start - see
        commandParser.js. "@juanlazy /summarize the thread" is the case people
        type, and the arguments are what follows the token."""
        self.assertEqual(
            strip_command("@juanlazy /summarize the thread"),
            "@juanlazy the thread",
        )
        self.assertEqual(
            strip_command("use the /summarize command"), "use the command"
        )

    def test_only_the_first_command_goes(self):
        """One command per message, matching the parser - a second one further
        in is somebody writing about it."""
        self.assertEqual(
            strip_command("/stop and then /summarize"), "and then /summarize"
        )


class CommandRoutingTests(SimpleTestCase):
    def test_a_declared_command_is_answered(self):
        api = FakeApi(messages=[message(content="/summarize the pricing thread")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("summarize"))

        self.assertEqual(len(dispatched), 1)
        trigger = dispatched[0]
        self.assertEqual(trigger.reason, TriggerReason.COMMAND)
        self.assertEqual(trigger.command_name, "summarize")
        self.assertEqual(trigger.query, "the pricing thread")

    def test_a_command_the_bot_does_not_declare_is_ignored(self):
        """A command frame reaches EVERY bot in the conversation. A bot that
        answered one it does not implement would speak up about something it
        cannot do - and in a room with several bots, all of them would."""
        api = FakeApi(messages=[message(content="/transcribe that clip")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("transcribe"))

        self.assertEqual(dispatched, [])
        # It FELL THROUGH rather than being dropped: an unknown command is
        # still an ordinary message, and in a DM it deserves an ordinary
        # answer. What it must not become is a COMMAND trigger.
        self.assertNotIn("fetch_messages", api.calls)

    def test_a_command_targeted_at_another_bot_is_ignored(self):
        api = FakeApi(messages=[message(content="/summarize:other the thread")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("summarize", target="other"))

        self.assertEqual(dispatched, [])

    def test_a_command_targeted_at_this_bot_is_answered(self):
        api = FakeApi(messages=[message(content="/summarize:helper the thread")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("summarize", target="helper"))

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].command_target, "helper")

    def test_a_bare_command_still_counts_as_resolved(self):
        """Its arguments are empty, which is not the same as unreadable.
        Judging it on `query` would drop every command that takes none."""
        api = FakeApi(messages=[message(content="/members")])
        runtime, dispatched = build_runtime(api, commands=["members"])

        runtime.handle_envelope(command_frame("members"))

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].query, "")

    def test_the_bots_own_command_is_never_answered(self):
        api = FakeApi(messages=[message(sender=BOT_ENTITY, content="/summarize x")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("summarize", sender=BOT_ENTITY))

        self.assertEqual(dispatched, [])
        self.assertEqual(api.calls, [])

    def test_an_unknown_command_that_also_mentions_falls_through_to_mention(self):
        """"/unknown @helper what is this" did still name the bot."""
        api = FakeApi(messages=[message(content="/unknown @helper what is this")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(
            command_frame("unknown", mentioner={"entityID": HUMAN_ENTITY, "username": "@human"})
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].reason, TriggerReason.MENTION)

    def test_a_command_wins_over_a_mention_on_the_same_message(self):
        api = FakeApi(messages=[message(content="/summarize:helper @helper please")])
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(
            command_frame(
                "summarize",
                target="helper",
                mentioner={"entityID": HUMAN_ENTITY, "username": "@human"},
            )
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].reason, TriggerReason.COMMAND)
        # Both the token AND the address are gone from what reaches retrieval.
        self.assertEqual(dispatched[0].query, "please")

    def test_the_wrong_message_is_not_answered(self):
        """Between the command being typed and this fetch landing, the same
        person may well have typed something else."""
        api = FakeApi(
            messages=[
                message(message_id="m1", content="/summarize the pricing thread"),
                message(message_id="m2", content="actually never mind"),
            ]
        )
        runtime, dispatched = build_runtime(api, commands=["summarize"])

        runtime.handle_envelope(command_frame("summarize"))

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].message_id, "m1")

    def test_no_arsenal_means_no_command_is_ever_ours(self):
        """The default. A bot that has declared nothing must not wake up for
        somebody else's command."""
        api = FakeApi(messages=[message(content="/summarize x")])
        runtime, dispatched = build_runtime(api)

        runtime.handle_envelope(command_frame("summarize"))

        self.assertEqual(dispatched, [])


class CommandTriggerTests(SimpleTestCase):
    def test_a_command_trigger_survives_the_queue(self):
        """Celery serialises arguments, so the new fields have to round-trip."""
        trigger = Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.COMMAND,
            author_entity_id=HUMAN_ENTITY,
            conversation_id="conv-1",
            command_name="summarize",
            command_target="helper",
            text="/summarize:helper x",
            query="x",
        )

        restored = Trigger.from_payload(trigger.to_payload())

        self.assertEqual(restored.reason, TriggerReason.COMMAND)
        self.assertEqual(restored.command_name, "summarize")
        self.assertEqual(restored.command_target, "helper")

class ArsenalTests(SimpleTestCase):
    """What a bot declares, read from chatterloop's bot_commands."""

    def test_only_the_bot_category_is_the_arsenal(self):
        """chatterloop runs `system` and `webhook` itself and never tells us -
        a `bot` command is the only kind Neon is responsible for."""
        from chatterloop import commands as module

        bot = SimpleNamespace(bot_id="bot-1", handle="neon")
        seen = {}

        class FakeModel:
            class objects:
                @staticmethod
                def filter(**kwargs):
                    seen.update(kwargs)

                    class QS:
                        @staticmethod
                        def values_list(field, flat=False):
                            return ["Summarize", "transcribe"]

                    return QS

        with mock.patch.object(
            module, "arsenal_for", module.arsenal_for
        ), mock.patch.dict(
            "sys.modules",
            {"chatterloop.external_models": SimpleNamespace(BotCommand=FakeModel)},
        ):
            names = module.arsenal_for(bot)

        self.assertEqual(names, frozenset({"summarize", "transcribe"}))
        self.assertEqual(seen["category"], "bot")
        self.assertIs(seen["is_active"], True)
        self.assertEqual(seen["bot_id"], "bot-1")

    def test_a_bot_with_no_id_has_no_arsenal(self):
        from chatterloop import commands as module

        self.assertEqual(
            module.arsenal_for(SimpleNamespace(bot_id="", handle="x")),
            frozenset(),
        )

    def test_a_failed_read_leaves_the_bot_mention_only(self):
        """Never fatal. A supervisor that refuses to start a bot because one
        read failed is worse than a bot that answers mentions only."""
        from chatterloop import commands as module

        class Exploding:
            class objects:
                @staticmethod
                def filter(**kwargs):
                    raise RuntimeError("chatterloop is unreachable")

        with mock.patch.dict(
            "sys.modules",
            {"chatterloop.external_models": SimpleNamespace(BotCommand=Exploding)},
        ):
            names = module.arsenal_for(SimpleNamespace(bot_id="b", handle="neon"))

        self.assertEqual(names, frozenset())

class BatchedArsenalTests(SimpleTestCase):
    """One query for every bot, not one per bot.

    The sweep refreshes each running bot every interval, so a per-bot read is
    O(bots) round trips against another service's database on a timer.
    """

    def test_one_query_covers_every_bot(self):
        from chatterloop import commands as module

        seen = {}

        class FakeModel:
            class objects:
                @staticmethod
                def filter(**kwargs):
                    seen.update(kwargs)

                    class QS:
                        @staticmethod
                        def values_list(*fields):
                            return [
                                ("bot-1", "Summarize"),
                                ("bot-1", "transcribe"),
                                ("bot-2", "digest"),
                            ]

                    return QS

        bots = [
            SimpleNamespace(bot_id="bot-1"),
            SimpleNamespace(bot_id="bot-2"),
            SimpleNamespace(bot_id="bot-3"),
        ]

        with mock.patch.dict(
            "sys.modules",
            {"chatterloop.external_models": SimpleNamespace(BotCommand=FakeModel)},
        ):
            arsenals = module.arsenals_for(bots)

        self.assertEqual(arsenals["bot-1"], frozenset({"summarize", "transcribe"}))
        self.assertEqual(arsenals["bot-2"], frozenset({"digest"}))
        # A bot that declares nothing simply has no entry - callers default.
        self.assertNotIn("bot-3", arsenals)
        self.assertEqual(sorted(seen["bot_id__in"]), ["bot-1", "bot-2", "bot-3"])
        self.assertEqual(seen["category"], "bot")

    def test_no_bots_asks_nothing(self):
        from chatterloop import commands as module

        self.assertEqual(module.arsenals_for([]), {})

    def test_a_failed_read_leaves_every_bot_mention_only(self):
        from chatterloop import commands as module

        class Exploding:
            class objects:
                @staticmethod
                def filter(**kwargs):
                    raise RuntimeError("chatterloop is unreachable")

        with mock.patch.dict(
            "sys.modules",
            {"chatterloop.external_models": SimpleNamespace(BotCommand=Exploding)},
        ):
            self.assertEqual(
                module.arsenals_for([SimpleNamespace(bot_id="b")]), {}
            )

