"""What the model ends up reading when a message carries media.

Every rule here decides a line of text a language model acts on, which is the
kind of thing that goes wrong invisibly: nothing throws, the reply is just
about the wrong thing. So these assert on the exact string.

SimpleTestCase throughout - `media` is pure, and a database would only add a
reason for these to fail that has nothing to do with what they check.
"""

from django.test import SimpleTestCase

from chatterloop import media


def record(**overrides):
    row = {
        "target_id": "msg-1",
        "source_type": "message_attachment",
        "content_type": "image",
        "status": "done",
        "text": "",
        "transcription": "",
        "caption": "",
        "shown_text": "",
        "language": "en",
        "is_music": None,
    }
    row.update(overrides)
    return row


def message(message_id="msg-1", message_type="image/jpeg", content="https://cdn.test/a.jpg"):
    return {
        "message_id": message_id,
        "message_type": message_type,
        "content": content,
        "sender_entity_id": "entity-ana",
    }


class NeedsContextTests(SimpleTestCase):
    def test_text_is_not_media(self):
        messages = [message(message_type="text", content="hello")]

        self.assertEqual(media.needs_context(messages), [])

    def test_a_notif_is_not_media(self):
        """Those are strings this platform wrote, not an upload."""
        messages = [message(message_type="notif", content="Ana joined")]

        self.assertEqual(media.needs_context(messages), [])

    def test_every_upload_kind_is_media(self):
        messages = [
            message("a", "image/jpeg"),
            message("b", "video/mp4"),
            message("c", "audio/mpeg"),
            message("d", "image"),
        ]

        self.assertEqual(media.needs_context(messages), ["a", "b", "c", "d"])

    def test_the_same_message_is_asked_about_once(self):
        """The recent window and the reply chain overlap, so the same
        attachment arrives twice - and paying twice for it is the whole reason
        this deduplicates rather than the caller."""
        messages = [message("a"), message("a"), message("b")]

        self.assertEqual(media.needs_context(messages), ["a", "b"])


class DescribeTests(SimpleTestCase):
    def test_a_transcript_is_quoted(self):
        """Somebody's words, not a description of them. A model that cannot
        tell the two apart paraphrases a quote as the platform's own summary."""
        line = media.describe(
            record(content_type="audio", is_music=False,
                   transcription="push the meeting to Thursday")
        )

        self.assertEqual(line, '[voice note] Says: "push the meeting to Thursday"')

    def test_a_caption_and_ocr_read_as_one_line(self):
        line = media.describe(
            record(caption="a whiteboard with a flowchart",
                   shown_text="Q3 roadmap - ship by Nov")
        )

        self.assertEqual(
            line,
            '[image] a whiteboard with a flowchart. Text shown: "Q3 roadmap - ship by Nov"',
        )

    def test_everything_at_once(self):
        line = media.describe(
            record(content_type="video", caption="a person speaking to camera",
                   transcription="we ship on Friday", shown_text="DEMO")
        )

        self.assertEqual(
            line,
            '[video] a person speaking to camera. Says: "we ship on Friday" '
            'Text shown: "DEMO"',
        )

    def test_an_empty_record_still_says_something_is_there(self):
        """"There is an image here and nothing is known about it" is
        information. Silence is not, and a URL is worse than either."""
        self.assertEqual(media.describe(record()), "[image]")

    def test_authored_text_is_used_when_there_is_no_analysis(self):
        """A post's own caption arrives in `text`, not in any media field."""
        line = media.describe(
            record(content_type="text", source_type="post", text="shipping today")
        )

        self.assertEqual(line, "[text] shipping today.")


class StatusTests(SimpleTestCase):
    def test_pending_is_said_out_loud(self):
        """THE POINT OF CARRYING STATUS. Analysis is asynchronous, so a photo
        sent seconds ago is usually pending - and a bot that says so is being
        honest, where one that ignores the attachment looks broken and one
        that guesses is worse."""
        for status in ("pending", "processing"):
            with self.subTest(status=status):
                self.assertEqual(
                    media.describe(record(status=status)),
                    "[image] Still being processed.",
                )

    def test_a_failure_is_distinguishable_from_a_wait(self):
        self.assertEqual(
            media.describe(record(status="failed")), "[image] Could not be read."
        )
        self.assertEqual(
            media.describe(record(status="skipped")), "[image] Not analysed."
        )

    def test_a_pending_record_never_shows_stale_analysis(self):
        """A document can hold text from an earlier attempt. Until it is done,
        that text is not what the caller is being told about."""
        line = media.describe(record(status="pending", transcription="half a sentence"))

        self.assertNotIn("half a sentence", line)


class LabelTests(SimpleTestCase):
    def test_audio_splits_on_whether_it_is_music(self):
        self.assertTrue(media.describe(record(content_type="audio", is_music=False))
                        .startswith("[voice note]"))
        self.assertTrue(media.describe(record(content_type="audio", is_music=True))
                        .startswith("[audio, music]"))

    def test_unclassified_audio_stays_neutral(self):
        """None means nothing classified it, which is not "not music" - and
        guessing would set an expectation about whether there is anything to
        reply to."""
        self.assertTrue(media.describe(record(content_type="audio", is_music=None))
                        .startswith("[audio]"))


class ClipTests(SimpleTestCase):
    def test_a_long_transcript_is_bounded(self):
        """A four-minute video's transcript is longer than the conversation it
        is in, and a turn that is mostly one attachment crowds out the thread
        the bot is answering."""
        line = media.describe(
            record(content_type="video", transcription="word " * 400)
        )

        self.assertLess(len(line), media.TRANSCRIPT_LIMIT + 60)
        self.assertTrue(line.endswith('..."'))

    def test_the_cut_lands_on_a_word(self):
        """A transcript ending mid-word reads as corruption, and a model will
        try to complete it."""
        line = media.describe(record(caption="alpha " * 200))

        self.assertNotIn("alp...", line)


class ApplyTests(SimpleTestCase):
    def test_the_url_is_replaced_by_what_the_media_is(self):
        messages = [message()]

        media.apply(messages, [record(caption="a cat on a keyboard")])

        self.assertEqual(messages[0]["content"], "[image] a cat on a keyboard.")

    def test_a_text_message_is_untouched(self):
        messages = [message(message_type="text", content="hello")]

        media.apply(messages, [record(target_id="msg-1", caption="nonsense")])

        self.assertEqual(messages[0]["content"], "hello")

    def test_media_with_no_record_still_loses_its_url(self):
        """Nothing analysed it, or nothing ever will. Either way the model
        learns more from "there is an image here" than from a URL it cannot
        open - and this is also what happens when the read itself failed."""
        messages = [message(content="https://cdn.test/secret.jpg")]

        media.apply(messages, [])

        self.assertEqual(messages[0]["content"], "[image]")

    def test_a_record_for_another_message_is_not_applied(self):
        messages = [message("msg-1"), message("msg-2")]

        media.apply(messages, [record(target_id="msg-2", caption="only this one")])

        self.assertEqual(messages[0]["content"], "[image]")
        self.assertEqual(messages[1]["content"], "[image] only this one.")

    def test_several_records_for_one_message_are_all_shown(self):
        messages = [message()]

        media.apply(
            messages,
            [
                record(content_type="image", caption="a chart"),
                record(content_type="text", text="look at this"),
            ],
        )

        self.assertEqual(
            messages[0]["content"], "[image] a chart.\n[text] look at this."
        )

    def test_a_malformed_record_costs_only_itself(self):
        """Another service's data. A shape change must degrade rather than
        stop the bot."""
        messages = [message("msg-1"), message("msg-2")]

        media.apply(
            messages,
            [{"no_target": True}, None, record(target_id="msg-2", caption="fine")],
        )

        self.assertEqual(messages[1]["content"], "[image] fine.")

    def test_nothing_to_do_is_not_an_error(self):
        media.apply([], [])
        media.apply(None, None)


class SummariseTests(SimpleTestCase):
    def test_a_post_and_its_attachment_read_as_two_lines(self):
        summary = media.summarise([
            record(content_type="text", source_type="post", text="our new office"),
            record(content_type="image", source_type="post_attachment",
                   caption="a room with desks"),
        ])

        self.assertEqual(
            summary, "[text] our new office.\n[image] a room with desks."
        )

    def test_nothing_summarises_to_nothing(self):
        self.assertEqual(media.summarise([]), "")
        self.assertEqual(media.summarise(None), "")
