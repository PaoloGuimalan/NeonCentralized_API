"""What the model reads when somebody sends the bot a post.

Like test_media, these assert on the exact string: every rule here decides text
a language model acts on, and when one is wrong nothing throws - the reply is
just about a thirty-digit number.

SimpleTestCase throughout: `shared_posts` is pure.
"""

from django.test import SimpleTestCase

from chatterloop import shared_posts
from chatterloop.client import _messages_from


def record(**overrides):
    row = {
        "target_id": "p-1",
        "source_type": "post",
        "content_type": "text",
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


def sent_alone(message_id="m-1", post_id="p-1", created_at=100):
    """The shape of a post sent with no note."""
    return {
        "message_id": message_id,
        "message_type": "post",
        "content": post_id,
        "created_at": created_at,
        "sender_entity_id": "entity-ana",
        "reply_target": None,
    }


def with_note(message_id="m-2", post_id="p-1", note="what do you think?",
              kind="post", created_at=200):
    """The shape of a post sent with a note: a text reply to the post."""
    return {
        "message_id": message_id,
        "message_type": "text",
        "content": note,
        "created_at": created_at,
        "sender_entity_id": "entity-ana",
        "is_reply": True,
        "replying_to": "",
        "reply_target": {"type": kind, "id": post_id},
    }


POST = [
    record(text="our new office"),
    record(source_type="post_attachment", content_type="image",
           caption="a room with desks"),
]


class PostOfTests(SimpleTestCase):
    def test_a_post_sent_alone_is_its_content(self):
        self.assertEqual(shared_posts.post_of(sent_alone()), ("post", "p-1"))

    def test_a_note_is_about_its_reply_target(self):
        self.assertEqual(shared_posts.post_of(with_note()), ("post", "p-1"))

    def test_moments_and_thoughts_are_posts_too(self):
        self.assertEqual(
            shared_posts.post_of(with_note(kind="moment")), ("moment", "p-1")
        )
        self.assertEqual(
            shared_posts.post_of(with_note(kind="thought")), ("thought", "p-1")
        )

    def test_a_reply_to_a_message_is_not_a_post(self):
        message = with_note()
        message["reply_target"] = {"type": "message", "id": "m-0"}

        self.assertIsNone(shared_posts.post_of(message))

    def test_plain_text_and_uploads_are_not_posts(self):
        for message_type, content in (("text", "hello"), ("image/jpeg", "https://cdn.test/a.jpg")):
            with self.subTest(message_type=message_type):
                self.assertIsNone(shared_posts.post_of({
                    "message_id": "m-9", "message_type": message_type,
                    "content": content, "reply_target": None,
                }))

    def test_a_post_with_no_id_is_nothing_to_read(self):
        self.assertIsNone(shared_posts.post_of(sent_alone(post_id="  ")))


class WantedTests(SimpleTestCase):
    def test_the_triggering_messages_post_comes_first(self):
        messages = [
            sent_alone("m-old", "p-old", created_at=10),
            sent_alone("m-new", "p-new", created_at=900),
            sent_alone("m-asked", "p-asked", created_at=50),
        ]

        self.assertEqual(
            shared_posts.wanted(messages, first="m-asked"),
            ["p-asked", "p-new", "p-old"],
        )

    def test_the_same_post_is_read_once(self):
        """The window and the reply chain overlap, so it arrives twice."""
        messages = [sent_alone("m-1", "p-1"), sent_alone("m-1", "p-1"),
                    with_note("m-2", "p-1")]

        self.assertEqual(shared_posts.wanted(messages), ["p-1"])

    def test_a_window_full_of_posts_is_bounded(self):
        """One call per post. The newest are what the conversation is about."""
        messages = [sent_alone(f"m-{i}", f"p-{i}", created_at=i) for i in range(10)]

        self.assertEqual(shared_posts.wanted(messages), ["p-9", "p-8", "p-7"])

    def test_no_posts_is_nothing_to_read(self):
        self.assertEqual(shared_posts.wanted([]), [])
        self.assertEqual(shared_posts.wanted(None), [])


class ApplyTests(SimpleTestCase):
    def test_a_post_sent_alone_becomes_the_post(self):
        messages = [sent_alone()]

        shared_posts.apply(messages, {"p-1": POST})

        self.assertEqual(
            messages[0]["content"],
            "[Shared a post]\n[text] our new office.\n[image] a room with desks.",
        )

    def test_a_note_reads_below_the_post_it_is_about(self):
        messages = [with_note()]

        shared_posts.apply(messages, {"p-1": POST})

        self.assertEqual(
            messages[0]["content"],
            "[About a post]\n[text] our new office.\n[image] a room with desks."
            "\n\nwhat do you think?",
        )

    def test_a_reply_to_a_moment_says_so(self):
        messages = [with_note(kind="moment", note="love this")]

        shared_posts.apply(messages, {"p-1": [record(content_type="image", caption="a sunset")]})

        self.assertEqual(
            messages[0]["content"], "[About a Moment]\n[image] a sunset.\n\nlove this"
        )

    def test_a_post_that_was_not_read_still_loses_its_id(self):
        """No access, a failed read, or past the cap. "There is a post here"
        is more than a bare id, and a note about nothing."""
        alone, noted = sent_alone(), with_note()

        shared_posts.apply([alone, noted], {})

        self.assertEqual(alone["content"], "[Shared a post]")
        self.assertEqual(noted["content"], "[About a post]\n\nwhat do you think?")

    def test_a_deleted_post_is_said_to_be_gone(self):
        messages = [sent_alone()]

        shared_posts.apply(messages, {"p-1": shared_posts.GONE})

        self.assertEqual(
            messages[0]["content"], "[Shared a post that is no longer available]"
        )

    def test_a_post_still_being_read_says_so(self):
        """Analysis is asynchronous, like an upload's."""
        messages = [sent_alone()]

        shared_posts.apply(
            messages,
            {"p-1": [record(content_type="video", source_type="post_attachment",
                            status="pending")]},
        )

        self.assertEqual(
            messages[0]["content"], "[Shared a post]\n[video] Still being processed."
        )

    def test_the_same_message_is_not_rewritten_twice(self):
        """A second pass would read the bracket as a post id."""
        messages = [sent_alone()]
        shared_posts.apply(messages, {"p-1": POST})
        once = messages[0]["content"]

        shared_posts.apply(messages, {"p-1": POST})

        self.assertEqual(messages[0]["content"], once)

    def test_other_messages_are_untouched(self):
        text = {"message_id": "m-9", "message_type": "text", "content": "hello",
                "reply_target": None}

        rewritten = shared_posts.apply([text], {"p-1": POST})

        self.assertEqual(text["content"], "hello")
        self.assertEqual(rewritten, {})

    def test_what_was_rewritten_is_reported_by_message(self):
        rewritten = shared_posts.apply([sent_alone("m-1"), with_note("m-2")], {})

        self.assertEqual(
            rewritten,
            {"m-1": ("[Shared a post]", True), "m-2": ("[About a post]", False)},
        )


class AskTests(SimpleTestCase):
    def test_sent_alone_the_post_is_the_question(self):
        """The query was the post's id."""
        self.assertEqual(
            shared_posts.ask("762856157557296690215357838746", ("[Shared a post]", True)),
            "[Shared a post]",
        )

    def test_with_a_note_the_note_is_asked_about_the_post(self):
        self.assertEqual(
            shared_posts.ask("what do you think?", ("[About a post]\n[image] a cat.", False)),
            "[About a post]\n[image] a cat.\n\nwhat do you think?",
        )

    def test_a_note_that_was_only_the_bots_handle_asks_about_the_post(self):
        self.assertEqual(
            shared_posts.ask("  ", ("[About a post]", False)), "[About a post]"
        )


class ClientShapeTests(SimpleTestCase):
    """The client has to keep `reply_target` - it is the only field saying a
    note is about a post."""

    def row(self, **overrides):
        row = {"message_id": "m-1", "sender_entity_id": "entity-ana",
               "content": "what do you think?", "message_type": "text",
               "is_reply": True, "replying_to": ""}
        row.update(overrides)
        return row

    def test_a_reply_target_is_kept(self):
        [message] = _messages_from(
            {"messages": [self.row(reply_target={"type": "Post", "id": "p-1"})]}, "c-1"
        )

        self.assertEqual(message["reply_target"], {"type": "post", "id": "p-1"})

    def test_no_or_malformed_reply_target_is_none(self):
        for value in (None, "p-1", {"type": "post"}, {"id": "p-1"}):
            with self.subTest(value=value):
                [message] = _messages_from(
                    {"messages": [self.row(reply_target=value)]}, "c-1"
                )
                self.assertIsNone(message["reply_target"])
