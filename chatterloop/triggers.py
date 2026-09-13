"""Normalised "something addressed me" events.

Five routes converge here so the policy and the answering task only ever see
one shape:

    MESSAGE  + MENTION   messages_list frame with a non-null `mentioner`
    MESSAGE  + REPLY     a message threaded under one of the bot's own
    MESSAGE  + DM        any message at all, in a conversation with no third
                         party it could instead be aimed at
    COMMENT  + MENTION   an unaddressed `notifications` ping, resolved against
                         the comment-mention route
    COMMENT  + REPLY     the same ping, resolved against the comment-reply route

The SOURCE says which surface it happened on, and therefore how to answer it.
The REASON says why the bot is entitled to answer at all. Both are recorded,
because "why did it answer that?" is a question the logs have to settle.
"""

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class TriggerSource(StrEnum):
    MESSAGE = "message"
    COMMENT = "comment"


class TriggerReason(StrEnum):
    """Why this counts as being addressed.

    MENTION is the original rule and still the only way to START a thread with
    the bot in a GROUP conversation.

    REPLY covers a message aimed directly at something the bot itself said,
    where re-typing the handle every turn would be ceremony no human
    conversation has.

    DM is neither, and needs no @handle or reply-threading at all: in a single
    conversation the bot is one of exactly two participants, so every message
    from the other one is addressed to it by construction.

    Deliberately NOT a value for "replied to somebody else in a group thread
    the bot is in". That is ordinary conversation between other people.
    """

    MENTION = "mention"
    REPLY = "reply"
    DM = "dm"


@dataclass(slots=True)
class Trigger:
    """Something the bot may want to answer."""

    source: TriggerSource
    author_entity_id: str
    author_handle: str = ""
    reason: TriggerReason = TriggerReason.MENTION

    # Messages: the conversation. Comments: empty - a comment lives on a post.
    conversation_id: str = ""
    # The message that addressed us. The bot answers threaded under it, so this
    # becomes `replyingTo` on the outgoing message.
    message_id: str = ""
    # Comments: the post and the comment doing the addressing.
    post_id: str = ""
    comment_id: str = ""

    # The text that addressed us, once fetched. Empty until resolved: the
    # frame carries no content.
    text: str = ""
    # `text` with the bot's own @handle removed - what actually gets embedded.
    query: str = ""

    realm_name: str = ""
    is_single: bool = False
    occurred_at: str = ""
    # Stable key for deduplication. Derived from what identifies the underlying
    # object, so a redelivered frame collapses onto the same key - and so does
    # the SAME message arriving once as a mention and once as a reply, which is
    # exactly what a "@bot, and also replying to you" message does.
    dedupe_key: str = ""

    @property
    def is_resolved(self):
        """Whether the trigger carries enough to answer.

        A trigger with no text is something we know happened but cannot read.
        Answering one would mean generating a reply to an unknown question.
        """
        return bool(self.query.strip())

    @property
    def scope(self):
        """What cooldowns and hourly caps are counted against."""
        return self.conversation_id or self.post_id or "global"

    def to_payload(self):
        """A JSON-safe dict for the Celery task.

        Celery serialises arguments, and a dataclass with StrEnum members is
        not JSON. Sent as a plain dict rather than pickled, so the queue never
        carries executable state.
        """
        payload = asdict(self)
        payload["source"] = str(self.source)
        payload["reason"] = str(self.reason)
        return payload

    @classmethod
    def from_payload(cls, payload):
        data = dict(payload)
        data["source"] = TriggerSource(data["source"])
        data["reason"] = TriggerReason(data["reason"])
        return cls(**data)
