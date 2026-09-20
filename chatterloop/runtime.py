"""Frame in, decision out.

    GET /v1/events
        |
        v
    route by event name
        |
        +-- messages_list, mentioner != null ----> MENTION trigger
        |
        +-- messages_list, DM --------------------> DM trigger
        |
        +-- messages_list, otherwise -------------> probe /replies for a message
        |                                           threaded under one of ours
        v
    resolve the text (a read - the frame carries none)
        |
        v
    policy.evaluate -> IGNORE (with a reason) or RESPOND
        |
        v
    enqueue answer_trigger  (Celery does the expensive half)

WHY THE SPLIT
-------------
Everything here is cheap and stateful - it needs the bot's dedupe window, its
cooldowns, and an open SSE connection. Everything after it (retrieval, a model
call, a send) is expensive and stateless. Keeping the first half on an asyncio
supervisor means an idle bot costs a coroutine rather than a thread stack;
pushing the second half to Celery means the expensive part scales on workers
that hold nothing.

TWO WAYS IN, AND THE SECOND COSTS A READ. A mention is self-describing: the
frame carries a non-null `mentioner` only for recipients the server resolved a
mention against. A reply is not - the frame has no message id and no
`replyingTo` - so "was that a reply to me?" cannot be answered without asking.
It is asked through a route that answers exactly that question, scoped to the
token's own entity, rather than by fetching history and inspecting it here.

LIVE EVENTS ONLY
----------------
The bot answers what happens while it is listening, and nothing else.

A realtime FRAME is inherently live - there is no replay, so a frame published
while the bot was down is simply gone. But everything the bot READS to resolve
one is not: the reply probe returns a window, and comment notifications are
DURABLE and accumulate while offline. A bot that answered everything it found
would come back from a restart and work through the backlog - each item a real
message to a real person about something they said hours ago.

So every candidate resolved from a read is checked against `started_at_ms` and
dropped if it predates this lease. The cost is real and accepted: a mention
that arrives during a deploy is never answered. The alternative is a bot whose
restart is a broadcast.

An unreadable timestamp counts as STALE, never fresh. Silence on a row we
cannot date is recoverable; a burst is not.
"""

import logging
import time

from .client import ChatterloopAPIError, TokenRejected
from .frames import (
    EVENT_MESSAGES_LIST,
    EVENT_NOTIFICATIONS,
    EVENT_NOTIFICATIONS_RELOAD,
    IGNORED_EVENTS,
    parse_envelope,
    parse_messages_list,
)
from .mentions import (
    is_addressed_to,
    normalise_handle,
    strip_command,
    strip_mentions,
)
from .triggers import Trigger, TriggerReason, TriggerSource

logger = logging.getLogger(__name__)


class BotIdentity:
    """The bot's entity, as the platform sees it.

    `entity_id` is the only field that matters functionally - a bot acts
    through an Entity exactly like a person does. The handles are for reading
    text.
    """

    __slots__ = ("entity_id", "handle", "aliases")

    def __init__(self, entity_id, handle, aliases=()):
        if not entity_id:
            raise ValueError("bot entity_id is required")
        self.entity_id = str(entity_id)
        self.handle = normalise_handle(handle)
        self.aliases = frozenset(normalise_handle(a) for a in aliases if a)

    @property
    def handles(self):
        return {self.handle} | set(self.aliases) if self.handle else set(self.aliases)

    def is_self(self, entity_id):
        """Whether an entity id is the bot itself.

        The single most important check in the runtime: a bot that treats its
        own messages as input will answer itself forever.
        """
        return bool(entity_id) and str(entity_id) == self.entity_id


class BotRuntime:
    """One bot's decision loop. Not async - the supervisor runs it off-loop.

    Deliberately synchronous so it is testable without an event loop, and so
    the reads it makes cannot accidentally block the SSE connection: the
    supervisor hands each frame to a thread and processes one at a time per
    bot, which also preserves ordering.
    """

    def __init__(
        self,
        identity,
        policy,
        token,
        dispatch,
        api=None,
        history_window=40,
        reply_probe_window=25,
        answer_replies=True,
        answer_dms=True,
        commands=(),
        only_live_events=True,
        started_at_ms=None,
    ):
        self.identity = identity
        self.policy = policy
        self.token = token
        # Called with a resolved Trigger when the policy says RESPOND. The
        # supervisor passes something that enqueues a Celery task; tests pass a
        # list's append.
        self.dispatch = dispatch
        # Injected so tests need no network. Defaults to the real client.
        if api is None:
            from . import client as api
        self.api = api

        self.history_window = history_window
        self.reply_probe_window = reply_probe_window
        # The kill switch for the whole unnamed-reply path. Off, the bot is
        # mention-only: one early return and no read at all on a message that
        # did not name it.
        self.answer_replies = answer_replies
        # The kill switch for treating every message in a DM as addressed. Off,
        # a DM is judged exactly like a group.
        self.answer_dms = answer_dms

        # The bot's ARSENAL: the command names it answers to.
        #
        # Empty by default, and that is the safe default rather than a missing
        # feature. A `/command` frame reaches every bot in the conversation, so
        # a bot that treated an unknown one as addressed would wake up for
        # somebody else's command - and in a room with several bots, all of
        # them would answer at once.
        self.commands = frozenset(str(name).lower() for name in commands if name)

        self.only_live_events = only_live_events
        self.started_at_ms = (
            int(time.time() * 1000) if started_at_ms is None else started_at_ms
        )

        # conversation_id -> type, resolved once and kept: the platform never
        # changes a conversation's type after creation, so this is one read per
        # conversation over the runtime's lifetime, not one per unaddressed
        # message. Only definitive answers are cached, so a transient read
        # failure retries rather than becoming permanent.
        self._conversation_types = {}
        self.stats = {"frames": 0, "triggers": 0, "dispatched": 0, "ignored": 0, "errors": 0}

    # ----------------------------------------------------------- liveness

    def _is_live(self, created_at_ms):
        """Whether something read from a store happened on this lease's watch.

        A zero or missing timestamp is NOT live. Every path that calls this is
        one where being wrong means sending a real message about something
        somebody said hours ago, so an undatable row is refused rather than
        given the benefit of the doubt.
        """
        if not self.only_live_events:
            return True
        return bool(created_at_ms) and created_at_ms >= self.started_at_ms

    # ------------------------------------------------------------ routing

    def handle_envelope(self, raw):
        self.stats["frames"] += 1

        envelope = parse_envelope(raw)
        if envelope is None:
            return

        if envelope.event in IGNORED_EVENTS:
            return

        # The webapp checks both on every handler; an unauthenticated or failed
        # frame carries no usable payload.
        if not (envelope.auth and envelope.status):
            return

        if envelope.event == EVENT_MESSAGES_LIST:
            self._on_messages_list(envelope)
        elif envelope.event in (EVENT_NOTIFICATIONS, EVENT_NOTIFICATIONS_RELOAD):
            self._on_notifications(envelope)
        else:
            logger.info("unhandled event on entity channel: %s", envelope.event)

    # ----------------------------------------------------------- messages

    def _on_messages_list(self, envelope):
        payload = parse_messages_list(envelope)
        if payload is None:
            return

        # A deletion ping, not a delivery.
        if payload.deleted_message_id:
            return

        # Loop prevention, AHEAD of the mention gate now that a frame without a
        # mention is no longer free. The bot never spends an API call on its
        # own message, and this is what keeps the reply path bounded: its own
        # answers are threaded under somebody else's message, and a bot that
        # read them back would sustain a conversation with itself that needs no
        # @handle at all.
        #
        # Silent rather than counted as an ignore: every message the bot sends
        # comes back on its own channel, so counting these would bury the
        # ignore reasons that mean something under one that never does.
        if self.identity.is_self(payload.entity_id):
            return

        # A command, FIRST. It is the most specific thing a message can be, and
        # reading one as an ordinary mention would hand the arguments to the
        # model as though they were a question.
        #
        # Falls through when the command is not ours: a message like
        # "/unknown @assistant what is this" did still name the bot, and the
        # mention path below is the right answer for it.
        if payload.command is not None and self._command_is_ours(payload.command):
            trigger = Trigger(
                source=TriggerSource.MESSAGE,
                reason=TriggerReason.COMMAND,
                author_entity_id=payload.entity_id,
                conversation_id=payload.conversation_id,
                command_name=payload.command.name,
                command_target=payload.command.target,
                occurred_at=envelope.date_time,
            )
            if payload.mentioner is not None:
                trigger.author_handle = normalise_handle(payload.mentioner.username)
                trigger.realm_name = payload.mentioner.realm_name
                trigger.is_single = payload.mentioner.is_single
            self._guard(trigger, self._resolve_command)
            return

        if payload.mentioner is not None:
            trigger = Trigger(
                source=TriggerSource.MESSAGE,
                reason=TriggerReason.MENTION,
                author_entity_id=payload.entity_id,
                author_handle=normalise_handle(payload.mentioner.username),
                conversation_id=payload.conversation_id,
                realm_name=payload.mentioner.realm_name,
                is_single=payload.mentioner.is_single,
                occurred_at=envelope.date_time,
            )
            self._guard(trigger, self._resolve_mention)
            return

        # No mention on the frame. In a DM that is not evidence of anything -
        # the bot is one of exactly two participants. Checked BEFORE the reply
        # probe: a DM needs neither an @mention nor reply-threading, so there
        # is nothing for that probe to add here, only a read to skip.
        if self.answer_dms and self._is_dm(payload.conversation_id):
            trigger = Trigger(
                source=TriggerSource.MESSAGE,
                reason=TriggerReason.DM,
                author_entity_id=payload.entity_id,
                conversation_id=payload.conversation_id,
                is_single=True,
                occurred_at=envelope.date_time,
            )
            self._guard(trigger, self._resolve_dm)
            return

        if not self.answer_replies:
            return

        self._probe_for_reply(payload, envelope)

    def _is_dm(self, conversation_id):
        """Whether this conversation has exactly two participants.

        Cached forever once known. A miss costs one small read (`limit=1`);
        after that every future message in the SAME conversation is free.
        """
        cached = self._conversation_types.get(conversation_id)
        if cached is not None:
            return cached == "single"

        try:
            resolved = self.api.conversation_type(self.token, conversation_id)
        except TokenRejected:
            raise
        except ChatterloopAPIError as ex:
            # A read can fail for reasons that have nothing to do with this
            # frame, and one bad conversation must not stop the bot reading the
            # next one.
            self.stats["errors"] += 1
            logger.warning(
                "could not resolve conversation type for %s: %s", conversation_id, ex
            )
            return False

        if not resolved:
            # Do NOT cache a guess - a transient failure must not become a
            # permanent misclassification - and default to "not a DM", so the
            # reply probe still gets a chance rather than this method claiming
            # DM status it cannot back up.
            return False

        self._conversation_types[conversation_id] = resolved
        return resolved == "single"

    def _probe_for_reply(self, payload, envelope):
        """Ask the API whether that message was threaded under one of ours."""
        try:
            replies = self.api.fetch_replies_to_me(
                self.token, payload.conversation_id, self.reply_probe_window
            )
        except TokenRejected:
            raise
        except ChatterloopAPIError as ex:
            self.stats["errors"] += 1
            logger.warning(
                "could not probe replies in %s: %s", payload.conversation_id, ex
            )
            return

        message = self._newest_unhandled_reply(replies, payload.entity_id)
        if message is None:
            return

        trigger = Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.REPLY,
            author_entity_id=message["sender_entity_id"],
            author_handle=normalise_handle(message["sender_handle"]),
            conversation_id=payload.conversation_id,
            occurred_at=envelope.date_time,
        )
        self._guard(trigger, lambda t: self._resolve_reply(t, message))

    def _newest_unhandled_reply(self, replies, author_entity_id):
        """The reply this frame is about, or None.

        Scanned newest-first and narrowed to the frame's own sender: between
        the message being sent and this probe completing, other people may have
        spoken.

        Already-handled keys are skipped HERE rather than left to the policy,
        because the probe returns a WINDOW - every new message in a busy
        conversation re-offers replies already answered, and letting those
        through would mean paying for a history fetch to resolve something
        about to be ignored.

        The window is also why the liveness check lives here: the probe is
        happy to hand back a reply from yesterday, and on a freshly leased bot
        the dedupe set may be empty.
        """
        for message in reversed(replies):
            sender = message["sender_entity_id"]
            if self.identity.is_self(sender):
                continue
            if str(sender) != str(author_entity_id):
                continue
            key = f"msg:{message['message_id']}"
            if self.policy.has_seen(key):
                continue
            if not self._is_live(message["created_at"]):
                # Recorded, so the next frame in this conversation does not
                # reconsider the same stale reply and log it again.
                self.policy.record_seen(key)
                self.stats["ignored"] += 1
                logger.info("ignoring reply that predates this lease: %s", key)
                continue
            return message
        return None

    # ----------------------------------------------------------- comments

    def _on_notifications(self, envelope):
        """A notification ping names no subject, so both stores are read.

        The frame's `message` is a human-readable sentence and nothing else -
        the API's own documentation says to respond by reading the routes,
        never by parsing it.
        """
        for kind, fetch in (
            ("mention", self.api.fetch_comment_mentions),
            ("reply", self.api.fetch_comment_replies),
        ):
            try:
                pending = fetch(self.token, self.reply_probe_window)
            except TokenRejected:
                raise
            except ChatterloopAPIError as ex:
                self.stats["errors"] += 1
                logger.warning("could not read comment %ss: %s", kind, ex)
                continue

            for row in pending:
                self._consider_comment(row, envelope)

    def _consider_comment(self, row, envelope):
        key = f"comment:{row['comment_id']}"
        if self.policy.has_seen(key):
            return

        # Comment notifications are DURABLE - they wait indefinitely and pile
        # up while the bot is offline. Without this gate, a restart works
        # through the backlog and answers a page of comments at once, each one
        # about something said hours ago.
        if not self._is_live(row["created_at"]):
            self.policy.record_seen(key)
            self.stats["ignored"] += 1
            return

        trigger = Trigger(
            source=TriggerSource.COMMENT,
            reason=(
                TriggerReason.REPLY if row["kind"] == "reply" else TriggerReason.MENTION
            ),
            author_entity_id=row["author_entity_id"],
            author_handle=normalise_handle(row["author_handle"]),
            post_id=row["post_id"],
            comment_id=row["comment_id"],
            occurred_at=envelope.date_time,
        )
        trigger.text = row["text"]
        trigger.query = strip_mentions(row["text"], self.identity.handles)
        trigger.dedupe_key = key
        self._guard(trigger, lambda t: None)

    # ---------------------------------------------------------- resolving

    def _guard(self, trigger, resolve):
        """Resolve then decide, with one place for a failure to be contained.

        A read can fail for reasons that have nothing to do with this frame,
        and one bad frame must not stop the bot reading the next one. A
        rejected TOKEN is re-raised: that is not a per-frame problem, and the
        supervisor has to stop the whole bot rather than log it once per
        message forever.
        """
        self.stats["triggers"] += 1
        try:
            resolve(trigger)
            self._decide(trigger)
        except TokenRejected:
            raise
        except Exception as ex:
            self.stats["errors"] += 1
            logger.exception(
                "failed handling %s trigger in %s: %s",
                trigger.reason,
                trigger.conversation_id or trigger.post_id,
                ex,
            )

    def _resolve_mention(self, trigger):
        """Fetch the conversation and pull out the addressing message."""
        history, _ = self.api.fetch_messages(
            self.token, trigger.conversation_id, self.history_window
        )
        if not history:
            return

        # The addressing message is the most recent one from the mentioner that
        # actually names us. Scanning back rather than taking the last message
        # outright: between the mention being sent and this fetch completing,
        # other people may have spoken.
        handles = self.identity.handles
        for message in reversed(history):
            if self.identity.is_self(message["sender_entity_id"]):
                continue
            if str(message["sender_entity_id"]) != str(trigger.author_entity_id):
                continue
            if not is_addressed_to(message["content"], handles):
                continue
            self._attach(trigger, message)
            return

        # The platform said we were mentioned but the fetched window does not
        # show it - most likely the message landed outside the window.
        logger.info(
            "mention reported but not found in history for %s", trigger.conversation_id
        )

    def _command_is_ours(self, command):
        """Whether this bot should act on a command somebody typed.

        Two independent gates, and BOTH have to pass:

        THE TARGET. "/summarize:neon" names one bot. Every bot in the
        conversation sees the frame, so a bot that ignored the target would
        answer a command addressed to a different one - which is the exact
        ambiguity the `:handle` suffix exists to remove. An untargeted command
        is offered to every bot that declares it, which is what makes
        "/summarize" work in a room with one summarizer in it.

        THE ARSENAL. A name this bot does not declare is somebody else's
        command, or a typo. Either way it is not ours, and answering it would
        be a bot speaking up about something it cannot do.
        """
        if command.target and command.target not in self.identity.handles:
            return False
        return command.name in self.commands

    def _resolve_command(self, trigger):
        """Find the message that carried the command.

        Searched for by its TEXT rather than taken as the author's newest
        message: between the command being typed and this fetch completing the
        same person may well have typed something else, and answering a command
        with the wrong arguments is worse than not finding it.
        """
        history, _ = self.api.fetch_messages(
            self.token, trigger.conversation_id, self.history_window
        )
        if not history:
            return

        prefix = "/" + trigger.command_name
        for message in reversed(history):
            if self.identity.is_self(message["sender_entity_id"]):
                continue
            if str(message["sender_entity_id"]) != str(trigger.author_entity_id):
                continue
            if not str(message["content"]).lstrip().lower().startswith(prefix):
                continue
            self._attach(trigger, message)
            # The ARGUMENTS are the question, not the whole line. Left as typed,
            # "/summarize the pricing thread" would send the word "summarize"
            # to retrieval - a term that is about the instruction rather than
            # about anything the bot is being asked to find.
            #
            # Applied on top of what _attach already did, so a command that
            # also names the bot loses both the token and the address.
            trigger.query = strip_command(trigger.query)
            return

        logger.info(
            "command /%s reported but not found in history for %s",
            trigger.command_name,
            trigger.conversation_id,
        )

    def _resolve_dm(self, trigger):
        """Take the newest message from the other side, with no addressing check.

        A single conversation has exactly two participants, so "not the bot"
        and "the person who sent the triggering message" are the same set -
        there is no group of other people `is_addressed_to` would need to rule
        out.
        """
        history, _ = self.api.fetch_messages(
            self.token, trigger.conversation_id, self.history_window
        )
        if not history:
            return

        for message in reversed(history):
            if self.identity.is_self(message["sender_entity_id"]):
                continue
            trigger.author_handle = normalise_handle(message["sender_handle"])
            self._attach(trigger, message)
            return

        # Every message in the window was the bot's own - nothing to answer
        # yet, which is not a failure.

    def _resolve_reply(self, trigger, message):
        """Attach the already-identified reply. The probe returned it, so
        unlike the mention path there is nothing to search for."""
        self._attach(trigger, message)

    def _attach(self, trigger, message):
        trigger.text = message["content"]
        # Strip our own @handle before it reaches retrieval: "@assistant what
        # did we decide about pricing" is a question about pricing, and leaving
        # the address in drags retrieval toward the bot's own name - the one
        # term guaranteed to be irrelevant.
        trigger.query = strip_mentions(message["content"], self.identity.handles)
        trigger.message_id = message["message_id"]
        # Keyed on the MESSAGE, not on how it reached us. A message that both
        # names the bot and replies to it arrives twice - once per path - and
        # deserves one answer.
        trigger.dedupe_key = f"msg:{message['message_id']}"

    # ------------------------------------------------------------ deciding

    def _decide(self, trigger):
        decision = self.policy.evaluate(trigger)

        # Recorded for anything FINISHED WITH - a redelivery of something
        # already judged should not be judged again. A transient refusal is not
        # finished with: recording it would mean the probe skips this message
        # forever, so the refusal that was supposed to last an hour lasts for
        # good and the conversation never resumes.
        if not decision.transient:
            self.policy.record_seen(trigger.dedupe_key)

        if not decision.should_respond:
            self.stats["ignored"] += 1
            logger.info(
                "not replying (%s%s) to %s in %s",
                decision.reason,
                "; will reconsider" if decision.transient else "",
                trigger.reason,
                trigger.conversation_id or trigger.post_id,
            )
            return

        self.stats["dispatched"] += 1
        logger.info(
            "dispatching answer (%s%s) for %s in %s",
            decision.reason,
            f" after {decision.delay:.1f}s" if decision.delay else "",
            trigger.reason,
            trigger.conversation_id or trigger.post_id,
        )
        self.dispatch(trigger, decision.delay)
