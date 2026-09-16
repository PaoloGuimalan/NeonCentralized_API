"""Should the bot say something?

Answering is the EXCEPTION, not the default. A bot sees every event on its
entity's channel - every message in every conversation it belongs to - and
replies only when addressed. Each rule below is a separate reason to stay
silent, evaluated cheapest-first, and each returns a reason string so "why
didn't it answer?" is answerable from the logs.

THREE WAYS TO BE ADDRESSED, and the second and third are not a loosening of
the first:

  * an @mention, which is how a thread with the bot STARTS in a group;
  * a direct reply to something the bot itself said, which is how one
    continues - requiring the handle every turn is ceremony no human
    conversation has;
  * any message at all, in a DM, where there is no third party a message could
    instead be about.

All three are explicit acts aimed at the bot. A reply to somebody ELSE in a
group thread the bot happens to be in is still nothing to do with it.

WHY THE STATE IS NOT JUST IN MEMORY
-----------------------------------
The reference implementation kept dedupe, cooldown and the hourly cap in one
process, because one process both decided and sent. Neon splits those: the
supervisor decides, a Celery worker sends. So `record_reply` - which must fire
only AFTER a successful send, or a broken outbound path silently rate-limits
the bot into silence - happens somewhere the deciding process cannot reach.

Sharing the state through Redis is what keeps that rule intact. It also
survives a supervisor restart, which the in-memory version did not: the
liveness gate covers a backlog, but not a frame redelivered thirty seconds
after a redeploy.
"""

import logging
import time
from collections import OrderedDict
from enum import StrEnum

from .triggers import TriggerReason

logger = logging.getLogger(__name__)

# How long a dedupe key is remembered. Long enough that a redelivery or a
# reply-probe window cannot slip past it, short enough that the keys expire on
# their own rather than growing forever.
SEEN_TTL_SECONDS = 24 * 3600
HOUR_SECONDS = 3600

# How many consecutive bot-to-bot turns one conversation gets before the
# collaboration is cut off.
#
# A ceiling is not optional. Two bots answering each other never run out of
# things to say on their own - every threaded reply re-addresses the other, so
# "the task is finished" is a judgement only the models make, and a model that
# fails to make it produces completions until somebody looks at the bill. The
# budget is what bounds the case where they do not stop.
#
# It counts TURNS IN A ROW, and a human speaking resets it: a person asking for
# something new is exactly the signal that the previous collaboration ended,
# and the one event that should hand back a full budget.
BOT_TURN_BUDGET = 40


class Verdict(StrEnum):
    RESPOND = "respond"
    IGNORE = "ignore"


class Decision:
    """A verdict, why, and - for two of them - what to do about it later.

    TRANSIENT vs TERMINAL
    ---------------------
    `transient` means "not now", not "never". The caller records a dedupe key
    for terminal refusals only; recording a transient one burns the message,
    and burning it is what turned "rate limited for an hour" into a
    conversation that never resumed - the probe skips a seen key forever, so
    nothing retries once the window rolls off.

    `delay` turns the cooldown from a DROP into a WAIT. For a person typing
    "@bot" five times, dropping is right - they want one answer. For two bots
    working through a task, dropping ends the collaboration on the first fast
    exchange, so the answer is dispatched with a countdown instead and the
    pacing survives.
    """

    __slots__ = ("verdict", "reason", "transient", "delay")

    def __init__(self, verdict, reason, transient=False, delay=0.0):
        self.verdict = verdict
        self.reason = reason
        self.transient = transient
        self.delay = delay

    @property
    def should_respond(self):
        return self.verdict is Verdict.RESPOND

    def with_delay(self, delay):
        return Decision(self.verdict, self.reason, self.transient, delay)

    def __repr__(self):
        suffix = f", delay={self.delay:.1f}s" if self.delay else ""
        return f"Decision({self.verdict}, {self.reason!r}{suffix})"


# Kept distinct so the logs say WHICH rule let a reply through. "Addressed" is
# one word for three quite different situations, and the difference is the
# first thing anyone wants when a reply looks unwarranted.
RESPOND_MENTIONED = Decision(Verdict.RESPOND, "addressed by mention")
RESPOND_REPLIED_TO = Decision(Verdict.RESPOND, "direct reply to the bot")
RESPOND_DM = Decision(Verdict.RESPOND, "message in a direct conversation")

_RESPOND_BY_REASON = {
    TriggerReason.MENTION: RESPOND_MENTIONED,
    TriggerReason.REPLY: RESPOND_REPLIED_TO,
    TriggerReason.DM: RESPOND_DM,
}


# --------------------------------------------------------------------- stores


class InMemoryPolicyStore:
    """Per-process state. Used by tests, and as the fallback when Redis is not
    configured - a single supervisor with no worker split still behaves
    correctly, it just forgets everything on restart."""

    def __init__(self, dedupe_window=4096):
        self._seen = OrderedDict()
        self._dedupe_window = dedupe_window
        self._last_reply_at = {}
        self._reply_times = {}
        self._bot_turns = {}

    def has_seen(self, key):
        return bool(key) and key in self._seen

    def record_seen(self, key):
        if not key or key in self._seen:
            return
        self._seen[key] = True
        while len(self._seen) > self._dedupe_window:
            self._seen.popitem(last=False)

    def last_reply_at(self, scope):
        return self._last_reply_at.get(scope)

    def recent_reply_count(self, scope, now):
        times = self._reply_times.get(scope)
        if not times:
            return 0
        cutoff = now - HOUR_SECONDS
        kept = [t for t in times if t >= cutoff]
        self._reply_times[scope] = kept
        return len(kept)

    def record_reply(self, scope, now):
        self._last_reply_at[scope] = now
        self._reply_times.setdefault(scope, []).append(now)

    def bot_turns(self, scope):
        return self._bot_turns.get(scope, 0)

    def record_bot_turn(self, scope):
        self._bot_turns[scope] = self._bot_turns.get(scope, 0) + 1

    def reset_bot_turns(self, scope):
        self._bot_turns.pop(scope, None)


class RedisPolicyStore:
    """Shared state, so the supervisor and the workers agree.

    Keys are namespaced per bot. Every key carries a TTL - nothing here is
    worth keeping once a bot has been quiet for a day, and an expiring key is
    one fewer thing to clean up.
    """

    def __init__(self, client, bot_id):
        self.client = client
        self.prefix = f"neon:bot:{bot_id}"

    def has_seen(self, key):
        if not key:
            return False
        return bool(self.client.exists(f"{self.prefix}:seen:{key}"))

    def record_seen(self, key):
        if not key:
            return
        self.client.set(f"{self.prefix}:seen:{key}", 1, ex=SEEN_TTL_SECONDS)

    def last_reply_at(self, scope):
        raw = self.client.get(f"{self.prefix}:lastreply:{scope}")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def recent_reply_count(self, scope, now):
        key = f"{self.prefix}:replies:{scope}"
        # Trimmed on read rather than by a sweeper: the only moment the count
        # matters is now, and trimming here means no background job.
        self.client.zremrangebyscore(key, "-inf", now - HOUR_SECONDS)
        return int(self.client.zcard(key) or 0)

    def bot_turns(self, scope):
        return int(self.client.get(f"{self.prefix}:botturns:{scope}") or 0)

    def record_bot_turn(self, scope):
        key = f"{self.prefix}:botturns:{scope}"
        self.client.incr(key)
        # Refreshed on every turn rather than set once: a collaboration that
        # runs for hours should not have its budget silently reset mid-task by
        # an expiry, and one that stopped should not hold the count for a day.
        self.client.expire(key, HOUR_SECONDS)

    def reset_bot_turns(self, scope):
        self.client.delete(f"{self.prefix}:botturns:{scope}")

    def record_reply(self, scope, now):
        self.client.set(f"{self.prefix}:lastreply:{scope}", now, ex=HOUR_SECONDS)
        key = f"{self.prefix}:replies:{scope}"
        # Scored AND membered by the timestamp: two replies in the same
        # microsecond would collapse, which is a rounding error in a cap of 30
        # and not worth a uuid per reply.
        self.client.zadd(key, {str(now): now})
        self.client.expire(key, HOUR_SECONDS)


def build_store(bot_id):
    """A Redis-backed store where Redis exists, in-memory otherwise.

    Falls back rather than failing: a deployment without Redis still runs one
    supervisor correctly, and gets a clear warning about what it loses.
    """
    try:
        from django_redis import get_redis_connection

        return RedisPolicyStore(get_redis_connection("default"), bot_id)
    except Exception as ex:
        logger.warning(
            "no Redis for bot policy state; falling back to per-process memory. "
            "Reply budgets will not be shared with workers and will reset on "
            "restart. (%s)",
            ex,
        )
        return InMemoryPolicyStore()


# --------------------------------------------------------------------- policy


class AddressedOnlyPolicy:
    """Reply only when addressed, with loop and flood protection.

    Being addressed is the product rule. The rest is what stops a bot with that
    rule from still being a problem: bots that answer bots, bots that answer
    themselves, and bots that answer the same thing twice because the stream
    redelivered.
    """

    def __init__(
        self,
        identity,
        store=None,
        cooldown_seconds=2.0,
        max_replies_per_hour=30,
        ignore_entity_ids=frozenset(),
        is_bot_author=None,
        allow_bot_conversations=False,
        bot_turn_budget=BOT_TURN_BUDGET,
    ):
        self.identity = identity
        self.store = store if store is not None else InMemoryPolicyStore()
        self.cooldown_seconds = cooldown_seconds
        self.max_replies_per_hour = max_replies_per_hour
        # Other bots, or any entity that should never get a reply. Two bots in
        # one realm that both answer mentions will otherwise mention each other
        # until somebody notices the bill.
        self.ignore_entity_ids = {str(e) for e in ignore_entity_ids if e}

        # Whether an entity is a bot. A predicate rather than a set, because
        # the answer lives in chatterloop's `bot_bot` and has to cover bots
        # that are not Neon's - a set built from Neon's own rows would call a
        # stranger's bot a person.
        self.is_bot_author = is_bot_author
        # The per-bot toggle. OFF means bot-authored triggers are refused
        # outright, which is the old behaviour made explicit.
        self.allow_bot_conversations = allow_bot_conversations
        self.bot_turn_budget = bot_turn_budget

    def evaluate(self, trigger, now=None):
        now = time.time() if now is None else now
        scope = trigger.scope

        # 1. Never answer yourself. Cheapest check, and the one whose failure
        #    is unbounded - a self-reply is itself a message, which produces
        #    another event. It matters more on the reply path than it ever did
        #    on the mention path: the bot's own answers are threaded under
        #    somebody else's message, so a bot that read them back would answer
        #    its own thread forever without a single @handle being typed.
        if self.identity.is_self(trigger.author_entity_id):
            return Decision(Verdict.IGNORE, "author is the bot itself")

        # 2. Never answer an entity on the ignore list (other bots).
        if str(trigger.author_entity_id) in self.ignore_entity_ids:
            return Decision(Verdict.IGNORE, "author is on the ignore list")

        # 3. At-least-once delivery, plus a reply probe that returns a WINDOW
        #    of recent replies, means repeats are normal rather than
        #    exceptional.
        if trigger.dedupe_key and self.store.has_seen(trigger.dedupe_key):
            return Decision(Verdict.IGNORE, "already handled")

        # 4. Something we cannot read is not a question we can answer.
        if not trigger.is_resolved:
            return Decision(Verdict.IGNORE, "trigger text could not be resolved")

        from_bot = bool(
            self.is_bot_author and self.is_bot_author(trigger.author_entity_id)
        )

        # 5. A bot talking to a bot, where that was not asked for.
        #
        #    TRANSIENT, because a toggle is exactly the kind of refusal that
        #    stops applying: somebody switching it on wants the conversation in
        #    front of them to continue, not to have to retype the message that
        #    was refused while it was off.
        if from_bot and not self.allow_bot_conversations:
            return Decision(
                Verdict.IGNORE,
                "bot-to-bot replies are off for this bot",
                transient=True,
            )

        # 6. The collaboration budget, which replaces the hourly cap while two
        #    bots are working. Terminal on purpose - the point of a ceiling is
        #    that waiting does not lift it, and a transient refusal here would
        #    re-offer the same message on every subsequent frame forever.
        if from_bot:
            used = self.store.bot_turns(scope)
            if used >= self.bot_turn_budget:
                return Decision(
                    Verdict.IGNORE,
                    f"bot collaboration budget spent ({used}/{self.bot_turn_budget} "
                    f"turns); a person speaking here resets it",
                )

        # 7. Per-conversation cooldown.
        #
        #    For a person it is a DROP: somebody typing "@bot" five times in a
        #    row wants one answer, not five.
        #
        #    Between collaborating bots it is a WAIT. Dropping would end the
        #    exchange on the first time one answers inside five seconds, which
        #    at typical latency is immediately - and the pacing is wanted, so
        #    the answer is delayed rather than discarded.
        last = self.store.last_reply_at(scope)
        if last is not None and (now - last) < self.cooldown_seconds:
            if not (from_bot and self.allow_bot_conversations):
                # Transient, but deliberately still recorded by the caller as
                # handled - see `record_seen`. The five identical mentions ARE
                # the thing being collapsed, so each one is genuinely finished
                # with, not postponed.
                return Decision(
                    Verdict.IGNORE, "within cooldown for this conversation"
                )
            wait = self.cooldown_seconds - (now - last)
            return _RESPOND_BY_REASON[trigger.reason].with_delay(wait)

        # 8. Hourly ceiling per conversation - the backstop for anything the
        #    rules above did not anticipate. Transient: it expires on its own,
        #    and recording it as handled is what previously turned an hour's
        #    pause into a conversation that never came back.
        #
        #    Skipped for collaborating bots, which are bounded by the turn
        #    budget above instead.
        if not from_bot:
            if self.store.recent_reply_count(scope, now) >= self.max_replies_per_hour:
                return Decision(
                    Verdict.IGNORE, "hourly reply limit reached", transient=True
                )

        return _RESPOND_BY_REASON[trigger.reason]

    def has_seen(self, dedupe_key):
        """Whether this key has already been judged.

        Public because the reply probe needs it BEFORE building a trigger. The
        probe returns a window, so most of what it offers on any given frame is
        old news; discovering that inside `evaluate` would mean paying for a
        history fetch to resolve something about to be ignored.
        """
        return self.store.has_seen(dedupe_key)

    def record_seen(self, dedupe_key):
        """Mark a trigger as handled.

        Recorded for terminally IGNORED triggers too: a redelivery of something
        already judged not worth answering should not be re-judged, and
        re-judging is not free once a fetch is involved.

        NOT for transient ones - see `Decision.transient`. Recording a refusal
        that will stop applying is what made the hourly cap permanent: the
        probe skips a seen key forever, so when the window rolled off there was
        nothing left un-handled to answer, and the conversation simply never
        came back.
        """
        self.store.record_seen(dedupe_key)

    def record_reply(self, trigger, now=None):
        """Called after a reply is actually SENT, not when it is decided.

        A reply that failed to send must not consume the conversation's budget
        - otherwise a broken outbound path silently rate-limits the bot into
        silence. This is why the store is shared: the send happens in a worker,
        not in the process that made the decision.

        This is also where the collaboration budget moves. Answering a BOT
        spends a turn; answering a PERSON hands the budget back, because
        somebody asking for something new is the clearest available signal
        that the previous collaboration is over.
        """
        self.store.record_reply(trigger.scope, time.time() if now is None else now)

        if self.is_bot_author and self.is_bot_author(trigger.author_entity_id):
            self.store.record_bot_turn(trigger.scope)
        else:
            self.store.reset_bot_turns(trigger.scope)
