"""Holds an SSE connection per leased bot, and decides what to answer.

SHAPE
-----
    sweep loop  ------> find live bots, try to lease each one
        |                     |
        |                     v
        |               per bot: a reader task and a handler task
        |                     |
        |    reader: stream /v1/events, push raw frames onto a queue
        |    handler: drain the queue, one frame at a time, off-loop
        |                     |
        v                     v
    renew loop            Celery: answer_trigger
    (drops what it loses)

WHY A QUEUE BETWEEN THE TWO
---------------------------
Resolving a frame can cost an HTTP read (the reply probe, the DM check), which
is synchronous. Doing that on the event loop would stall every OTHER bot's
stream behind one slow conversation. Doing it with `asyncio.to_thread` straight
from the reader would process a bot's frames CONCURRENTLY and out of order,
which breaks dedupe and cooldown - both assume they see events in sequence.

A bounded queue with one handler per bot keeps ordering, keeps the loop free,
and makes backpressure visible: a full queue drops frames with a log line
rather than growing until the container dies.

SELF-BALANCING
--------------
There is no shard configuration. Every supervisor sweeps for unleased bots and
takes what it can get, so N supervisors divide the work by racing, and a
crashed one's bots are picked up within a lease TTL by whoever sweeps next.
"""

import asyncio
import logging

from django.utils.timezone import now

from .client import ChatterloopAPIError, TokenRejected, stream_events
from .control import listen
from .leases import LeaseManager, RENEW_INTERVAL_SECONDS
from .authors import is_bot_entity
from .policy import AddressedOnlyPolicy, build_store
from .runtime import BotIdentity, BotRuntime

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 20
# Bounded on purpose. A bot in a very busy realm can out-produce its handler,
# and the right answer is to drop the overflow loudly rather than buffer it
# until the process dies - a frame is a hint to refetch, not the record.
FRAME_QUEUE_SIZE = 256

RECONNECT_DELAY_SECONDS = 2.0
MAX_RECONNECT_DELAY_SECONDS = 30.0


class BotWorker:
    """One leased bot: a reader, a handler, and the runtime between them."""

    def __init__(self, bot, token, lease_acquired_ms):
        self.bot_id = str(bot.pk)
        self.handle = bot.handle
        self.token = token
        self.queue = asyncio.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._running = True
        self._tasks = []
        self.dropped_frames = 0
        # Set to a reason when the bot must not be retried - a dead token, or
        # one missing its grant. The sweep reads this and leaves the bot alone.
        self.fatal = ""

        identity = BotIdentity(bot.entity_id, bot.verified_handle or bot.handle)
        policy = AddressedOnlyPolicy(
            identity,
            store=build_store(bot.pk),
            is_bot_author=is_bot_entity,
            allow_bot_conversations=bot.allow_bot_conversations,
        )
        self.runtime = BotRuntime(
            identity=identity,
            policy=policy,
            token=token,
            dispatch=self._dispatch,
            # Stamped when the LEASE is acquired, not at import: this is the
            # watermark that stops a restart working through a backlog, and it
            # has to mean "since this supervisor took responsibility for this
            # bot", which is now.
            started_at_ms=lease_acquired_ms,
        )

    def _dispatch(self, trigger, delay=0.0):
        # Imported here rather than at module import: Celery's task registry
        # wants Django set up, and this module is imported by a management
        # command that does that itself.
        from .tasks import answer_trigger

        if delay > 0:
            # The cooldown, kept as pacing rather than spent as a refusal. Two
            # bots answering each other instantly reads as machinery; a few
            # seconds apart reads as a conversation, and it is the same wait
            # either way - the only question was whether the turn survived it.
            answer_trigger.apply_async(
                args=[self.bot_id, trigger.to_payload()], countdown=delay
            )
            return
        answer_trigger.delay(self.bot_id, trigger.to_payload())

    def start(self):
        self._tasks = [
            asyncio.create_task(self._read(), name=f"read:{self.handle}"),
            asyncio.create_task(self._handle(), name=f"handle:{self.handle}"),
        ]

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("error stopping worker for %s", self.handle)
        self._tasks = []

    def _on_envelope(self, raw):
        try:
            self.queue.put_nowait(raw)
        except asyncio.QueueFull:
            self.dropped_frames += 1
            # Logged every time rather than sampled: a dropped frame is a
            # message the bot will never see, and silently absorbing that is
            # how a bot gets a reputation for ignoring people.
            logger.warning(
                "frame queue full for %s; dropped a frame (%s total)",
                self.handle,
                self.dropped_frames,
            )

    async def _read(self):
        """Stream events, reconnecting with backoff.

        A clean disconnect is EXPECTED - the server caps a stream's lifetime at
        an hour - so it reads as routine rather than as an error.
        """
        delay = RECONNECT_DELAY_SECONDS
        while self._running:
            try:
                await stream_events(
                    self.token,
                    self._on_envelope,
                    should_continue=lambda: self._running,
                )
                delay = RECONNECT_DELAY_SECONDS
                if self._running:
                    logger.info("event stream ended for %s; reconnecting", self.handle)
            except asyncio.CancelledError:
                raise
            except TokenRejected as ex:
                # Will be exactly as bad on the next attempt. Retrying turns a
                # misconfiguration into a burst of traffic that looks like an
                # attack, and buries the one log line that explains it.
                self.fatal = ex.message
                logger.error(
                    "bot %s cannot subscribe: %s", self.handle, ex.message
                )
                return
            except ChatterloopAPIError as ex:
                logger.warning(
                    "event stream failed for %s (%s); retrying in %ss",
                    self.handle,
                    ex,
                    delay,
                )
            except Exception as ex:
                logger.warning(
                    "event stream error for %s (%s); retrying in %ss",
                    self.handle,
                    ex,
                    delay,
                )

            if not self._running:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY_SECONDS)

    async def _handle(self):
        """Drain the queue, one frame at a time, off the event loop."""
        while self._running:
            raw = await self.queue.get()
            try:
                await asyncio.to_thread(self.runtime.handle_envelope, raw)
            except asyncio.CancelledError:
                raise
            except TokenRejected as ex:
                self.fatal = ex.message
                logger.error(
                    "bot %s: token rejected while handling a frame: %s",
                    self.handle,
                    ex.message,
                )
                self._running = False
                return
            except Exception:
                logger.exception("bot %s failed handling a frame", self.handle)
            finally:
                self.queue.task_done()


class Supervisor:
    """Sweeps for bots to run, and runs the ones it wins."""

    def __init__(self, redis_client, sweep_interval=SWEEP_INTERVAL_SECONDS, once=False):
        self.leases = LeaseManager(redis_client)
        self.sweep_interval = sweep_interval
        # Set when somebody switches a bot on or off, so the next sweep happens
        # now rather than up to `sweep_interval` later. Best effort - see
        # chatterloop/control.py - and the timer is what guarantees
        # convergence.
        self._nudged = asyncio.Event()
        self._listener = None
        # For tests and for `--once`: do one sweep and return rather than
        # looping forever.
        self.once = once
        self.workers = {}
        self._running = True
        # Bots we have stopped trying, and why. A dead token does not fix
        # itself, so re-leasing every sweep would be a loop of the same error.
        self._refused = {}

    # ------------------------------------------------------------- lookup

    def _candidates(self):
        """Bots that are switched ON and able to answer, with a usable token.

        `is_online` is the on/off switch. A bot with it false keeps its
        identity, its token and its agent binding - it simply is not leased, so
        no event stream is opened for it and no frames reach it. That is the
        whole mechanism: there is no separate "disconnect" step, because not
        being a candidate IS being offline.

        Filtered in the database rather than in Python: a deployment with
        thousands of bots should not load them all to discard most.
        """
        from django.db.models import Prefetch

        from .models import ChatterloopBot, ChatterloopToken

        # The live tokens come back WITH the bots. Without this prefetch
        # `_token_for` issues one query per bot, so a sweep costs O(bots) round
        # trips every interval - and a sweep runs whether or not anything
        # changed, on every supervisor. At a few hundred bots that is the
        # single largest source of idle database load in the system.
        live_tokens = ChatterloopToken.objects.filter(
            revoked_at__isnull=True, provisioned_at__isnull=False
        ).order_by("-created_at")

        return list(
            ChatterloopBot.objects.filter(
                status=ChatterloopBot.STATUS_ACTIVE,
                is_online=True,
                agent__isnull=False,
                model__isnull=False,
                tokens__revoked_at__isnull=True,
                tokens__provisioned_at__isnull=False,
            )
            .select_related("organization")
            .prefetch_related(Prefetch("tokens", queryset=live_tokens, to_attr="live_tokens"))
            .distinct()
        )

    @staticmethod
    def _token_for(bot):
        # Prefer the prefetched list - see `_candidates`. Falls back to a query
        # so this stays correct for a bot loaded any other way.
        prefetched = getattr(bot, "live_tokens", None)
        if prefetched is not None:
            credential = prefetched[0] if prefetched else None
        else:
            credential = (
                bot.tokens.filter(revoked_at__isnull=True, provisioned_at__isnull=False)
                .order_by("-created_at")
                .first()
            )
        if credential is None:
            return ""
        try:
            return credential.token
        except Exception:
            # An unreadable secret - a rotated encryption key with no fallback.
            # Named rather than swallowed: the only remedy is minting a
            # replacement token, and nothing else will say so.
            logger.error(
                "bot %s has a token that cannot be decrypted; mint a new one",
                bot.handle,
            )
            return ""

    # -------------------------------------------------------------- sweep

    async def sweep(self):
        try:
            bots = await asyncio.to_thread(self._candidates)
        except Exception:
            logger.exception("could not list bots to run")
            return

        seen = set()
        for bot in bots:
            bot_id = str(bot.pk)
            seen.add(bot_id)

            if bot_id in self._refused:
                continue
            if bot_id in self.workers:
                worker = self.workers[bot_id]
                if worker.fatal:
                    self._refused[bot_id] = worker.fatal
                    await self._drop(bot_id, release=True)
                continue

            try:
                won = await asyncio.to_thread(self.leases.acquire, bot_id)
            except Exception:
                logger.exception("could not acquire a lease for %s", bot.handle)
                continue
            if not won:
                continue

            token = await asyncio.to_thread(self._token_for, bot)
            if not token:
                await asyncio.to_thread(self.leases.release, bot_id)
                continue

            worker = BotWorker(bot, token, int(now().timestamp() * 1000))
            worker.start()
            self.workers[bot_id] = worker
            logger.info("running bot @%s", bot.handle)

        # A bot that stopped being a candidate - switched offline, deactivated,
        # unbound from its agent, token revoked - is dropped and its lease
        # released, so the row in Redis does not outlive the reason for it.
        #
        # This is how the on/off switch takes effect: `stop()` closes the SSE
        # connection, and releasing the lease means a peer can pick the bot up
        # the moment it is switched back on, rather than waiting out a TTL.
        for bot_id in list(self.workers):
            if bot_id not in seen:
                logger.info("bot %s is no longer runnable; stopping", bot_id)
                await self._drop(bot_id, release=True)

    async def _drop(self, bot_id, release):
        worker = self.workers.pop(bot_id, None)
        if worker is not None:
            await worker.stop()
        if release:
            await asyncio.to_thread(self.leases.release, bot_id)

    async def renew(self):
        """Renew every lease, and stop any bot we lost.

        Losing a lease is not a retryable error: another supervisor holds the
        bot and may already be answering for it, so continuing to consume would
        be the double-reply this whole mechanism exists to prevent.
        """
        try:
            lost = await asyncio.to_thread(self.leases.renew_all)
        except Exception:
            logger.exception("could not renew leases")
            return
        for bot_id in lost:
            logger.warning("lost the lease for %s; stopping it", bot_id)
            await self._drop(bot_id, release=False)

    def _wake(self):
        """Called from the control-channel thread, so it hops to the loop."""
        try:
            self._loop.call_soon_threadsafe(self._nudged.set)
        except RuntimeError:
            # The loop is closing. The nudge is redundant at that point.
            pass

    async def _wait(self, seconds):
        """Sleep, unless somebody switches a bot in the meantime.

        Returns True when it was a nudge rather than the timeout, so the caller
        knows to sweep straight away.
        """
        try:
            await asyncio.wait_for(self._nudged.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return False
        self._nudged.clear()
        return True

    async def run(self):
        logger.info("supervisor starting (owner %s)", self.leases.owner)
        self._loop = asyncio.get_running_loop()

        if not self.once:
            self._listener = asyncio.create_task(
                asyncio.to_thread(listen, self._wake, lambda: self._running),
                name="bot-control",
            )

        try:
            while self._running:
                await self.sweep()
                if self.once:
                    return
                # Renewal has to happen more often than the sweep, since the
                # lease TTL is shorter than the sweep interval would allow.
                elapsed = 0.0
                while self._running and elapsed < self.sweep_interval:
                    nudged = await self._wait(
                        min(RENEW_INTERVAL_SECONDS, self.sweep_interval)
                    )
                    await self.renew()
                    if nudged:
                        # Somebody switched a bot. Sweep now rather than
                        # finishing out the interval.
                        break
                    elapsed += RENEW_INTERVAL_SECONDS
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Stop everything and give the leases back.

        Released rather than left to expire: without this, every deploy leaves
        each bot unheld for up to a lease TTL, which is a gap where nobody is
        listening.
        """
        self._running = False
        if self._listener is not None:
            self._listener.cancel()
            try:
                await self._listener
            except (asyncio.CancelledError, Exception):
                pass
            self._listener = None
        for bot_id in list(self.workers):
            await self._drop(bot_id, release=False)
        try:
            await asyncio.to_thread(self.leases.release_all)
        except Exception:
            logger.exception("could not release leases on shutdown")
        logger.info("supervisor stopped")

    def stop(self):
        self._running = False
