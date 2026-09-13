"""Telling running supervisors that something changed.

WHY THIS EXISTS
---------------
The supervisor sweeps for runnable bots on a timer, so switching a bot on or
off is eventually picked up without any signalling at all. "Eventually" is up
to a whole sweep interval, which for a button somebody just pressed reads as
the button not working.

So a toggle publishes here, and any supervisor listening sweeps immediately.

STRICTLY AN OPTIMISATION
------------------------
Nothing depends on the message arriving. Redis pub/sub is fire-and-forget: a
supervisor that is starting up, briefly disconnected, or simply not subscribed
misses it entirely - and then its next scheduled sweep converges on the same
answer. That property is what keeps this safe to add; a mechanism the system
NEEDED would have to be a queue with delivery guarantees, and would be a much
bigger thing than a button deserves.

Every function here swallows its own errors for the same reason. A Redis blip
must not fail the request that switched a bot off - the database row is the
source of truth, and it has already been written.
"""

import logging

logger = logging.getLogger(__name__)

CONTROL_CHANNEL = "neon:bots:control"


def _client():
    from django_redis import get_redis_connection

    return get_redis_connection("default")


def announce(bot_id="", reason=""):
    """Nudge every listening supervisor to re-sweep now.

    The payload is informational - a supervisor re-reads the database rather
    than trusting the message, so a stale or duplicated one costs one extra
    query and cannot cause a wrong decision.
    """
    try:
        _client().publish(CONTROL_CHANNEL, f"{bot_id}:{reason}"[:200])
    except Exception as ex:
        # Best effort by design. The scheduled sweep is the real mechanism.
        logger.debug("could not announce a bot change (%s); the sweep will catch it", ex)


def listen(on_message, should_continue):
    """Block on the control channel, calling `on_message` per nudge.

    Runs in a thread - the supervisor hands this to `asyncio.to_thread` -
    because redis-py's pub/sub is synchronous and blocking it on the event loop
    would stall every bot's stream.

    Returns when `should_continue()` goes false, or if Redis is unreachable. A
    supervisor with no control channel is degraded, not broken: it still sweeps
    on its timer.
    """
    try:
        pubsub = _client().pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(CONTROL_CHANNEL)
    except Exception as ex:
        logger.warning(
            "no control channel (%s); bot on/off changes will be picked up by "
            "the next sweep instead of immediately",
            ex,
        )
        return

    try:
        while should_continue():
            try:
                # A timeout rather than a blocking listen(), so shutdown does
                # not wait for the next message to arrive.
                message = pubsub.get_message(timeout=1.0)
            except Exception as ex:
                logger.warning("control channel read failed: %s", ex)
                return
            if message is not None:
                on_message()
    finally:
        try:
            pubsub.close()
        except Exception:
            pass
