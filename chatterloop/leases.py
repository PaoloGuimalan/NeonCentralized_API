"""One consumer per bot, across every host.

WHY THIS IS NOT A SOCKET LOCK
-----------------------------
The reference implementation (rag_service/chatterloop/single_instance.py) binds
a TCP port derived from the bot's entity id. Binding is atomic, needs no
dependency, and correctly stops a second copy on the SAME machine - which is
the mistake it was written for, an old process left running in a background
terminal.

It cannot see a second machine. Neon runs supervisors as a scalable deployment,
so "two instances of this bot" is no longer a developer error, it is what
happens on any rollout where the old pod has not exited before the new one
starts. Two supervisors on one bot means every mention answered twice - which
is exactly what the reference's own docstring describes as having actually
happened, with different generated text under 100ms apart.

So the lock moves to Redis, where every host can see it.

HOW
---
    SET neon:bot:lease:<bot_id> <owner> NX EX 30

`NX` makes acquisition atomic - there is no window where two supervisors both
believe they won. The lease then has to be RENEWED every few seconds, which is
what makes a crash recoverable: a supervisor that dies stops renewing, the key
expires, and a peer picks the bot up on its next sweep. No health check, no
manual failover, and no shard configuration - N supervisors self-balance by
racing for whatever is unheld.

RENEWAL COMPARES THE OWNER
--------------------------
A plain `EXPIRE` would let a supervisor that lost its lease (paused long
enough for it to expire, then resumed) extend somebody else's. The renew and
release paths therefore run a Lua script that checks the value first, so both
are compare-and-act in one atomic step.
"""

import logging
import os
import socket
import uuid

logger = logging.getLogger(__name__)

# Long enough to survive a slow GC pause or a brief network stall, short enough
# that a crashed supervisor's bots are picked up quickly. Renewed at a third of
# this, so two consecutive renewal failures are survivable.
LEASE_TTL_SECONDS = 30
RENEW_INTERVAL_SECONDS = 10

# Renew only if we still hold it. Returns 1 when extended, 0 when the lease is
# gone or belongs to somebody else - which the caller must treat as losing the
# bot, not as a transient error.
_RENEW_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("expire", KEYS[1], ARGV[2])
end
return 0
"""

# Release only our own lease. Without the comparison, a supervisor shutting
# down after its lease had already expired and been taken would delete the new
# owner's key.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
end
return 0
"""


def owner_id():
    """Identifies this supervisor process.

    Host and pid so a lease in Redis can be traced back to something an
    operator can actually look at, plus a random suffix because a container
    restart can reuse a pid in a namespace.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def lease_key(bot_id):
    return f"neon:bot:lease:{bot_id}"


def running_bot_ids(bot_ids, client=None):
    """Which of these bots a supervisor currently holds a lease on.

    ONE round trip for the whole list, not one per bot: a listing renders every
    bot an organization has, and a per-bot check would turn one page into N
    Redis calls.

    Derived rather than stored, which is the point. A supervisor that crashed
    stops renewing and its lease expires, so this starts answering False on its
    own - there is no "actually running" column to go stale, and no cleanup job
    to forget to write.

    Returns an empty set if Redis is unreachable: the honest answer to "is
    anything running this?" when the thing that would know cannot be reached is
    not "yes".
    """
    bot_ids = [str(value) for value in bot_ids]
    if not bot_ids:
        return set()

    try:
        if client is None:
            from django_redis import get_redis_connection

            client = get_redis_connection("default")
        held = client.mget([lease_key(bot_id) for bot_id in bot_ids])
    except Exception:
        logger.warning("could not read bot leases; reporting none as running")
        return set()

    return {
        bot_id for bot_id, owner in zip(bot_ids, held or []) if owner is not None
    }


class LeaseManager:
    """Acquires and renews leases for one supervisor process."""

    def __init__(self, client, owner=None, ttl=LEASE_TTL_SECONDS):
        self.client = client
        self.owner = owner or owner_id()
        self.ttl = ttl
        self._held = set()

    @property
    def held(self):
        return frozenset(self._held)

    def acquire(self, bot_id):
        """Try to take a bot. True if we now hold it.

        Already holding it counts as success and refreshes the TTL, so a sweep
        that re-offers a bot we already run is harmless.
        """
        if bot_id in self._held:
            return self.renew(bot_id)

        won = self.client.set(lease_key(bot_id), self.owner, nx=True, ex=self.ttl)
        if won:
            self._held.add(bot_id)
            logger.info("acquired lease", extra={"bot_id": bot_id, "owner": self.owner})
        return bool(won)

    def renew(self, bot_id):
        """Extend a lease we hold. False means we no longer hold it.

        A False here is not a retryable error - another supervisor has the bot
        and may already be answering for it. The caller must stop consuming.
        """
        extended = self.client.eval(
            _RENEW_SCRIPT, 1, lease_key(bot_id), self.owner, self.ttl
        )
        if not extended:
            self._held.discard(bot_id)
            logger.warning(
                "lost lease", extra={"bot_id": bot_id, "owner": self.owner}
            )
        return bool(extended)

    def renew_all(self):
        """Renew everything we hold. Returns the set we LOST."""
        lost = set()
        for bot_id in list(self._held):
            if not self.renew(bot_id):
                lost.add(bot_id)
        return lost

    def release(self, bot_id):
        """Give a bot up, so a peer can take it immediately.

        Called on clean shutdown. Without it the bot is unavailable until the
        TTL expires, which turns every deploy into a gap of up to `ttl`
        seconds where nobody is listening.
        """
        self._held.discard(bot_id)
        try:
            self.client.eval(_RELEASE_SCRIPT, 1, lease_key(bot_id), self.owner)
        except Exception:
            # Best effort. The TTL is the backstop, so a failure here costs at
            # most `ttl` seconds of that bot being unheld.
            logger.warning("could not release lease", extra={"bot_id": bot_id})

    def release_all(self):
        for bot_id in list(self._held):
            self.release(bot_id)
