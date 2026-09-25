"""Is the entity that just spoke a bot?

WHY THIS IS A LOOKUP AND NOT A SET
----------------------------------
The obvious version builds a set of Neon's own `ChatterloopBot.entity_id`
values and asks whether the author is in it. That answers a narrower question
than the one being asked: a realm can contain bots minted by somebody else, and
calling one of those a person is how a bot-to-bot loop gets past the toggle
that exists to prevent it.

`bot_bot` is chatterloop's own register of every bot on the platform, and Neon
already reads that database. So the question is answered where it is true
rather than where it is convenient.

WHY CACHING IS SAFE HERE
------------------------
An entity does not stop being a bot. `entity_entity.type` is set at creation
and the row is never converted, so a cached answer cannot go stale in the
direction that matters - the worst case is a bot minted after the cache filled,
which is a miss rather than a wrong answer, and a miss is a database read.

Bounded so a busy realm cannot grow it without limit, and kept per process:
this is asked once per frame on the supervisor's hot path, and a Redis round
trip per frame would cost more than the query it avoids.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

CACHE_SIZE = 4096

# The System bot, as every chatterloop service agrees on it: a fixed id written
# by user_service's bot/migrations/0004_system_bot.py, and the sender of every
# notice and built-in command answer (server/reusables/hooks/systemNotice.js,
# worker_service's systemreply.go).
SYSTEM_BOT_ENTITY_ID = "00000000-0000-4000-8000-000000000002"


def is_system_entity(entity_id):
    """True for the System bot. Bots never answer it.

    Deliberately NOT left to `is_bot_entity` and the `allow_bot_conversations`
    toggle. The System bot is a bot - it has a `bot_bot` row - so a bot allowed
    to talk to other bots would answer it. But nothing it posts is addressed to
    anyone: it is a command's result or a notice about one, and a bot replying
    to it is a bot replying to the platform.
    """
    return str(entity_id or "").strip().lower() == SYSTEM_BOT_ENTITY_ID


@lru_cache(maxsize=CACHE_SIZE)
def _lookup(entity_id):
    from .external_models import Bot

    try:
        return Bot.objects.filter(entity_id=entity_id).exists()
    except Exception:
        # The chatterloop database being unreachable must not decide that a
        # bot is a person - that is the direction where being wrong starts an
        # unbounded exchange. Refusing to cache means the next frame asks
        # again, which is what should happen once the database is back.
        logger.exception("could not determine whether %s is a bot", entity_id)
        raise


def is_bot_entity(entity_id):
    """True when this entity is a bot, False when it is not or cannot be told.

    Failing to False rather than raising, because this is consulted mid-frame
    and the alternative to an answer is dropping a message somebody sent. False
    is also the conservative direction for the toggle it feeds: it means "treat
    this as a person", and a person is subject to the ordinary hourly cap
    rather than the collaboration budget.
    """
    if not entity_id:
        return False
    try:
        return _lookup(str(entity_id))
    except Exception:
        return False


def forget():
    """Drop the cache. For tests, and for a process that has been told a bot
    was just minted."""
    _lookup.cache_clear()
