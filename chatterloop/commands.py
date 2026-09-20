"""What a bot answers to: its arsenal of `/commands`.

WHERE THE LIST COMES FROM
-------------------------
chatterloop's `bot_commands`, read through the external mirror. Not a
Neon-side list, because that is the same table `/help` reads and the same one
the composer's menu is built from - keeping a second copy here would mean the
menu could offer a command the bot does not answer, or the bot could answer
one nobody was told about.

ONLY THE `bot` CATEGORY
-----------------------
chatterloop runs `system` and `webhook` itself and never tells us. A `bot`
command is delivered ONLY as a field on the `messages_list` frame - no queue,
no delivery receipt - which is precisely why the bot has to know its own names
to recognise one.

EMPTY IS THE SAFE ANSWER
------------------------
A command frame reaches every bot in the conversation. A bot that treated an
unknown name as addressed would wake up for somebody else's command, and in a
room with several bots all of them would answer at once. So every failure here
- no row, a database that will not answer - returns nothing, and the bot stays
mention-only rather than becoming over-eager.
"""

import logging

logger = logging.getLogger(__name__)

# The category Neon is responsible for. See the module docstring.
BOT_CATEGORY = "bot"


def arsenals_for(bots):
    """Every bot's arsenal, in ONE query.

    The sweep refreshes each running bot on every interval, so a per-bot read
    here would be O(bots) round trips per sweep against ANOTHER SERVICE's
    database - the same cost `_candidates` prefetches its tokens to avoid, and
    worse for being cross-service. A sweep runs whether or not anything
    changed, so that load is constant rather than proportional to activity.

    Returns `{bot_id: frozenset(names)}`, with no entry for a bot that
    declares nothing. Callers use `.get(bot_id, frozenset())`.
    """
    from .external_models import BotCommand

    wanted = [b.bot_id for b in bots if getattr(b, "bot_id", "")]
    if not wanted:
        return {}

    try:
        rows = BotCommand.objects.filter(
            bot_id__in=wanted,
            category=BOT_CATEGORY,
            is_active=True,
        ).values_list("bot_id", "name")
    except Exception as ex:
        # Same rule as arsenal_for: never fatal. Every bot falls back to
        # mention-only rather than the sweep failing.
        logger.warning(
            "could not read arsenals (%s); bots will answer mentions only", ex
        )
        return {}

    grouped = {}
    for bot_id, name in rows:
        if name:
            grouped.setdefault(bot_id, set()).add(str(name).lower())
    return {bot_id: frozenset(names) for bot_id, names in grouped.items()}


def arsenal_for(bot):
    """The command names this bot declares, lowercased.

    `bot` is Neon's own ChatterloopBot row; `bot.bot_id` is the chatterloop
    `bot_bot.id` the commands hang off.

    Returns a frozenset so the runtime can hold it without copying, and so a
    caller cannot accidentally mutate one bot's arsenal through another's.
    """
    from .external_models import BotCommand

    if not getattr(bot, "bot_id", ""):
        return frozenset()

    try:
        names = BotCommand.objects.filter(
            bot_id=bot.bot_id,
            category=BOT_CATEGORY,
            is_active=True,
        ).values_list("name", flat=True)
        return frozenset(str(name).lower() for name in names if name)
    except Exception as ex:
        # Never fatal. A bot with no arsenal is mention-only, which is how
        # every bot worked before commands existed - far better than a
        # supervisor that will not start a bot because one read failed.
        logger.warning(
            "could not read the arsenal for @%s (%s); the bot will answer "
            "mentions only",
            getattr(bot, "handle", "?"),
            ex,
        )
        return frozenset()
