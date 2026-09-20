"""The per-bot control endpoint: run a bot, or stop it, from anywhere.

WHAT THIS IS
------------
A small developer API. Every bot has one URL and one key, issued when the bot
is minted and shown in the dashboard to copy. Anything that can make an HTTPS
request can use it:

    POST https://neon.example/api/chatterloop/bots/<id>/control?action=wake
    Authorization: Bearer <the bot's control key>

NOT A COMMAND
-------------
Deliberately a different mechanism. A `/command` is delivered to a bot on its
OWN event stream, and an offline bot has no stream - so `/wake` as a bot
command could never reach the one bot that needs to hear it. chatterloop owns
`/wake` internally and calls this; so can a cron job, a deploy script, or
somebody with curl.

The other direction is already covered too: Neon's own webhooks are Tools, so
this is not a second webhook system - it is the control plane for a bot's
presence, and nothing else.

WHY A KEY PER BOT AND NOT A SESSION
-----------------------------------
The caller is another service, or a script, acting for somebody Neon never
sees. It holds no Neon session, and it should not need one: the key scopes
authority to exactly one bot and exactly one power - whether that bot's
supervisor holds a stream. Losing it is a bot somebody can wake and sleep, not
an account.

Stored encrypted with the same Fernet key as a bot's token secret.
"""

import logging
import secrets

from django.urls import reverse

from neon.utils.crypto import decrypt, encrypt

logger = logging.getLogger(__name__)

WAKE = "wake"
SLEEP = "sleep"
ACTIONS = (WAKE, SLEEP)


def generate_control_key():
    return secrets.token_urlsafe(32)


def ensure_control_key(bot):
    """The bot's control key, minting one on first use.

    Returns the plaintext, which is what the dashboard shows to copy.

    A key that cannot be decrypted is REPLACED rather than trusted: whatever
    the holder has would no longer match anything Neon can check, so the only
    honest outcome is a new key and a copy that works.
    """
    existing = (bot.control_key_encrypted or "").strip()
    if existing:
        try:
            return decrypt(existing)
        except Exception:
            logger.warning(
                "bot @%s: the control key could not be decrypted; issuing a "
                "new one, so any copy of the old key stops working",
                bot.handle,
            )

    key = generate_control_key()
    bot.control_key_encrypted = encrypt(key)
    bot.save(update_fields=["control_key_encrypted", "updated_at"])
    return key


def control_path(bot):
    """The endpoint's path.

    From `reverse()`, never a literal: a hand-written path here would be
    copied into somebody else's configuration and 404 there rather than here.

    `bots:control`, not a chatterloop route - see bot_api_urls.py for why the
    public surface is addressed neutrally.
    """
    return reverse("bots:control", args=[bot.pk])


def control_url(bot, base=""):
    """The full URL to copy, when a base is known.

    `base` is passed in rather than read from settings so the dashboard can
    build it from the REQUEST - the host somebody is actually using - and fall
    back to NEON_PUBLIC_BASE_URL only when there is no request to ask.
    """
    base = str(base or "").rstrip("/")
    return f"{base}{control_path(bot)}" if base else control_path(bot)


def resolve_action(value):
    """`wake` | `sleep`, or None.

    Case-insensitive and whitespace-tolerant, because this value is typed into
    a config box by hand as often as it is generated.
    """
    action = str(value or "").strip().lower()
    return action if action in ACTIONS else None
