"""Minting bots and tokens into chatterloop's database.

THE TOKEN FORMAT, AND WHY IT IS SAFE TO RE-IMPLEMENT
---------------------------------------------------
    clt_<12 hex prefix>_<64 hex secret>

stored as a plain unsalted SHA-256 of the WHOLE string. There is no salt, no
stretching and no framework-specific encoding, which is what makes a fourth
implementation tolerable - user_service's Python is canonical, Node has a
hand-port, developer_service has a Go one, and this is the fourth. The format
was chosen to be trivially re-implementable precisely so that could happen
without a subtle disagreement. Verified against
`developer_service/internal/auth/token.go`.

A plain hash is correct HERE and would be wrong for a password: the secret is
32 bytes from `secrets`, so there is no dictionary to run and no work factor
that would meaningfully slow an attacker who already has the column.

AUTHORIZATION IS AN INTERSECTION
--------------------------------
developer_service allows a request only when the codename is in the token's
`scopes` AND the owning entity holds an explicit `entity_entitypermission`
grant. Writing one without the other issues a credential that authenticates
perfectly and is refused by every route, with a 403 that looks identical
either way. So `mint_bot` always writes both, in one transaction.

The five codenames are deny-by-default on purpose. user_service's catalog says
why: every other global permission resolves through a predicate that returns
True for any entity without an Account - bots and realms both - so seeding
these the usual way would have handed them to every bot on the platform.

ORDERING ACROSS TWO DATABASES
-----------------------------
`transaction.atomic(using="chatterloop")` makes the chatterloop side
all-or-nothing. It cannot include Neon's own rows - no transaction spans two
connections - so Neon writes its record FIRST, in `status=provisioning`, with
the ids it generated. A failure then leaves a Neon row naming exactly which
chatterloop entity to clean up. The reverse order would leave a live
credential nothing knows about, and therefore nothing can revoke.
"""

import hashlib
import logging
import secrets
import uuid

from django.db import transaction
from django.utils.timezone import now

from .external_models import (
    Account,
    Bot,
    Entity,
    EntityPermission,
    Realm,
    Token,
)
from .models import ChatterloopBot, ChatterloopToken

logger = logging.getLogger(__name__)

TOKEN_NAMESPACE = "clt"
PREFIX_BYTES = 6  # 12 hex characters
SECRET_BYTES = 32  # 64 hex characters

# The capabilities developer_service gates. Every one is GLOBAL-scoped in the
# platform catalog (user_service/entity/permissions.py); a realm-scoped one
# would need the role matrix, which that service deliberately does not carry.
SCOPE_EVENTS_SUBSCRIBE = "events.subscribe"
SCOPE_MESSAGES_READ = "messages.read"
SCOPE_NOTIFICATIONS_READ = "notifications.read"
SCOPE_MESSAGES_SEND = "messages.send"
SCOPE_COMMENTS_CREATE = "comments.create"

ALL_SCOPES = (
    SCOPE_EVENTS_SUBSCRIBE,
    SCOPE_MESSAGES_READ,
    SCOPE_NOTIFICATIONS_READ,
    SCOPE_MESSAGES_SEND,
    SCOPE_COMMENTS_CREATE,
)

# What a conversational bot needs to do its job: hear about events, read the
# thread it was mentioned in, and answer. A narrower default than "everything"
# would mean a bot that silently cannot reply.
DEFAULT_SCOPES = ALL_SCOPES


class ProvisioningError(Exception):
    """Minting could not proceed. The message is shown to the user."""


class HandleUnavailable(ProvisioningError):
    """The requested handle is taken by a bot, a person or a page."""


def generate_token():
    """A fresh credential. Returns `(prefix, secret, token, token_hash)`."""
    prefix = secrets.token_hex(PREFIX_BYTES)
    secret = secrets.token_hex(SECRET_BYTES)
    token = f"{TOKEN_NAMESPACE}_{prefix}_{secret}"
    return prefix, secret, token, hash_token(token)


def hash_token(token):
    """SHA-256 hex of the whole token string, matching every other
    implementation. Not a password hash - see the module docstring."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def handle_conflict(handle):
    """Who already answers to `@handle`, or None.

    Checks all THREE handle-bearing tables, not just `bot_bot`. The unique
    constraint covers bots only, so the database would accept a bot named after
    an existing person - and developer_service resolves a handle by UNION
    across `user_account.username`, `community_realm.slug` and `bot_bot.handle`,
    keeping the FIRST match. The loser of that race silently stops matching
    mentions, with nothing in any log to explain it.

    Returns a human-readable description of the clash so the message can say
    what kind of thing took the name, without naming the account itself.
    """
    handle = (handle or "").strip()
    if not handle:
        return "an empty handle is not usable"

    if Bot.objects.filter(handle__iexact=handle).exists():
        return "another bot"
    if Account.objects.filter(username__iexact=handle).exists():
        return "a Chatterloop account"
    if Realm.objects.filter(slug__iexact=handle).exists():
        return "a Chatterloop page"
    return None


def validate_scopes(scopes):
    """Reject anything outside the five developer_service can decide.

    It refuses an unknown codename rather than guessing, so a typo here would
    produce a token that authenticates and is then refused - caught at mint
    time instead.
    """
    if not scopes:
        return list(DEFAULT_SCOPES)
    unknown = [scope for scope in scopes if scope not in ALL_SCOPES]
    if unknown:
        raise ProvisioningError(
            "Unknown scope(s): "
            + ", ".join(sorted(unknown))
            + ". Valid scopes are: "
            + ", ".join(ALL_SCOPES)
        )
    # Deduplicated and ordered so the stored value is stable and comparable.
    return [scope for scope in ALL_SCOPES if scope in set(scopes)]


def _grant_scopes(entity_id, scopes, created_by_entity_id=None):
    """One `effect='grant'` row per scope, global-scoped.

    Checked before inserting rather than relying on the unique constraint:
    Postgres treats NULLs as distinct in a unique index, so
    (entity, permission, NULL) does not actually collide with itself and a
    re-mint would quietly accumulate duplicate rows.
    """
    existing = set(
        EntityPermission.objects.filter(
            entity_id=entity_id, permission__in=scopes, realm_id=None
        ).values_list("permission", flat=True)
    )
    stamp = now()
    EntityPermission.objects.bulk_create(
        [
            EntityPermission(
                id=str(uuid.uuid4()),
                entity_id=entity_id,
                permission=scope,
                effect=EntityPermission.GRANT,
                realm_id=None,
                reason="Issued by Neon for a developer bot.",
                created_by_id=created_by_entity_id,
                created_at=stamp,
                expires_at=None,
            )
            for scope in scopes
            if scope not in existing
        ]
    )


@transaction.atomic(using="chatterloop")
def _write_chatterloop_bot(
    *,
    entity_id,
    bot_id,
    token_id,
    name,
    handle,
    description,
    owner_entity_id,
    token_name,
    token_hash,
    prefix,
    scopes,
    expires_at,
    created_by_entity_id,
):
    """Everything that goes into chatterloop, all-or-nothing.

    Atomic because a half-written bot is worse than no bot: an Entity with no
    Bot row renders as a raw uuid everywhere, and a Token whose grants did not
    land authenticates and then fails every request.
    """
    stamp = now()

    Entity.objects.create(id=entity_id, type=Entity.BOT, created_at=stamp)

    Bot.objects.create(
        id=bot_id,
        entity_id=entity_id,
        name=name[:80],
        handle=handle[:50],
        description=description or "",
        profile="none",
        owner_entity_id=owner_entity_id,
        # Never True from here. The moderator is the platform's own bot and
        # Neon has no business minting another one.
        is_system=False,
        is_verified=False,
        is_active=True,
        created_at=stamp,
    )

    Token.objects.create(
        id=token_id,
        entity_id=entity_id,
        name=token_name[:120],
        description="Issued by Neon.",
        prefix=prefix,
        token_hash=token_hash,
        scopes=list(scopes),
        # NULL: a page-owned bot still acts globally, and confining the token
        # to that realm would stop it answering a direct message.
        realm_id=None,
        created_by_id=created_by_entity_id,
        created_at=stamp,
        expires_at=expires_at,
        revoked_at=None,
        is_active=True,
        # Both NULL means unlimited. Set together or not at all - a bare number
        # with no unit is not a rate limit.
        rate_limit_int=None,
        rate_limit_type=None,
    )

    _grant_scopes(entity_id, scopes, created_by_entity_id)


def mint_bot(
    *,
    organization,
    created_by,
    name,
    handle,
    owner_entity_id,
    agent=None,
    model=None,
    provider_credential=None,
    connected_account=None,
    description="",
    scopes=None,
    expires_at=None,
):
    """Create a bot on chatterloop and record it in Neon.

    Returns `(ChatterloopBot, ChatterloopToken)`. The token's plaintext is
    available as `token.token` and is shown to the user exactly once.
    """
    scopes = validate_scopes(scopes)

    conflict = handle_conflict(handle)
    if conflict:
        raise HandleUnavailable(
            f"@{handle} is already used by {conflict}. Choose another handle."
        )

    # Every id is generated HERE, before anything is written, so Neon's record
    # can name the chatterloop rows even if the chatterloop write fails.
    entity_id = str(uuid.uuid4())
    bot_id = str(uuid.uuid4())
    token_id = str(uuid.uuid4())
    prefix, secret, token_value, token_hash = generate_token()

    bot = ChatterloopBot.objects.create(
        organization=organization,
        created_by=created_by,
        entity_id=entity_id,
        bot_id=bot_id,
        name=name,
        handle=handle,
        description=description or "",
        owner_entity_id=owner_entity_id,
        connected_account=connected_account,
        agent=agent,
        model=model,
        provider_credential=provider_credential,
        status=ChatterloopBot.STATUS_PROVISIONING,
    )

    credential = ChatterloopToken(
        bot=bot,
        token_id=token_id,
        prefix=prefix,
        name=f"{name} (Neon)",
        scopes=scopes,
        expires_at=expires_at,
    )
    credential.set_secret(secret)
    credential.save()

    try:
        _write_chatterloop_bot(
            entity_id=entity_id,
            bot_id=bot_id,
            token_id=token_id,
            name=name,
            handle=handle,
            description=description,
            owner_entity_id=owner_entity_id,
            token_name=credential.name,
            token_hash=token_hash,
            prefix=prefix,
            scopes=scopes,
            expires_at=expires_at,
            created_by_entity_id=getattr(created_by, "entity_id", None),
        )
    except Exception as ex:
        logger.exception("failed to mint chatterloop bot %s", entity_id)
        bot.status = ChatterloopBot.STATUS_FAILED
        # Recorded on the row rather than only logged: the person who clicked
        # the button is the one who needs to know, and "it didn't work" with
        # the reason only in a log file is how this becomes a support ticket.
        bot.status_reason = str(ex)[:1000]
        bot.save(update_fields=["status", "status_reason", "updated_at"])
        raise ProvisioningError(f"Could not create the bot on Chatterloop: {ex}") from ex

    stamp = now()
    bot.status = ChatterloopBot.STATUS_ACTIVE
    bot.status_reason = ""
    bot.save(update_fields=["status", "status_reason", "updated_at"])

    credential.provisioned_at = stamp
    credential.save(update_fields=["provisioned_at"])

    # Carried on the instance only, never persisted in the clear. The caller
    # shows it once; after that `token.token` decrypts it from storage.
    credential.plaintext = token_value
    return bot, credential


def rotate_token(bot, name=None, scopes=None, expires_at=None):
    """Issue a replacement credential for an existing bot.

    The previous tokens are NOT revoked automatically. Rotation is normally
    "start using the new one, then cut off the old" - revoking first would take
    the bot offline for however long it takes to redeploy - so revoking is a
    separate, deliberate step.
    """
    if bot.status not in (ChatterloopBot.STATUS_ACTIVE, ChatterloopBot.STATUS_DEACTIVATED):
        raise ProvisioningError("This bot was never fully provisioned.")

    scopes = validate_scopes(scopes)
    token_id = str(uuid.uuid4())
    prefix, secret, token_value, token_hash = generate_token()

    credential = ChatterloopToken(
        bot=bot,
        token_id=token_id,
        prefix=prefix,
        name=name or f"{bot.name} (Neon)",
        scopes=scopes,
        expires_at=expires_at,
    )
    credential.set_secret(secret)
    credential.save()

    try:
        with transaction.atomic(using="chatterloop"):
            Token.objects.create(
                id=token_id,
                entity_id=bot.entity_id,
                name=credential.name[:120],
                description="Issued by Neon.",
                prefix=prefix,
                token_hash=token_hash,
                scopes=list(scopes),
                realm_id=None,
                created_by_id=getattr(bot.created_by, "entity_id", None),
                created_at=now(),
                expires_at=expires_at,
                revoked_at=None,
                is_active=True,
                rate_limit_int=None,
                rate_limit_type=None,
            )
            # A rotation may widen the scopes, and a scope with no grant is a
            # silent 403 - so the grants are re-checked every time, not only at
            # first mint.
            _grant_scopes(
                bot.entity_id, scopes, getattr(bot.created_by, "entity_id", None)
            )
    except Exception as ex:
        logger.exception("failed to rotate token for bot %s", bot.entity_id)
        credential.delete()
        raise ProvisioningError(f"Could not issue a new token: {ex}") from ex

    credential.provisioned_at = now()
    credential.save(update_fields=["provisioned_at"])
    credential.plaintext = token_value
    return credential


def revoke_token(credential, reason=""):
    """Cut off one credential, on both sides.

    Chatterloop first. If Neon's update then failed, the credential is already
    dead and Neon merely looks out of date - the harmless direction. The
    reverse would leave Neon showing "revoked" for a token that still works.
    """
    stamp = now()

    if credential.provisioned_at is not None:
        with transaction.atomic(using="chatterloop"):
            Token.objects.filter(id=credential.token_id).update(
                revoked_at=stamp, is_active=False
            )

    credential.revoked_at = stamp
    credential.save(update_fields=["revoked_at"])
    logger.info(
        "revoked token %s for bot %s (%s)",
        credential.prefix,
        credential.bot.entity_id,
        reason or "no reason given",
    )
    return credential


def deactivate_bot(bot, reason=""):
    """Take a bot offline and revoke every credential it holds.

    Used when an agent is unbound, a user deletes the bot, or the identity it
    speaks as is disconnected. Deactivating without revoking would leave a
    working credential for a bot the user believes is off.
    """
    for credential in bot.tokens.filter(revoked_at__isnull=True):
        revoke_token(credential, reason=reason or "bot deactivated")

    if bot.status == ChatterloopBot.STATUS_ACTIVE:
        with transaction.atomic(using="chatterloop"):
            Bot.objects.filter(id=bot.bot_id).update(is_active=False)

    bot.status = ChatterloopBot.STATUS_DEACTIVATED
    bot.status_reason = reason
    bot.save(update_fields=["status", "status_reason", "updated_at"])
    return bot


def reactivate_bot(bot):
    """Bring a deactivated bot back. Does NOT restore its tokens.

    Revocation is permanent by design - a revoked credential may have leaked,
    and un-revoking it would defeat the point - so a reactivated bot needs a
    fresh token before it can do anything.
    """
    if bot.status != ChatterloopBot.STATUS_DEACTIVATED:
        raise ProvisioningError("That bot is not deactivated.")

    with transaction.atomic(using="chatterloop"):
        Bot.objects.filter(id=bot.bot_id).update(is_active=True)

    bot.status = ChatterloopBot.STATUS_ACTIVE
    bot.status_reason = ""
    bot.save(update_fields=["status", "status_reason", "updated_at"])
    return bot


def grant_report(entity_id, scopes):
    """Which of `scopes` the entity actually holds a live grant for.

    The other half of the intersection, read straight from the table
    developer_service reads. A scope on a token with no grant here is a silent
    403 at request time; this is what lets the UI say so beforehand.
    """
    rows = EntityPermission.objects.filter(
        entity_id=entity_id, permission__in=list(scopes), realm_id=None
    ).values_list("permission", "effect", "expires_at")

    stamp = now()
    granted, denied = set(), set()
    for permission, effect, expires_at in rows:
        if expires_at is not None and expires_at <= stamp:
            continue
        if effect == EntityPermission.DENY:
            denied.add(permission)
        elif effect == EntityPermission.GRANT:
            granted.add(permission)

    # An explicit deny is absolute and short-circuits a grant, matching the
    # resolver's first rule.
    effective = granted - denied
    return {
        "granted": sorted(effective),
        "missing": sorted(set(scopes) - effective),
    }
