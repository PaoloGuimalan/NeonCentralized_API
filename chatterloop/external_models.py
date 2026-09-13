"""Chatterloop's own tables, as Neon sees them.

EVERY MODEL HERE IS `managed = False`
-------------------------------------
These rows belong to `user_service`, which owns their schema and their
migrations. Declaring them unmanaged means Django gives us the ORM - queries,
joins, `select_related` - while `makemigrations` ignores them entirely, so Neon
can never generate a migration that alters somebody else's database. The
router (routers.py) refuses `allow_migrate` for this alias on top of that, so
the protection does not depend on remembering `managed = False` on a future
model.

`db_table` is pinned explicitly on all of them. Django would otherwise derive
it from this app's label and get `chatterloop_account` rather than
`user_account`, and the physical name is the part that has to match.

READ VS WRITE
-------------
The models in this file split into two groups, and the split is worth keeping
in mind when adding to it:

  * IDENTITY (Account, Realm, RealmMember) - read only. Neon resolves who
    somebody is and which pages they may act as. Nothing here is ever written.
  * PROVISIONING (Entity, Bot, Token, EntityPermission) - written, and only by
    chatterloop/provisioning.py, inside one transaction.

Schema mirrored from user_service: `user/models.py`, `community/models.py`,
`entity/models.py` and `bot/models.py`. Only the columns Neon actually uses are
declared - an unmanaged model does not need to describe every column, and
listing ones we never read would be four more things to keep in step for no
benefit.
"""

from django.db import models


class Entity(models.Model):
    """`entity_entity` - the thing a user, a page or a bot all *are*.

    Every identity on chatterloop resolves to one of these, which is why
    `bot_bot.owner_entity` can point at a person or at a page without caring
    which.
    """

    USER = "user"
    BOT = "bot"
    REALM = "realm"

    id = models.CharField(max_length=40, primary_key=True)
    type = models.CharField(max_length=150)
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "entity_entity"

    def __str__(self):
        return f"{self.type}:{self.id}"


class Account(models.Model):
    """`user_account` - a person on chatterloop.

    Read only. Neon never writes here: credentials, verification and
    compliance are user_service's, and sign-in goes through its HTTP endpoint
    precisely so those rules keep applying.

    `password` is deliberately NOT declared. Neon has the database access to
    read it and no reason to: verifying it here would bypass chatterloop's own
    lockout and compliance checks, and a column that is never declared cannot
    be selected by accident.
    """

    id = models.CharField(max_length=150, primary_key=True)
    entity = models.OneToOneField(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="chatterloop_account",
    )
    username = models.CharField(max_length=150)
    first_name = models.CharField(max_length=150)
    middle_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.EmailField()
    profile = models.CharField(max_length=500)
    is_active = models.BooleanField()
    is_verified = models.BooleanField()
    join_type = models.CharField(max_length=150)

    class Meta:
        managed = False
        db_table = "user_account"

    @property
    def full_name(self):
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p and p != "N/A").strip()

    def __str__(self):
        return self.username


class Realm(models.Model):
    """`community_realm` - a page, server, group, conference, channel or voice.

    Only `type="page"` is relevant to Neon, because a page is the only realm
    chatterloop itself lets a person act as (see `RealmMember`).
    """

    PAGE = "page"

    id = models.CharField(max_length=150, primary_key=True)
    realm_id = models.CharField(max_length=150)
    entity = models.OneToOneField(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="chatterloop_realm",
    )
    name = models.CharField(max_length=150)
    slug = models.TextField(null=True)
    profile = models.CharField(max_length=500)
    description = models.TextField(null=True)
    type = models.CharField(max_length=150)
    is_active = models.BooleanField()

    class Meta:
        managed = False
        db_table = "community_realm"

    def __str__(self):
        return f"{self.name} ({self.type})"


class RealmMember(models.Model):
    """`community_member` - an entity's membership of a realm, with its role.

    This is the table that decides whether Neon may let somebody connect a
    page. The rule is not Neon's invention: `EntitySwitch` in
    `user/entity_switch_views.py` permits acting as a realm only when the
    realm is a page AND the caller's role here is owner or admin. Neon applies
    the same condition and adds nothing to it, so the two can never disagree
    about who may speak for a page.
    """

    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"

    # Roles chatterloop lets an account ACT AS the realm. Deliberately not
    # "can post in it" - moderating a page is not the same authority as being
    # able to publish a bot under its name.
    ACTING_ROLES = (OWNER, ADMIN)

    member_id = models.CharField(max_length=150, primary_key=True)
    entity = models.ForeignKey(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="realm_memberships",
    )
    realm = models.ForeignKey(
        Realm,
        on_delete=models.DO_NOTHING,
        db_column="realm_id",
        related_name="members",
    )
    nickname = models.CharField(max_length=150, null=True)
    role = models.CharField(max_length=150)
    date_joined = models.DateTimeField(null=True)

    class Meta:
        managed = False
        db_table = "community_member"

    def __str__(self):
        return f"{self.entity_id} @ {self.realm_id} ({self.role})"


# ---------------------------------------------------------------------------
# PROVISIONING
#
# The four tables Neon WRITES, and the only ones. Written exclusively by
# chatterloop/provisioning.py, inside one transaction on the chatterloop
# connection.
#
# Schema mirrored from user_service `entity/models.py` and `bot/models.py`, and
# cross-checked against developer_service's own reader
# (`internal/auth/token.go`), which is what actually consumes these rows. Where
# the two could drift the Go reader wins, because it is the thing that has to
# accept what Neon writes.
# ---------------------------------------------------------------------------


class Bot(models.Model):
    """`bot_bot` - a bot's public identity.

    `entity` is the key fact; everything else is presentation. `owner_entity`
    is an FK to Entity rather than to an account, which is exactly what lets a
    PAGE own a bot - the payoff described in the plan, and the reason a bot can
    outlive its creator leaving.

    HANDLE UNIQUENESS IS NARROWER THAN IT LOOKS
    -------------------------------------------
    `handle` is unique across BOTS only. Accounts carry theirs as
    `user_account.username` and realms as `community_realm.slug`, with no
    shared registry between the three - user_service's own comment calls this
    "a real gap". So the database will happily accept a bot whose handle is
    already somebody's username, and the consequence is not cosmetic:
    developer_service resolves a handle by UNION across all three tables and
    keeps the first match, so mention matching for one of them silently stops
    working. `provisioning.handle_conflict()` checks all three before minting.
    """

    id = models.CharField(max_length=40, primary_key=True)
    entity = models.OneToOneField(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="bot",
    )
    name = models.CharField(max_length=80)
    handle = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, default="")
    # A CDN URL or the string sentinel "none", matching Account.profile and
    # Realm.profile. Nothing on this platform stores an actual file here.
    profile = models.CharField(max_length=500, default="none")
    owner_entity = models.ForeignKey(
        Entity,
        null=True,
        on_delete=models.DO_NOTHING,
        db_column="owner_entity_id",
        related_name="owned_bots",
    )
    # Platform-owned. Neon writes False, always - the moderator is the only
    # system bot and Neon has no business minting another.
    is_system = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "bot_bot"

    def __str__(self):
        return f"{self.name} (@{self.handle})"


class Token(models.Model):
    """`entity_token` - a long-lived API credential.

    NEON HOLDS THE ONLY COPY OF THE SECRET
    --------------------------------------
    This table stores `token_hash` - SHA-256 of the whole
    `clt_<prefix>_<secret>` string - and nothing reversible. That is deliberate
    on chatterloop's side: the token is returned once at issue and is
    unrecoverable afterwards. Since Neon's bot runtime has to present it on
    every reconnect, Neon keeps the plaintext encrypted in its own
    `ChatterloopToken.secret_encrypted`. Lose that and the only remedy is
    minting a replacement.

    `scopes` IS A CEILING, NOT A GRANT
    ----------------------------------
    A request is allowed only if the codename is in `scopes` AND the owning
    entity has an explicit grant row. Writing scopes without the matching
    `EntityPermission` rows issues a token that authenticates perfectly and is
    refused by every route - with a 403 identical to the one the reverse
    mistake produces. provisioning.py always writes both.
    """

    id = models.CharField(max_length=40, primary_key=True)
    entity = models.ForeignKey(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="tokens",
    )
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, default="")
    # The clear-text lookup half. Unique, so a collision is an error at issue
    # rather than an ambiguous verify later.
    prefix = models.CharField(max_length=16, unique=True)
    # SHA-256 hex of the full token string - exactly 64 characters.
    token_hash = models.CharField(max_length=64)
    scopes = models.JSONField(default=list)
    # NULL means not realm-confined. Neon writes NULL even for a page-owned
    # bot: confining the token to that realm would stop it answering a DM.
    realm = models.ForeignKey(
        Realm,
        null=True,
        on_delete=models.DO_NOTHING,
        db_column="realm_id",
        related_name="entity_tokens",
    )
    created_by = models.ForeignKey(
        Entity,
        null=True,
        on_delete=models.DO_NOTHING,
        db_column="created_by_id",
        related_name="tokens_created",
    )
    created_at = models.DateTimeField()
    expires_at = models.DateTimeField(null=True)
    last_used_at = models.DateTimeField(null=True)
    # Revocation is a timestamp rather than a delete, so an incident keeps its
    # audit trail: which token, whose, and when it was cut off.
    revoked_at = models.DateTimeField(null=True)
    is_active = models.BooleanField(default=True)
    # Set together or left NULL together - user_service's Token.clean()
    # enforces the pairing, and a bare number with no unit is not a rate limit.
    # Both NULL means unlimited, which is what Neon writes.
    rate_limit_int = models.PositiveIntegerField(null=True)
    rate_limit_type = models.CharField(max_length=10, null=True)

    class Meta:
        managed = False
        db_table = "entity_token"

    def __str__(self):
        return f"{self.name} ({self.prefix}...)"


class EntityPermission(models.Model):
    """`entity_entitypermission` - an explicit grant or deny for one entity.

    WHY NEON HAS TO WRITE THESE AT ALL
    ----------------------------------
    The five developer-API codenames are deliberately absent from the
    platform's default predicate, which makes them deny-by-default.
    user_service's catalog comment is explicit about why: seeding them the
    usual way "would have handed them to every bot on the platform the moment
    they were created". So a freshly minted bot holds none of them, and a token
    carrying the scopes without these rows authenticates and is then refused by
    every route.

    `realm` is NULL because all five are global-scoped. Note that Postgres
    treats NULLs as distinct in a unique index, so the (entity, permission,
    realm) constraint does NOT in fact prevent duplicate global rows -
    provisioning.py checks before inserting rather than relying on it.
    """

    GRANT = "grant"
    DENY = "deny"

    id = models.CharField(max_length=40, primary_key=True)
    entity = models.ForeignKey(
        Entity,
        on_delete=models.DO_NOTHING,
        db_column="entity_id",
        related_name="permission_overrides",
    )
    permission = models.CharField(max_length=150)
    effect = models.CharField(max_length=10)
    realm = models.ForeignKey(
        Realm,
        null=True,
        on_delete=models.DO_NOTHING,
        db_column="realm_id",
        related_name="permission_overrides",
    )
    reason = models.TextField(null=True, default=None)
    created_by = models.ForeignKey(
        Entity,
        null=True,
        on_delete=models.DO_NOTHING,
        db_column="created_by_id",
        related_name="permission_overrides_created",
    )
    created_at = models.DateTimeField()
    expires_at = models.DateTimeField(null=True)

    class Meta:
        managed = False
        db_table = "entity_entitypermission"

    def __str__(self):
        return f"{self.effect} {self.permission} -> {self.entity_id}"
