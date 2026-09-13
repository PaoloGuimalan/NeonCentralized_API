"""A stand-in for chatterloop's schema, so provisioning can be tested for real.

WHY THIS EXISTS
---------------
The external models are `managed = False` and the router refuses `allow_migrate`
for that alias, so nothing creates their tables - which is exactly right in
production and leaves tests with nowhere to write. The alternative is mocking
the chatterloop side, and a mock cannot catch the failures that actually
matter here: a column Neon writes that does not exist, a NOT NULL it does not
fill, a transaction that does not roll back.

So the tables are created by hand against the test connection, from the same
column lists the models declare. This mirrors `user_service`'s schema; if that
schema changes, these tests are where the disagreement should surface.

Not a fixture of chatterloop's REAL schema - only the columns Neon touches,
plus the identity columns the handle check reads.
"""

from django.db import connections

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS entity_entity (
        id          VARCHAR(40) PRIMARY KEY,
        type        VARCHAR(150) NOT NULL,
        created_at  TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bot_bot (
        id               VARCHAR(40) PRIMARY KEY,
        entity_id        VARCHAR(40) NOT NULL UNIQUE,
        name             VARCHAR(80) NOT NULL,
        handle           VARCHAR(50) NOT NULL UNIQUE,
        description      TEXT NOT NULL DEFAULT '',
        profile          VARCHAR(500) NOT NULL DEFAULT 'none',
        owner_entity_id  VARCHAR(40) NULL,
        is_system        BOOL NOT NULL DEFAULT 0,
        is_verified      BOOL NOT NULL DEFAULT 0,
        is_active        BOOL NOT NULL DEFAULT 1,
        created_at       TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_token (
        id               VARCHAR(40) PRIMARY KEY,
        entity_id        VARCHAR(40) NOT NULL,
        name             VARCHAR(120) NOT NULL,
        description      TEXT NOT NULL DEFAULT '',
        prefix           VARCHAR(16) NOT NULL UNIQUE,
        token_hash       VARCHAR(64) NOT NULL,
        scopes           TEXT NOT NULL DEFAULT '[]',
        realm_id         VARCHAR(150) NULL,
        created_by_id    VARCHAR(40) NULL,
        created_at       TIMESTAMP NOT NULL,
        expires_at       TIMESTAMP NULL,
        last_used_at     TIMESTAMP NULL,
        revoked_at       TIMESTAMP NULL,
        is_active        BOOL NOT NULL DEFAULT 1,
        rate_limit_int   INTEGER NULL,
        rate_limit_type  VARCHAR(10) NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_entitypermission (
        id             VARCHAR(40) PRIMARY KEY,
        entity_id      VARCHAR(40) NOT NULL,
        permission     VARCHAR(150) NOT NULL,
        effect         VARCHAR(10) NOT NULL,
        realm_id       VARCHAR(150) NULL,
        reason         TEXT NULL,
        created_by_id  VARCHAR(40) NULL,
        created_at     TIMESTAMP NOT NULL,
        expires_at     TIMESTAMP NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_account (
        id          VARCHAR(150) PRIMARY KEY,
        entity_id   VARCHAR(40) NOT NULL UNIQUE,
        username    VARCHAR(150) NOT NULL,
        first_name  VARCHAR(150) NOT NULL DEFAULT '',
        middle_name VARCHAR(150) NOT NULL DEFAULT 'N/A',
        last_name   VARCHAR(150) NOT NULL DEFAULT '',
        email       VARCHAR(254) NOT NULL DEFAULT '',
        profile     VARCHAR(500) NOT NULL DEFAULT 'none',
        is_active   BOOL NOT NULL DEFAULT 1,
        is_verified BOOL NOT NULL DEFAULT 1,
        join_type   VARCHAR(150) NOT NULL DEFAULT 'system'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS community_realm (
        id          VARCHAR(150) PRIMARY KEY,
        realm_id    VARCHAR(150) NOT NULL,
        entity_id   VARCHAR(40) NOT NULL UNIQUE,
        name        VARCHAR(150) NOT NULL,
        slug        TEXT NULL,
        profile     VARCHAR(500) NOT NULL DEFAULT 'none',
        description TEXT NULL,
        type        VARCHAR(150) NOT NULL,
        is_active   BOOL NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS community_member (
        member_id   VARCHAR(150) PRIMARY KEY,
        entity_id   VARCHAR(40) NOT NULL,
        realm_id    VARCHAR(150) NOT NULL,
        nickname    VARCHAR(150) NULL,
        role        VARCHAR(150) NOT NULL,
        date_joined TIMESTAMP NULL
    )
    """,
]


def create_chatterloop_schema():
    """Create the tables in the chatterloop test connection.

    Called from `setUp`, so the per-test transaction rolls the DDL back
    afterwards and each test starts from an empty schema of its own.
    """
    with connections["chatterloop"].cursor() as cursor:
        for statement in SCHEMA:
            cursor.execute(statement)
