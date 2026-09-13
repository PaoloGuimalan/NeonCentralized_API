"""Settings for running tests and checking migrations without infrastructure.

`python manage.py test --settings=neon.settings_test` needs no Postgres, no
Redis, no Pinecone and no chatterloop. That matters for the same reason
`go test ./...` needs nothing in developer_service: a test suite that requires
a live environment is one nobody runs before pushing.

Use it to prove the migration graph applies from zero, which is the one thing
`makemigrations --check` cannot tell you:

    python manage.py migrate --settings=neon.settings_test
"""

from .settings import *  # noqa: F401,F403

# SQLite in memory. The chatterloop alias is configured but never migrated -
# the router refuses it - so this entry exists only so Django can resolve the
# connection by name.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    },
    "chatterloop": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    },
}

# A fixed, throwaway Fernet key. Deliberately committed and deliberately not
# the real one: tests need encryption to round-trip, not to be secret.
TOKEN_ENCRYPTION_KEY = "cGxhY2Vob2xkZXJfdGVzdF9rZXlfMzJfYnl0ZXNfeHg="
TOKEN_ENCRYPTION_KEY_FALLBACKS = []

# Run Celery work inline rather than needing a broker.
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Local-memory cache instead of Redis.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

CHATTERLOOP_API_BASE_URL = "https://chatterloop.test"

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
