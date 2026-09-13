"""Which database each model lives in.

THE ONE RULE THAT MATTERS
-------------------------
`allow_migrate` returns False for the chatterloop alias unconditionally, for
every app and every model. Neon must never run a migration against another
service's database - not even one Django believes is a no-op, because
`makemigrations` comparing an unmanaged model against a live schema and
deciding something needs changing is exactly the failure this prevents.

That is enforced here rather than relying on `managed = False` alone, so a
future model added to external_models.py without that flag still cannot write
schema.

WHAT THIS IS NOT
----------------
A security boundary. Neon holds full credentials for chatterloop's database,
so this router constrains *Neon's own code paths*, not what Neon is capable of.
It stops an accidental write, not a determined or buggy one. See the plan's
Risks section.
"""

CHATTERLOOP_ALIAS = "chatterloop"
CHATTERLOOP_APP_LABEL = "chatterloop"

# Models in the chatterloop app that are Neon's OWN rows and therefore belong
# in Neon's database, not chatterloop's. Everything else in that app is a
# projection of a chatterloop table.
#
# Kept as names rather than imported classes: a router is loaded very early in
# Django's startup, before the app registry is populated, so importing models
# here would be a circular import.
NEON_OWNED_MODELS = {
    "chatterloopbot",
    "chatterlooptoken",
}


def _is_external(model):
    """Whether this model is a projection of a chatterloop-owned table."""
    if model._meta.app_label != CHATTERLOOP_APP_LABEL:
        return False
    return model._meta.model_name not in NEON_OWNED_MODELS


class ChatterloopRouter:

    def db_for_read(self, model, **hints):
        return CHATTERLOOP_ALIAS if _is_external(model) else None

    def db_for_write(self, model, **hints):
        return CHATTERLOOP_ALIAS if _is_external(model) else None

    def allow_relation(self, obj1, obj2, **hints):
        """Permit relations only within one database.

        Returning None (undecided) for the same-database case lets Django fall
        back to its own check. Two objects on different databases are refused
        outright rather than left to produce a confusing failure deeper in.
        """
        db1 = CHATTERLOOP_ALIAS if _is_external(type(obj1)) else "default"
        db2 = CHATTERLOOP_ALIAS if _is_external(type(obj2)) else "default"
        if db1 == db2:
            return True
        return False

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """Never migrate the chatterloop database. Never put Neon's tables in it."""
        if db == CHATTERLOOP_ALIAS:
            return False
        # A chatterloop projection must not be created in Neon's database
        # either - it would produce an empty local table shadowing the real one.
        if app_label == CHATTERLOOP_APP_LABEL and model_name is not None:
            return model_name in NEON_OWNED_MODELS
        return None
