"""Give the conversations written with a NULL `created_by` an owner.

WHAT LEFT THE NULLS BEHIND
--------------------------
`chatterloop/tasks.py::_mirror_conversation` used to write `bot.created_by`
straight through. That column is SET_NULL, so a bot whose minting account had
been deleted mirrored its exchange with no author - and while
`Conversation.created_by` was still NOT NULL the whole exchange died there,
which is what migration 0014 relaxed the column to get past.

`chatterloop/ownership.py` is the rule that replaced the straight-through
write, and it always resolves to somebody. This migration applies the same
rule to the rows written before it existed; nothing new can add to them.

WHY THE CHAIN IS REBUILT HERE INSTEAD OF IMPORTED
-------------------------------------------------
`ownership.conversation_owner` takes live model classes. A data migration must
see the schema as it was at this point in the graph, or it stops applying from
zero the first time one of those models changes - so the walk is reproduced
below against `apps.get_model` historical models and frozen with the rest of
the migration. The two are expected to read the same; if the live rule changes,
this one deliberately does not.

HOW A ROW IS LINKED BACK TO ITS BOT
-----------------------------------
By the name `_mirror_conversation` wrote: "@<handle>: <chatterloop id>". The
footprint carries only the organization and the chatterloop conversation, so
the name is the only thing that names the bot - and `ChatterloopBot` has a
unique constraint on (organization, handle), so a handle resolves to at most
one bot. Anything that does not parse falls through to the organization's
owner rather than being guessed at.
"""

from django.db import migrations

# `ConnectedAccount.PROVIDER_CHATTERLOOP`, spelled out: a historical model
# carries fields, not class attributes.
PROVIDER_CHATTERLOOP = "chatterloop"

BATCH = 500


def _handle_from_name(name):
    """The bot handle out of a mirrored conversation's name, or None."""
    if not name or not name.startswith("@"):
        return None
    marker = name.find(":")
    if marker < 2:
        return None
    return name[1:marker].strip() or None


def _owner_for_bot(bot, by_entity, live_claims, connectors):
    """`ownership.conversation_owner` steps 1-3, on ids already fetched.

    Returns None when none of them resolve; the caller applies step 4, the
    organization owner.
    """
    if bot.created_by_id:
        return bot.created_by_id

    # The account that connected the page this bot publishes under.
    if bot.connected_account_id:
        account_id = connectors.get(bot.connected_account_id)
        if account_id:
            return account_id

    if bot.owner_entity_id:
        # A person's own mirrored identity first, then a LIVE page claim - a
        # disconnected page's dead row is the authority that was given up.
        return by_entity.get(bot.owner_entity_id) or live_claims.get(
            bot.owner_entity_id
        )

    return None


def backfill_created_by(apps, schema_editor):
    Conversation = apps.get_model("messenger", "Conversation")
    ChatterloopBot = apps.get_model("chatterloop", "ChatterloopBot")
    ConnectedAccount = apps.get_model("core", "ConnectedAccount")
    Organization = apps.get_model("organization", "Organization")
    Account = apps.get_model("user", "Account")

    rows = list(
        Conversation.objects.filter(created_by__isnull=True).only(
            "conversation_id", "organization_id", "name"
        )
    )
    if not rows:
        return

    # Everything below is resolved in a fixed number of queries rather than
    # per row: many conversations share one bot, and every bot in an
    # organization shares its owner.
    organization_ids = {row.organization_id for row in rows}
    handles = {h for h in (_handle_from_name(row.name) for row in rows) if h}

    bots = {}
    if handles:
        bots = {
            (bot.organization_id, bot.handle): bot
            for bot in ChatterloopBot.objects.filter(
                organization_id__in=organization_ids, handle__in=handles
            )
        }

    owner_entity_ids = {
        bot.owner_entity_id for bot in bots.values() if bot.owner_entity_id
    }
    connection_ids = {
        bot.connected_account_id
        for bot in bots.values()
        if bot.connected_account_id
    }

    by_entity = {}
    live_claims = {}
    if owner_entity_ids:
        by_entity = dict(
            Account.objects.filter(entity_id__in=owner_entity_ids).values_list(
                "entity_id", "id"
            )
        )
        live_claims = dict(
            ConnectedAccount.objects.filter(
                provider=PROVIDER_CHATTERLOOP,
                external_id__in=owner_entity_ids,
                is_active=True,
            ).values_list("external_id", "account_id")
        )

    connectors = {}
    if connection_ids:
        connectors = dict(
            ConnectedAccount.objects.filter(id__in=connection_ids).values_list(
                "id", "account_id"
            )
        )

    organization_owners = dict(
        Organization.objects.filter(id__in=organization_ids).values_list(
            "id", "created_by_id"
        )
    )

    repaired = []
    unresolved = 0
    for row in rows:
        owner_id = None

        handle = _handle_from_name(row.name)
        if handle:
            bot = bots.get((row.organization_id, handle))
            if bot is not None:
                owner_id = _owner_for_bot(bot, by_entity, live_claims, connectors)

        # Step 4, and the one that makes the chain terminate:
        # `Organization.created_by` is NOT NULL. It comes back empty only for a
        # conversation whose organization row is gone - `Conversation.organization`
        # is DO_NOTHING, so an orphan is possible - and there is nobody to name
        # for one of those.
        owner_id = owner_id or organization_owners.get(row.organization_id)

        if owner_id is None:
            unresolved += 1
            continue

        row.created_by_id = owner_id
        repaired.append(row)

    for start in range(0, len(repaired), BATCH):
        Conversation.objects.bulk_update(
            repaired[start : start + BATCH], ["created_by"]
        )

    print(f"  backfilled created_by on {len(repaired)} conversation(s)")
    if unresolved:
        # Said out loud rather than left to be discovered by a later NOT NULL
        # migration failing: these rows have no organization to take an owner
        # from and need deciding on, not repairing.
        print(
            f"  {unresolved} conversation(s) left NULL - no organization row to "
            "resolve an owner from"
        )


class Migration(migrations.Migration):

    dependencies = [
        ("messenger", "0014_alter_conversation_created_by"),
        # Read, not altered. Named so `apps.get_model` sees each of them in the
        # state this repair was written against.
        ("chatterloop", "0004_chatterloopbot_provider_credential"),
        ("core", "0003_alter_connectedaccount_id"),
        ("organization", "0005_remove_providercredential_unique_credential_per_org_service_and_more"),
        ("user", "0041_alter_account_id_alter_account_password_and_more"),
    ]

    operations = [
        migrations.RunPython(
            backfill_created_by,
            # Reversing would mean putting the NULLs back, and nothing records
            # which rows they were. The column is still nullable, so unapplying
            # 0014 needs nothing undone here.
            migrations.RunPython.noop,
            elidable=False,
        ),
    ]
