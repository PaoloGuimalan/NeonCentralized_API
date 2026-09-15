"""Record which surface each conversation was started from.

WHY EXISTING ROWS NEED A GUESS AND NEW ONES DO NOT
--------------------------------------------------
From here on the three writers set `origin` themselves - they know for certain
at the moment they write. The rows that already exist have only the footprint
each writer happened to leave, so they are classified once, here, by the shape
of it:

    "chatterloop:<org>:..."     a bot mirror        -> chatterloop
    "<org>:<email>:<id>"        a third-party app   -> external
    NULL, or anything else      Neon's own frontend -> native

Deliberately a migration and not a model property. A prefix test is a heuristic
- `ConversationView` lets a client send whatever footprint it likes, so a
native row is free to look like either integration - and a heuristic belongs in
a one-time repair where it can be reasoned about, not in a read path where it
would quietly re-run on every list forever.

The default on the column is `native`, so this only has to find the two
integration cases; anything it does not match is already right.
"""

from django.db import migrations, models

CHATTERLOOP_PREFIX = "chatterloop:"

BATCH = 500


def classify_existing(apps, schema_editor):
    Conversation = apps.get_model("messenger", "Conversation")

    # Only rows with a footprint can be anything but native, and the column
    # already defaults to native - so this never has to touch the rest.
    rows = list(
        Conversation.objects.exclude(footprint=None)
        .exclude(footprint="")
        .only("conversation_id", "organization_id", "footprint")
    )
    if not rows:
        return

    reclassified = []
    for row in rows:
        if row.footprint.startswith(CHATTERLOOP_PREFIX):
            row.origin = "chatterloop"
        elif row.footprint.startswith(f"{row.organization_id}:"):
            # ExternalChatView namespaces per organization AND per person:
            # "<organization id>:<email>:<external conversation id>". The
            # organization prefix is the part that cannot be coincidence for a
            # footprint stored against that same organization.
            row.origin = "external"
        else:
            continue
        reclassified.append(row)

    for start in range(0, len(reclassified), BATCH):
        Conversation.objects.bulk_update(
            reclassified[start : start + BATCH], ["origin"]
        )

    print(f"  classified {len(reclassified)} conversation(s) as non-native")


class Migration(migrations.Migration):

    dependencies = [
        ("messenger", "0015_backfill_conversation_created_by"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="origin",
            field=models.CharField(
                choices=[
                    ("native", "Neon platform"),
                    ("external", "External app"),
                    ("chatterloop", "Chatterloop bot"),
                ],
                db_index=True,
                default="native",
                help_text="Which surface this conversation was started from.",
                max_length=20,
            ),
        ),
        migrations.RunPython(
            classify_existing,
            # Nothing to undo: unapplying drops the column, which takes every
            # value with it.
            migrations.RunPython.noop,
            elidable=False,
        ),
    ]
