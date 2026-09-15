"""Give `origin` a DATABASE-level default, so the column is really additive.

0016 added it NOT NULL with a Python-side default only. Django uses a `default`
to backfill existing rows and then drops it from the schema, so the column
ended up NOT NULL with no database default - and Django names every field in
the INSERTs it builds, which means an image that predates the field omits it
and violates the constraint.

The database is always migrated before the new image is running, so that window
is not hypothetical: it broke every conversation INSERT - native chat, the
external endpoint and bot mirroring alike - between 0016 applying and the
deploy landing.

`db_default` closes it. Old code inserts without the column and gets "native",
new code says which surface explicitly, and the schema and the image can roll
out in either order.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('messenger', '0016_conversation_origin'),
    ]

    operations = [
        migrations.AlterField(
            model_name='conversation',
            name='origin',
            field=models.CharField(choices=[('native', 'Neon platform'), ('external', 'External app'), ('chatterloop', 'Chatterloop bot')], db_default='native', db_index=True, default='native', help_text='Which surface this conversation was started from.', max_length=20),
        ),
    ]
