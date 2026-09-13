"""Make Tool and Role belong to an organization, and scope uniqueness to it.

WHY THIS IS HAND-WRITTEN
------------------------
`organization` is non-nullable, and existing Tool/Role rows have no value for
it, so `makemigrations` stops and asks what to put there. That question is the
whole risk in this migration: answering it wrongly moves one tenant's tools
into another tenant's organization, and the unique constraints added
immediately afterwards make that awkward to unpick.

So the column is added nullable, filled in deliberately, and only then made
non-nullable.

HOW THE ORGANIZATION IS CHOSEN
------------------------------
In order:

  1. `NEON_DEFAULT_ORGANIZATION_ID`, if set. Explicit beats inferred.
  2. The only organization, if there is exactly one. Unambiguous by
     definition - there is no other answer it could be.
  3. Otherwise, FAIL, naming the candidates.

The third case is the point. With several organizations there is no way to
know which one owns a global Tool row, and a migration that guesses would
hand one customer's tooling - including its stored credentials - to another.
Refusing costs an operator one environment variable; guessing costs a data
leak nobody would notice until much later.
"""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def _resolve_default_organization(apps, schema_editor):
    Organization = apps.get_model("organization", "Organization")
    Tool = apps.get_model("llm", "Tool")
    Role = apps.get_model("llm", "Role")

    # Nothing to place: a fresh database needs no default at all, and must not
    # be blocked by one being unset.
    if not Tool.objects.exists() and not Role.objects.exists():
        return None

    configured = getattr(settings, "NEON_DEFAULT_ORGANIZATION_ID", None)
    if configured:
        organization = Organization.objects.filter(pk=configured).first()
        if organization is None:
            raise RuntimeError(
                f"NEON_DEFAULT_ORGANIZATION_ID is set to {configured!r} but no "
                "organization with that id exists."
            )
        return organization

    organizations = list(Organization.objects.all()[:2])
    if len(organizations) == 1:
        return organizations[0]

    if not organizations:
        raise RuntimeError(
            "There are Tool/Role rows to place but no organizations exist. "
            "Create the organization that owns them, then set "
            "NEON_DEFAULT_ORGANIZATION_ID and re-run."
        )

    names = ", ".join(
        f"{o.pk} ({o.name})" for o in Organization.objects.all().order_by("created_at")
    )
    raise RuntimeError(
        "More than one organization exists, so the organization that owns the "
        "existing Tool/Role rows cannot be inferred. Set "
        "NEON_DEFAULT_ORGANIZATION_ID to one of: " + names
    )


def backfill(apps, schema_editor):
    organization = _resolve_default_organization(apps, schema_editor)
    if organization is None:
        return

    Tool = apps.get_model("llm", "Tool")
    Role = apps.get_model("llm", "Role")

    Tool.objects.filter(organization__isnull=True).update(organization=organization)
    Role.objects.filter(organization__isnull=True).update(organization=organization)


def unbackfill(apps, schema_editor):
    """Reversing only clears the column.

    It cannot restore the pre-migration state any more meaningfully than that:
    the rows had no organization before, and which one they were given is
    exactly the information being discarded.
    """
    Tool = apps.get_model("llm", "Tool")
    Role = apps.get_model("llm", "Role")
    Tool.objects.update(organization=None)
    Role.objects.update(organization=None)


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0008_model_service"),
        ("organization", "0002_organization_llm_api_key"),
    ]

    operations = [
        # 1. The global unique constraints come off first: they are what stops
        #    two organizations owning a tool of the same name, and they have to
        #    be gone before the per-organization ones can mean anything.
        migrations.AlterField(
            model_name="tool",
            name="name",
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name="role",
            name="name",
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name="agent",
            name="slug",
            field=models.SlugField(),
        ),
        # 2. Nullable, so existing rows survive the schema change.
        migrations.AddField(
            model_name="tool",
            name="organization",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="tools",
                to="organization.organization",
            ),
        ),
        migrations.AddField(
            model_name="role",
            name="organization",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="roles",
                to="organization.organization",
            ),
        ),
        # 3. Fill it in, or refuse to continue.
        migrations.RunPython(backfill, unbackfill),
        # 4. Now it can be required.
        migrations.AlterField(
            model_name="tool",
            name="organization",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="tools",
                to="organization.organization",
            ),
        ),
        migrations.AlterField(
            model_name="role",
            name="organization",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="roles",
                to="organization.organization",
            ),
        ),
        # 5. Uniqueness, now scoped to the tenant.
        migrations.AddConstraint(
            model_name="tool",
            constraint=models.UniqueConstraint(
                fields=("organization", "name"), name="unique_tool_name_per_org"
            ),
        ),
        migrations.AddConstraint(
            model_name="role",
            constraint=models.UniqueConstraint(
                fields=("organization", "name"), name="unique_role_name_per_org"
            ),
        ),
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.UniqueConstraint(
                fields=("organization", "slug"), name="unique_agent_slug_per_org"
            ),
        ),
    ]
