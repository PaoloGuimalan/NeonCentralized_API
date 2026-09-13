"""Tests for the organization-scoping of Tool, Role and Agent.

The backfill resolver gets the most attention here because it is the one piece
of this change that can silently do damage: putting one tenant's tools into
another tenant's organization is not visible from the outside, and the unique
constraints added straight afterwards make it awkward to unpick.
"""

from importlib import import_module

from django.apps import apps as django_apps
from django.db import IntegrityError
from django.test import TestCase, override_settings

from llm.models import Agent, Role, Tool
from organization.models import Organization
from user.models import Account

# Imported by path because the module name starts with a digit, so it cannot
# be reached with a normal import statement.
backfill_migration = import_module("llm.migrations.0009_org_scoped_tools_and_roles")
resolve = backfill_migration._resolve_default_organization


def make_account(email="owner@example.com", username="owner"):
    return Account.objects.create(
        username=username,
        first_name="Owner",
        last_name="Person",
        email=email,
    )


def make_org(account, name, slug):
    return Organization.objects.create(name=name, slug=slug, created_by=account)


class BackfillResolverTests(TestCase):
    """Which organization inherits pre-existing global Tool/Role rows."""

    def setUp(self):
        self.account = make_account()

    def test_no_rows_to_place_needs_no_organization(self):
        """A fresh database must not be blocked by an unset default.

        There is nothing to place, so refusing would stop a brand-new
        deployment migrating for no reason at all.
        """
        make_org(self.account, "A", "a")
        make_org(self.account, "B", "b")
        self.assertIsNone(resolve(django_apps, None))

    def test_single_organization_is_unambiguous(self):
        org = make_org(self.account, "Only", "only")
        Tool.objects.create(organization=org, name="search")
        self.assertEqual(resolve(django_apps, None), org)

    @override_settings(NEON_DEFAULT_ORGANIZATION_ID=None)
    def test_multiple_organizations_refuses_to_guess(self):
        """The case this whole design exists for."""
        org_a = make_org(self.account, "Acme", "acme")
        make_org(self.account, "Other", "other")
        Tool.objects.create(organization=org_a, name="search")

        with self.assertRaises(RuntimeError) as caught:
            resolve(django_apps, None)

        message = str(caught.exception)
        # The error has to name the candidates, or an operator cannot act on it.
        self.assertIn("NEON_DEFAULT_ORGANIZATION_ID", message)
        self.assertIn("Acme", message)
        self.assertIn("Other", message)

    def test_configured_organization_wins(self):
        make_org(self.account, "Acme", "acme")
        chosen = make_org(self.account, "Chosen", "chosen")
        Tool.objects.create(organization=chosen, name="search")

        with override_settings(NEON_DEFAULT_ORGANIZATION_ID=str(chosen.pk)):
            self.assertEqual(resolve(django_apps, None), chosen)

    def test_configured_organization_that_does_not_exist_is_an_error(self):
        org = make_org(self.account, "Acme", "acme")
        Tool.objects.create(organization=org, name="search")

        with override_settings(NEON_DEFAULT_ORGANIZATION_ID="no-such-org"):
            with self.assertRaises(RuntimeError) as caught:
                resolve(django_apps, None)
        self.assertIn("no-such-org", str(caught.exception))


class PerOrganizationUniquenessTests(TestCase):
    """Two organizations must be able to own the same names."""

    def setUp(self):
        self.account = make_account()
        self.org_a = make_org(self.account, "Acme", "acme")
        self.org_b = make_org(self.account, "Beta", "beta")

    def test_two_organizations_can_each_have_a_tool_called_search(self):
        Tool.objects.create(organization=self.org_a, name="search")
        Tool.objects.create(organization=self.org_b, name="search")
        self.assertEqual(Tool.objects.filter(name="search").count(), 2)

    def test_one_organization_cannot_have_two_tools_called_search(self):
        Tool.objects.create(organization=self.org_a, name="search")
        with self.assertRaises(IntegrityError):
            Tool.objects.create(organization=self.org_a, name="search")

    def test_two_organizations_can_each_have_a_role_called_support(self):
        Role.objects.create(organization=self.org_a, name="support", system_prompt="x")
        Role.objects.create(organization=self.org_b, name="support", system_prompt="x")
        self.assertEqual(Role.objects.filter(name="support").count(), 2)

    def test_two_organizations_can_each_have_an_agent_slug(self):
        Agent.objects.create(organization=self.org_a, name="Helper", slug="helper")
        Agent.objects.create(organization=self.org_b, name="Helper", slug="helper")
        self.assertEqual(Agent.objects.filter(slug="helper").count(), 2)

    def test_one_organization_cannot_reuse_an_agent_slug(self):
        Agent.objects.create(organization=self.org_a, name="Helper", slug="helper")
        with self.assertRaises(IntegrityError):
            Agent.objects.create(organization=self.org_a, name="Other", slug="helper")
