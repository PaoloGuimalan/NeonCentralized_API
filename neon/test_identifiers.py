"""Every model must equal itself.

THE BUG THESE PIN DOWN
----------------------
`id = models.CharField(default=uuid.uuid4)` hands a freshly created instance a
`UUID` object while the database returns a `str`. Django's `Model.__eq__`
compares primary keys directly, so an object and the same row fetched back
compared UNEQUAL - silently, with nothing raised and a comparison that looks
obviously correct taking the wrong branch.

Swept across every model rather than spot-checked, because the defect was
copy-pasted: it appeared in four apps, and a new model written from the same
template would reintroduce it without anybody noticing until an `assertEqual`
failed somewhere unrelated.
"""

import uuid

from django.apps import apps
from django.db import models
from django.test import TestCase
from django.utils.timezone import now

from chatterloop.models import ChatterloopBot, ChatterloopToken
from chatterloop.testing import create_chatterloop_schema
from core.models import ConnectedAccount
from llm.models import Agent, KnowledgeDocument, Model, Role, Service, Tool
from neon.testing import make_account, make_organization
from neon.utils import crypto
from neon.utils.identifiers import new_id
from organization.models import Member, Organization, ProviderCredential
from user.models import Account, Token

# Apps whose models Neon owns. `chatterloop`'s external projections are
# excluded per-model below: they mirror another service's tables, where the id
# is supplied rather than defaulted.
NEON_APPS = ("user", "organization", "llm", "core", "messenger", "chatterloop")


class NewIdTests(TestCase):

    def test_it_returns_a_string(self):
        value = new_id()
        self.assertIsInstance(value, str)

    def test_it_matches_the_format_already_stored(self):
        """Existing rows hold the 36-character hyphenated form, so new ones
        must too - otherwise the fix would need a data migration."""
        value = new_id()
        self.assertEqual(len(value), 36)
        self.assertEqual(value.count("-"), 4)
        # Round-trips through uuid, so it really is one.
        self.assertEqual(str(uuid.UUID(value)), value)

    def test_it_is_not_predictable(self):
        self.assertEqual(len({new_id() for _ in range(200)}), 200)


class NoModelDefaultsToAUuidObject(TestCase):
    """The sweep. A CharField defaulting to `uuid.uuid4` is the bug itself."""

    def test_no_charfield_defaults_to_the_uuid4_function(self):
        offenders = []
        for model in apps.get_models():
            if model._meta.app_label not in NEON_APPS:
                continue
            if not model._meta.managed:
                # A projection of another service's table - the id comes from
                # there, and this app must not invent one.
                continue
            for field in model._meta.get_fields():
                if not isinstance(field, models.CharField):
                    continue
                if field.default is uuid.uuid4:
                    offenders.append(f"{model.__name__}.{field.name}")

        self.assertEqual(
            offenders,
            [],
            "These CharFields default to the uuid4 FUNCTION, so a freshly "
            "created instance holds a UUID object while a fetched one holds a "
            "str - and the two compare unequal. Use "
            "neon.utils.identifiers.new_id instead: " + ", ".join(offenders),
        )

    def test_every_uuid_defaulting_charfield_produces_a_string(self):
        """Catches a future helper that forgets the `str()`."""
        checked = 0
        for model in apps.get_models():
            if model._meta.app_label not in NEON_APPS or not model._meta.managed:
                continue
            for field in model._meta.get_fields():
                if not isinstance(field, models.CharField):
                    continue
                if field.default in (models.NOT_PROVIDED, None) or not callable(
                    field.default
                ):
                    continue
                value = field.default()
                checked += 1
                self.assertIsInstance(
                    value,
                    str,
                    f"{model.__name__}.{field.name}'s default returned "
                    f"{type(value).__name__}, not str",
                )
        # Guards against the sweep silently checking nothing.
        self.assertGreater(checked, 10)


class InstancesEqualThemselves(TestCase):
    """The behaviour the bug actually broke, per affected model."""

    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()
        self.account = make_account("alice", entity_id="entity-alice")
        self.organization = make_organization(self.account, "Acme", "acme")

    def tearDown(self):
        crypto._cipher = None

    def assert_round_trips(self, created):
        model = type(created)
        fetched = model.objects.get(pk=created.pk)

        self.assertIsInstance(
            created.pk, str, f"{model.__name__}.pk is a {type(created.pk).__name__}"
        )
        self.assertEqual(
            created,
            fetched,
            f"{model.__name__} does not equal itself after a round trip",
        )
        # The consequences that are easy to miss.
        self.assertEqual(hash(created), hash(fetched))
        self.assertIn(created, [fetched])
        self.assertEqual(len({created, fetched}), 1)

    def test_account(self):
        self.assert_round_trips(self.account)

    def test_organization(self):
        self.assert_round_trips(self.organization)

    def test_member(self):
        self.assert_round_trips(
            Member.objects.get(organization=self.organization, account=self.account)
        )

    def test_token(self):
        self.assert_round_trips(Token.objects.create(account=self.account))

    def test_connected_account(self):
        self.assert_round_trips(
            ConnectedAccount.objects.create(
                account=self.account,
                provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
                external_id="entity-page",
                external_type=ConnectedAccount.TYPE_REALM,
                external_name="Acme Page",
            )
        )

    def test_provider_credential(self):
        service = Service.objects.create(name="OpenAI")
        credential = ProviderCredential(
            organization=self.organization, service=service
        )
        credential.api_key = "sk-test"
        credential.save()
        self.assert_round_trips(credential)

    def test_knowledge_document(self):
        self.assert_round_trips(
            KnowledgeDocument.objects.create(
                organization=self.organization, title="Policy", content="text"
            )
        )

    def test_chatterloop_bot(self):
        self.assert_round_trips(
            ChatterloopBot.objects.create(
                organization=self.organization,
                entity_id="entity-bot",
                bot_id="bot-1",
                name="Helper",
                handle="helper",
                owner_entity_id="entity-alice",
            )
        )

    def test_chatterloop_token(self):
        bot = ChatterloopBot.objects.create(
            organization=self.organization,
            entity_id="entity-bot",
            bot_id="bot-1",
            name="Helper",
            handle="helper",
            owner_entity_id="entity-alice",
        )
        credential = ChatterloopToken(
            bot=bot, token_id="t1", prefix="abc", name="Helper"
        )
        credential.set_secret("secret")
        credential.save()
        self.assert_round_trips(credential)


class NonPrimaryKeyFieldsTests(TestCase):
    """The same defect on columns that are not primary keys."""

    def setUp(self):
        self.account = make_account("alice")
        self.organization = make_organization(self.account, "Acme", "acme")

    def test_an_organizations_access_key_and_pin_are_strings(self):
        """Compared against values arriving as strings, so a UUID object here
        would never match one."""
        self.assertIsInstance(self.organization.access_key, str)
        self.assertIsInstance(self.organization.pin, str)

        fetched = Organization.objects.get(pk=self.organization.pk)
        self.assertEqual(self.organization.access_key, fetched.access_key)
        self.assertEqual(self.organization.pin, fetched.pin)

    def test_an_accounts_placeholder_password_is_a_string(self):
        self.assertIsInstance(self.account.password, str)

    def test_an_agents_external_uuid_is_a_string(self):
        """It is what a caller passes in a URL, so it is compared against
        strings constantly."""
        agent = Agent.objects.create(
            organization=self.organization, name="Helper", slug="helper"
        )
        self.assertIsInstance(agent.uuid, str)
        self.assertEqual(
            agent.uuid, Agent.objects.get(pk=agent.pk).uuid
        )

    def test_a_services_and_models_uuid_are_strings(self):
        service = Service.objects.create(name="OpenAI")
        model = Model.objects.create(service=service, model="gpt-4o-mini")
        self.assertIsInstance(service.uuid, str)
        self.assertIsInstance(model.uuid, str)


class RealisticComparisonTests(TestCase):
    """What the bug looked like in practice: a tenancy check taking the wrong
    branch because two references to one organization compared unequal."""

    def setUp(self):
        self.account = make_account("alice")
        self.organization = make_organization(self.account, "Acme", "acme")

    def test_a_tenancy_check_against_a_created_object_holds(self):
        from messenger.models import Conversation

        conversation = Conversation.objects.create(
            organization=self.organization, name="Chat", created_by=self.account
        )
        fetched = Conversation.objects.select_related("organization").get(
            pk=conversation.pk
        )

        # `if conversation.organization == request_organization` - the shape of
        # comparison that silently answered False.
        self.assertEqual(fetched.organization, self.organization)
        self.assertEqual(fetched.organization_id, self.organization.pk)

    def test_a_member_lookup_by_a_created_objects_pk_works(self):
        member = Member.objects.create(
            account=self.account,
            organization=self.organization,
            added_by=self.account,
            date_joined=now(),
        )
        self.assertTrue(Member.objects.filter(pk=member.pk).exists())
        self.assertEqual(Member.objects.get(pk=member.pk), member)

    def test_a_role_and_tool_compare_across_a_relation(self):
        tool = Tool.objects.create(organization=self.organization, name="search")
        role = Role.objects.create(
            organization=self.organization, name="Support", system_prompt="x"
        )
        role.tools.add(tool)

        self.assertEqual(role.organization, self.organization)
        self.assertIn(tool, role.tools.all())
