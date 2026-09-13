"""Which API key a bot's replies are billed to.

An organization can hold several keys for one provider. A bot either names one
- so its spend is attributable on its own - or names none and uses the
provider's default, which is what every bot did when only one key was allowed.
"""

import uuid
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from chatterloop.models import ChatterloopBot
from chatterloop.provisioning import mint_bot
from chatterloop.tasks import answer_trigger
from chatterloop.testing import create_chatterloop_schema
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource
from llm.models import Agent, Model, Role, Service
from neon.testing import client_for, make_account, make_organization
from neon.utils import crypto
from organization.credentials import CredentialNotConfigured, chat_api_key, default_credential
from organization.models import ProviderCredential


class CredentialTestCase(TestCase):
    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()

        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")
        self.client_acme = client_for(self.alice, self.org)

        self.openai = Service.objects.create(name="OpenAI")
        self.groq = Service.objects.create(name="Groq")
        self.model = Model.objects.create(service=self.openai, model="gpt-4o-mini")

        self.shared = self._credential("Shared", self.openai, "sk-shared", default=True)
        self.dedicated = self._credential("Acme Corp", self.openai, "sk-dedicated")

        role = Role.objects.create(
            organization=self.org, name="S", system_prompt="Be helpful."
        )
        self.agent = Agent.objects.create(
            organization=self.org, name="H", slug="h", role=role
        )

    def tearDown(self):
        crypto._cipher = None

    def _credential(self, name, service, key, default=False, embedding=False):
        credential = ProviderCredential(
            organization=self.org,
            service=service,
            name=name,
            is_default=default,
            is_embedding_default=embedding,
        )
        credential.api_key = key
        credential.save()
        return credential

    def _bot(self, handle="helper", credential=None):
        bot, _ = mint_bot(
            organization=self.org,
            created_by=self.alice,
            name="Helper",
            handle=handle,
            owner_entity_id=self.alice.entity_id,
            agent=self.agent,
            model=self.model,
            provider_credential=credential,
        )
        return bot


class ResolutionTests(CredentialTestCase):

    def test_a_bots_own_key_wins(self):
        self.assertEqual(
            chat_api_key(self.org, self.openai, self.dedicated), "sk-dedicated"
        )

    def test_no_assignment_falls_back_to_the_provider_default(self):
        self.assertEqual(chat_api_key(self.org, self.openai, None), "sk-shared")

    def test_a_single_key_is_the_default_without_being_flagged(self):
        """A single-key organization should not have to make a choice it does
        not yet have."""
        other = make_organization(make_account("bob"), "Beta", "beta")
        credential = ProviderCredential(
            organization=other, service=self.openai, name="Only"
        )
        credential.api_key = "sk-only"
        credential.save()

        self.assertEqual(default_credential(other, self.openai).pk, credential.pk)
        self.assertEqual(chat_api_key(other, self.openai), "sk-only")

    def test_several_keys_with_no_default_is_not_guessed_at(self):
        """Picking "the first row" would mean a bot silently changing which
        account it bills to when somebody adds a key."""
        other = make_organization(make_account("bob"), "Beta", "beta")
        for name in ("A", "B"):
            credential = ProviderCredential(
                organization=other, service=self.openai, name=name
            )
            credential.api_key = f"sk-{name}"
            credential.save()

        self.assertIsNone(default_credential(other, self.openai))
        with self.assertRaises(CredentialNotConfigured):
            chat_api_key(other, self.openai)

    def test_a_key_for_the_wrong_provider_is_ignored(self):
        """Assigned before somebody changed the bot's model. Sending an OpenAI
        key to Groq fails as an authentication error and reads as a revoked
        key rather than a mismatch."""
        groq_key = self._credential("Groq one", self.groq, "gsk-x", default=True)
        self.assertEqual(
            chat_api_key(self.org, self.openai, groq_key), "sk-shared"
        )

    def test_another_organizations_key_is_ignored(self):
        other = make_organization(make_account("bob"), "Beta", "beta")
        theirs = ProviderCredential(
            organization=other, service=self.openai, name="Theirs"
        )
        theirs.api_key = "sk-theirs"
        theirs.save()

        self.assertEqual(chat_api_key(self.org, self.openai, theirs), "sk-shared")

    def test_an_inactive_key_is_ignored(self):
        self.dedicated.is_active = False
        self.dedicated.save(update_fields=["is_active"])
        self.assertEqual(
            chat_api_key(self.org, self.openai, self.dedicated), "sk-shared"
        )


class AnsweringTests(CredentialTestCase):
    """The key actually reaches the model call."""

    def _answer(self, bot):
        captured = {}

        class FakeLLM:
            def stream_chat_completion(self, history, system_prompt, content, tools):
                yield "answer"

        def create(service, api_key, model):
            captured["api_key"] = api_key
            return FakeLLM()

        trigger = Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.MENTION,
            author_entity_id="entity-human",
            conversation_id="conv-1",
            message_id="m1",
            text="hi",
            query="hi",
            dedupe_key=f"msg:{uuid.uuid4().hex}",
        )
        with patch("chatterloop.tasks.LLMFactory") as factory, patch(
            "chatterloop.tasks.send_message"
        ), patch("chatterloop.tasks._retrieve", return_value=[]):
            factory.return_value.create.side_effect = create
            answer_trigger(str(bot.pk), trigger.to_payload())
        return captured.get("api_key")

    def test_a_bot_with_its_own_key_uses_it(self):
        bot = self._bot("dedicated", credential=self.dedicated)
        self.assertEqual(self._answer(bot), "sk-dedicated")

    def test_a_bot_without_one_uses_the_default(self):
        bot = self._bot("shared")
        self.assertEqual(self._answer(bot), "sk-shared")

    def test_two_bots_can_bill_to_different_keys(self):
        """The whole point of the feature."""
        a = self._bot("bot-a", credential=self.dedicated)
        b = self._bot("bot-b")
        self.assertEqual(self._answer(a), "sk-dedicated")
        self.assertEqual(self._answer(b), "sk-shared")


class ApiTests(CredentialTestCase):

    def test_a_bot_can_be_minted_against_a_key(self):
        response = self.client_acme.post(
            reverse("api-chatterloop:bots"),
            {
                "name": "Helper",
                "handle": "helper",
                "agent_uuid": self.agent.uuid,
                "model_uuid": self.model.uuid,
                "credential_id": str(self.dedicated.pk),
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["credential_name"], "Acme Corp")

        bot = ChatterloopBot.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(str(bot.provider_credential_id), str(self.dedicated.pk))

    def test_a_key_can_be_reassigned_later(self):
        bot = self._bot()
        response = self.client_acme.patch(
            reverse("api-chatterloop:bot-detail", args=[str(bot.pk)]),
            {"credential_id": str(self.dedicated.pk)},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        bot.refresh_from_db()
        self.assertEqual(str(bot.provider_credential_id), str(self.dedicated.pk))

    def test_clearing_the_assignment_returns_it_to_the_default(self):
        bot = self._bot(credential=self.dedicated)
        self.client_acme.patch(
            reverse("api-chatterloop:bot-detail", args=[str(bot.pk)]),
            {"credential_id": ""},
            format="json",
        )
        bot.refresh_from_db()
        self.assertIsNone(bot.provider_credential_id)

    def test_another_organizations_key_cannot_be_assigned(self):
        """Scoped like every other related id here - an unrestricted lookup
        would let a bot be pointed at another tenant's API key."""
        other = make_organization(make_account("bob"), "Beta", "beta")
        theirs = ProviderCredential(
            organization=other, service=self.openai, name="Theirs"
        )
        theirs.api_key = "sk-theirs"
        theirs.save()

        response = self.client_acme.post(
            reverse("api-chatterloop:bots"),
            {
                "name": "Helper",
                "handle": "helper",
                "agent_uuid": self.agent.uuid,
                "model_uuid": self.model.uuid,
                "credential_id": str(theirs.pk),
            },
            format="json",
        )
        self.assertEqual(response.status_code, 404)
        self.assertFalse(ChatterloopBot.objects.exists())

    def test_deleting_a_key_names_the_bots_it_drops(self):
        """They fall back to the default, which is a billing change nobody
        asked for - so it is reported rather than discovered from an invoice."""
        self._bot("pinned", credential=self.dedicated)

        response = self.client_acme.delete(
            reverse("api-organization:credential-detail", args=[str(self.dedicated.pk)])
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("@pinned", response.data["message"])

    def test_a_deleted_key_leaves_its_bots_running_on_the_default(self):
        bot = self._bot("pinned", credential=self.dedicated)
        self.client_acme.delete(
            reverse("api-organization:credential-detail", args=[str(self.dedicated.pk)])
        )
        bot.refresh_from_db()
        self.assertIsNone(bot.provider_credential_id)
        self.assertEqual(bot.status, ChatterloopBot.STATUS_ACTIVE)

    def test_the_listing_reports_how_many_bots_use_each_key(self):
        self._bot("pinned", credential=self.dedicated)
        response = self.client_acme.get(reverse("api-organization:credentials"))
        rows = {row["label"]: row for row in response.data["data"]}
        self.assertEqual(rows["Acme Corp"]["bot_count"], 1)
        self.assertEqual(rows["Shared"]["bot_count"], 0)
