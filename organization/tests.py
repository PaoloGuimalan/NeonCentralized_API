"""Tests for provider credentials and the encryption behind them."""

from cryptography.fernet import Fernet
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.test import TestCase, override_settings

from llm.models import Service
from neon.utils import crypto
from neon.utils.crypto import DecryptionFailed, decrypt, encrypt
from organization.models import Organization, ProviderCredential
from user.models import Account


def reset_cipher():
    """The cipher is cached per process, so a settings override needs it cleared."""
    crypto._cipher = None


class CryptoTests(TestCase):

    def setUp(self):
        reset_cipher()

    def tearDown(self):
        reset_cipher()

    def test_round_trip(self):
        self.assertEqual(decrypt(encrypt("sk-secret-value")), "sk-secret-value")

    def test_ciphertext_does_not_contain_the_plaintext(self):
        """The point of the exercise: a database dump must not read as keys."""
        self.assertNotIn("sk-secret-value", encrypt("sk-secret-value"))

    def test_empty_values_round_trip_as_empty(self):
        self.assertEqual(encrypt(""), "")
        self.assertEqual(encrypt(None), "")
        self.assertEqual(decrypt(""), "")

    def test_encryption_is_not_deterministic(self):
        """Fernet carries an IV, so the same key twice must not look the same.

        Otherwise a dump would reveal which organizations share a credential.
        """
        self.assertNotEqual(encrypt("same-key"), encrypt("same-key"))

    def test_a_different_key_cannot_decrypt(self):
        ciphertext = encrypt("sk-secret-value")
        reset_cipher()
        with override_settings(
            TOKEN_ENCRYPTION_KEY=Fernet.generate_key().decode(),
            TOKEN_ENCRYPTION_KEY_FALLBACKS=[],
        ):
            with self.assertRaises(DecryptionFailed):
                decrypt(ciphertext)

    def test_rotation_via_fallbacks(self):
        """A rotated key must not orphan everything already stored."""
        old_key = Fernet.generate_key().decode()
        reset_cipher()
        with override_settings(TOKEN_ENCRYPTION_KEY=old_key):
            ciphertext = encrypt("sk-secret-value")

        reset_cipher()
        with override_settings(
            TOKEN_ENCRYPTION_KEY=Fernet.generate_key().decode(),
            TOKEN_ENCRYPTION_KEY_FALLBACKS=[old_key],
        ):
            self.assertEqual(decrypt(ciphertext), "sk-secret-value")

    def test_missing_key_is_a_configuration_error(self):
        reset_cipher()
        with override_settings(TOKEN_ENCRYPTION_KEY=None):
            with self.assertRaises(ImproperlyConfigured):
                encrypt("anything")


class ProviderCredentialTests(TestCase):

    def setUp(self):
        reset_cipher()
        self.account = Account.objects.create(
            username="owner",
            first_name="Owner",
            last_name="Person",
            email="owner@example.com",
        )
        self.org = Organization.objects.create(
            name="Acme", slug="acme", created_by=self.account
        )
        self.openai = Service.objects.create(name="OpenAI")
        self.groq = Service.objects.create(name="Groq")

    def test_api_key_is_stored_encrypted_and_read_back(self):
        credential = ProviderCredential(organization=self.org, service=self.openai)
        credential.api_key = "sk-live-123"
        credential.save()

        stored = ProviderCredential.objects.get(pk=credential.pk)
        self.assertEqual(stored.api_key, "sk-live-123")
        self.assertNotIn("sk-live-123", stored.api_key_encrypted)

    def test_several_credentials_per_service_are_allowed(self):
        """The rule this used to enforce is deliberately gone.

        One key per provider meant every bot in an organization billed to the
        same account. Several - assigned per bot or shared - is what lets a
        customer-facing bot's spend be separated from everything else's.
        """
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, name="Shared"
        )
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, name="Billing: Acme Corp"
        )
        self.assertEqual(
            ProviderCredential.objects.filter(
                organization=self.org, service=self.openai
            ).count(),
            2,
        )

    def test_two_credentials_cannot_share_a_name(self):
        """A name is how a person tells them apart, so it has to be unique."""
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, name="Shared"
        )
        with self.assertRaises(IntegrityError):
            ProviderCredential.objects.create(
                organization=self.org, service=self.groq, name="Shared"
            )

    def test_only_one_default_per_provider(self):
        """Which key an unassigned bot bills to must never be ambiguous."""
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, name="A", is_default=True
        )
        with self.assertRaises(IntegrityError):
            ProviderCredential.objects.create(
                organization=self.org, service=self.openai, name="B", is_default=True
            )

    def test_two_providers_can_each_have_a_default(self):
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, name="A", is_default=True
        )
        ProviderCredential.objects.create(
            organization=self.org, service=self.groq, name="B", is_default=True
        )
        self.assertEqual(
            ProviderCredential.objects.filter(is_default=True).count(), 2
        )

    def test_an_organization_can_hold_several_providers(self):
        """The whole reason this model exists.

        A Groq chat key and an OpenAI embedding key at once is what
        Organization.llm_api_key could not express, and why RAG did nothing
        for a Groq-only organization.
        """
        ProviderCredential.objects.create(organization=self.org, service=self.groq)
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, is_embedding_default=True
        )
        self.assertEqual(self.org.provider_credentials.count(), 2)

    def test_only_one_embedding_default_per_organization(self):
        ProviderCredential.objects.create(
            organization=self.org, service=self.openai, is_embedding_default=True
        )
        with self.assertRaises(IntegrityError):
            ProviderCredential.objects.create(
                organization=self.org, service=self.groq, is_embedding_default=True
            )
