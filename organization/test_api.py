"""Tenant resolution and the organization API.

Every test here runs with TWO organizations in the database. A scoping test
with only one tenant present cannot fail, so the second one is part of the
fixture rather than an extra case.
"""

from django.test import TestCase
from django.urls import reverse

from llm.models import Service
from neon.testing import add_member, client_for, make_account, make_organization
from neon.utils import crypto
from organization.credentials import CredentialNotConfigured, embedding_api_key
from organization.models import Member, Organization, ProviderCredential
from organization.tenancy import OrganizationResolutionError, organization_for


def reset_cipher():
    crypto._cipher = None


class TenancyTests(TestCase):

    def setUp(self):
        self.alice = make_account("alice")
        self.bob = make_account("bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")

    def test_no_membership_is_refused(self):
        nobody = make_account("nobody")
        with self.assertRaises(OrganizationResolutionError) as caught:
            organization_for(nobody)
        self.assertEqual(caught.exception.status_code, 403)

    def test_sole_membership_resolves_without_being_named(self):
        """Every caller that predates the X-Organization header relies on this."""
        self.assertEqual(organization_for(self.alice), self.acme)

    def test_ambiguous_membership_names_the_candidates(self):
        add_member(self.beta, self.alice, self.bob)
        with self.assertRaises(OrganizationResolutionError) as caught:
            organization_for(self.alice)
        self.assertEqual(caught.exception.status_code, 409)
        # An error that says neither which organizations nor how to choose is
        # one the caller cannot act on.
        self.assertIn("X-Organization", caught.exception.message)
        self.assertIn("Acme", caught.exception.message)
        self.assertIn("Beta", caught.exception.message)

    def test_naming_one_resolves_the_ambiguity(self):
        add_member(self.beta, self.alice, self.bob)
        chosen = organization_for(self.alice, str(self.beta.pk))
        self.assertEqual(chosen, self.beta)

    def test_naming_an_organization_you_are_not_in_is_forbidden_not_missing(self):
        """403, not 404, on purpose.

        A 404 would answer "is this a real organization id?" for anyone who
        cares to ask. 403 gives an id that exists and one that does not exactly
        the same response.
        """
        with self.assertRaises(OrganizationResolutionError) as caught:
            organization_for(self.alice, str(self.beta.pk))
        self.assertEqual(caught.exception.status_code, 403)

        with self.assertRaises(OrganizationResolutionError) as invented:
            organization_for(self.alice, "does-not-exist")
        self.assertEqual(invented.exception.status_code, 403)


class OrganizationAPITests(TestCase):

    def setUp(self):
        self.alice = make_account("alice")
        self.bob = make_account("bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")

    def test_creating_an_organization_makes_the_creator_a_member(self):
        """Otherwise the new organization is unreachable through every other
        endpoint - including the one that would add somebody to it."""
        newcomer = make_account("newcomer")

        response = client_for(newcomer).post(
            reverse("api-organization:organizations"),
            {"name": "Third Co"},
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        organization = Organization.objects.get(name="Third Co")
        self.assertEqual(organization.slug, "third-co")
        self.assertTrue(
            Member.objects.filter(organization=organization, account=newcomer).exists()
        )

    def test_slug_collisions_are_resolved_rather_than_rejected(self):
        response = client_for(make_account("newcomer")).post(
            reverse("api-organization:organizations"), {"name": "Acme"}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["slug"], "acme-2")

    def test_listing_shows_only_your_own_organizations(self):
        response = client_for(self.alice).get(
            reverse("api-organization:organizations")
        )
        self.assertEqual([row["name"] for row in response.data["data"]], ["Acme"])

    def test_a_member_who_is_not_the_owner_cannot_rename_the_organization(self):
        carol = make_account("carol")
        add_member(self.acme, carol, self.alice)

        response = client_for(carol, self.acme).patch(
            reverse("api-organization:organization-current"),
            {"name": "Renamed"},
            format="json",
        )

        self.assertEqual(response.status_code, 403)
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.name, "Acme")

    def test_the_owner_can_rename_the_organization(self):
        response = client_for(self.alice, self.acme).patch(
            reverse("api-organization:organization-current"),
            {"name": "Acme Inc"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.name, "Acme Inc")

    def test_secrets_are_not_in_the_organization_representation(self):
        """`access_key` and `pin` authenticate the organization; `llm_api_key`
        is a provider credential. None of them belong in a listing every
        member can read."""
        response = client_for(self.alice, self.acme).get(
            reverse("api-organization:organization-current")
        )
        for field in ("access_key", "pin", "llm_api_key"):
            self.assertNotIn(field, response.data["data"])

    def test_members_are_scoped_to_the_organization(self):
        carol = make_account("carol")
        add_member(self.beta, carol, self.bob)

        response = client_for(self.alice, self.acme).get(
            reverse("api-organization:members")
        )
        self.assertEqual(
            [row["username"] for row in response.data["data"]], ["alice"]
        )

    def test_adding_an_unknown_email_explains_why_rather_than_inviting(self):
        """Neon cannot create accounts - chatterloop owns identity - so there
        is nobody here to invite who has not signed in at least once."""
        response = client_for(self.alice, self.acme).post(
            reverse("api-organization:members"),
            {"email": "stranger@example.com"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("sign in to Neon", str(response.data))

    def test_adding_an_existing_account_works(self):
        carol = make_account("carol", email="carol@example.com")
        response = client_for(self.alice, self.acme).post(
            reverse("api-organization:members"),
            {"email": "carol@example.com"},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(
            Member.objects.filter(organization=self.acme, account=carol).exists()
        )

    def test_the_owner_cannot_be_removed(self):
        """An organization whose owner left can never be administered again:
        `Organization.created_by` is the only thing expressing ownership."""
        member = Member.objects.get(organization=self.acme, account=self.alice)
        response = client_for(self.alice, self.acme).delete(
            reverse("api-organization:member-detail", args=[str(member.pk)])
        )
        self.assertEqual(response.status_code, 409)
        self.assertTrue(Member.objects.filter(pk=member.pk).exists())

    def test_a_member_can_remove_themselves(self):
        carol = make_account("carol")
        member = add_member(self.acme, carol, self.alice)

        response = client_for(carol, self.acme).delete(
            reverse("api-organization:member-detail", args=[str(member.pk)])
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Member.objects.filter(pk=member.pk).exists())

    def test_a_member_cannot_remove_someone_else(self):
        carol = make_account("carol")
        dave = make_account("dave")
        add_member(self.acme, carol, self.alice)
        daves_row = add_member(self.acme, dave, self.alice)

        response = client_for(carol, self.acme).delete(
            reverse("api-organization:member-detail", args=[str(daves_row.pk)])
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(Member.objects.filter(pk=daves_row.pk).exists())

    def test_another_organizations_member_is_not_found(self):
        """Not 403: inside a scoped view the row is simply not in scope."""
        bobs_row = Member.objects.get(organization=self.beta, account=self.bob)
        response = client_for(self.alice, self.acme).delete(
            reverse("api-organization:member-detail", args=[str(bobs_row.pk)])
        )
        self.assertEqual(response.status_code, 404)


class ProviderCredentialAPITests(TestCase):

    def setUp(self):
        reset_cipher()
        self.alice = make_account("alice")
        self.bob = make_account("bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")
        self.openai = Service.objects.create(name="OpenAI")
        self.groq = Service.objects.create(name="Groq")

    def tearDown(self):
        reset_cipher()

    def _create(self, client, **payload):
        return client.post(
            reverse("api-organization:credentials"), payload, format="json"
        )

    def test_the_api_key_never_comes_back(self):
        """The whole point of write_only.

        A credential readable through the API is one that any XSS, any
        over-broad token and any logged response body can exfiltrate - and
        nothing in the product needs to read it, because it is used
        server-side only.
        """
        response = self._create(
            client_for(self.alice, self.acme),
            service=self.openai.pk,
            api_key="sk-live-abcd1234",
        )

        self.assertEqual(response.status_code, 201)
        self.assertNotIn("api_key", response.data["data"])
        self.assertNotIn("sk-live-abcd1234", str(response.data))

        stored = ProviderCredential.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(stored.api_key, "sk-live-abcd1234")

    def test_the_hint_shows_enough_to_tell_two_keys_apart(self):
        response = self._create(
            client_for(self.alice, self.acme),
            service=self.openai.pk,
            api_key="sk-live-abcd1234",
        )
        self.assertTrue(response.data["data"]["api_key_hint"].endswith("1234"))
        self.assertTrue(response.data["data"]["has_api_key"])

    def test_credentials_are_scoped_to_the_organization(self):
        ProviderCredential.objects.create(organization=self.beta, service=self.openai)
        response = client_for(self.alice, self.acme).get(
            reverse("api-organization:credentials")
        )
        self.assertEqual(response.data["data"], [])

    def test_a_second_embedding_default_replaces_the_first(self):
        """A partial unique index allows one per organization, so without
        unsetting the old one this is an IntegrityError the user reads as
        "something went wrong" rather than "this one instead"."""
        client = client_for(self.alice, self.acme)
        first = self._create(
            client, service=self.openai.pk, api_key="sk-a", is_embedding_default=True
        )
        second = self._create(
            client, service=self.groq.pk, api_key="gsk-b", is_embedding_default=True
        )

        self.assertEqual(second.status_code, 201)
        self.assertFalse(
            ProviderCredential.objects.get(
                pk=first.data["data"]["id"]
            ).is_embedding_default
        )
        self.assertEqual(
            ProviderCredential.objects.filter(
                organization=self.acme, is_embedding_default=True
            ).count(),
            1,
        )

    def test_a_second_key_for_the_same_provider_is_allowed(self):
        """What this endpoint used to refuse, and now must not."""
        client = client_for(self.alice, self.acme)
        self._create(client, service=self.openai.pk, api_key="sk-a", name="Shared")
        response = self._create(
            client, service=self.openai.pk, api_key="sk-b", name="Billing: Acme"
        )
        self.assertEqual(response.status_code, 201)

    def test_a_name_the_user_typed_is_not_silently_renamed(self):
        """Auto-generated names disambiguate; typed ones do not.

        Quietly turning somebody's "Billing: Acme" into "Billing: Acme 2" is
        how two keys end up looking interchangeable when they bill different
        accounts.
        """
        client = client_for(self.alice, self.acme)
        self._create(client, service=self.openai.pk, api_key="sk-a", name="Shared")
        response = self._create(
            client, service=self.groq.pk, api_key="sk-b", name="Shared"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("name", response.data)

    def test_an_unnamed_key_is_named_after_its_provider(self):
        """A list of keys with no names is not something anybody can act on."""
        client = client_for(self.alice, self.acme)
        first = self._create(client, service=self.openai.pk, api_key="sk-a")
        second = self._create(client, service=self.openai.pk, api_key="sk-b")

        self.assertEqual(first.data["data"]["name"], "OpenAI")
        # Disambiguated rather than rejected.
        self.assertEqual(second.data["data"]["name"], "OpenAI 2")

    def test_the_first_key_for_a_provider_becomes_its_default(self):
        """A single-key organization should not have to make a choice it does
        not yet have."""
        client = client_for(self.alice, self.acme)
        first = self._create(client, service=self.openai.pk, api_key="sk-a")
        self.assertTrue(first.data["data"]["is_default"])

        second = self._create(client, service=self.openai.pk, api_key="sk-b")
        self.assertFalse(second.data["data"]["is_default"])

    def test_promoting_a_second_key_demotes_the_first(self):
        client = client_for(self.alice, self.acme)
        first = self._create(client, service=self.openai.pk, api_key="sk-a")
        second = self._create(client, service=self.openai.pk, api_key="sk-b")

        response = client.patch(
            reverse(
                "api-organization:credential-detail",
                args=[second.data["data"]["id"]],
            ),
            {"is_default": True},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(
            ProviderCredential.objects.get(pk=first.data["data"]["id"]).is_default
        )
        self.assertEqual(
            ProviderCredential.objects.filter(
                organization=self.acme, service=self.openai, is_default=True
            ).count(),
            1,
        )

    def test_editing_without_sending_a_key_keeps_the_existing_one(self):
        """The edit form cannot show the key, so it cannot send it back.

        Treating absent as "clear it" would wipe the credential every time
        somebody toggled the embedding-default flag.
        """
        client = client_for(self.alice, self.acme)
        created = self._create(client, service=self.openai.pk, api_key="sk-keep-me")

        response = client.patch(
            reverse(
                "api-organization:credential-detail",
                args=[created.data["data"]["id"]],
            ),
            {"is_embedding_default": True},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        stored = ProviderCredential.objects.get(pk=created.data["data"]["id"])
        self.assertEqual(stored.api_key, "sk-keep-me")
        self.assertTrue(stored.is_embedding_default)

    def test_a_non_owner_cannot_add_credentials(self):
        carol = make_account("carol")
        add_member(self.acme, carol, self.alice)
        response = self._create(
            client_for(carol, self.acme), service=self.openai.pk, api_key="sk-x"
        )
        self.assertEqual(response.status_code, 403)

    def test_a_non_owner_can_still_see_that_credentials_exist(self):
        """Members need to know whether the organization is configured; they
        just cannot change it."""
        carol = make_account("carol")
        add_member(self.acme, carol, self.alice)
        ProviderCredential.objects.create(organization=self.acme, service=self.openai)

        response = client_for(carol, self.acme).get(
            reverse("api-organization:credentials")
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["data"]), 1)

    def test_a_credential_from_another_organization_is_not_found(self):
        theirs = ProviderCredential.objects.create(
            organization=self.beta, service=self.openai
        )
        response = client_for(self.alice, self.acme).get(
            reverse("api-organization:credential-detail", args=[str(theirs.pk)])
        )
        self.assertEqual(response.status_code, 404)


class EmbeddingCredentialResolutionTests(TestCase):
    """Which key RAG embeds with - the bug that made RAG a no-op for Groq orgs."""

    def setUp(self):
        reset_cipher()
        self.alice = make_account("alice")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.openai = Service.objects.create(name="OpenAI")
        self.groq = Service.objects.create(name="Groq")

    def tearDown(self):
        reset_cipher()

    def _credential(self, service, key, **extra):
        credential = ProviderCredential(
            organization=self.acme, service=service, **extra
        )
        credential.api_key = key
        credential.save()
        return credential

    def test_the_flagged_credential_wins(self):
        self._credential(self.openai, "sk-embed", is_embedding_default=True)
        self._credential(self.groq, "gsk-chat")
        self.assertEqual(embedding_api_key(self.acme), "sk-embed")

    def test_an_openai_credential_is_used_even_without_the_flag(self):
        """Embeddings are OpenAI-only, so a single OpenAI key is unambiguous."""
        self._credential(self.openai, "sk-embed")
        self.assertEqual(embedding_api_key(self.acme), "sk-embed")

    def test_the_legacy_field_still_works(self):
        """Nothing breaks between this deploying and a key being moved across."""
        self.acme.llm_api_key = "sk-legacy"
        self.acme.save()
        self.assertEqual(embedding_api_key(self.acme), "sk-legacy")

    def test_a_groq_only_organization_is_told_rather_than_silently_failing(self):
        """The actual bug: `organization.llm_api_key` returned the Groq key,
        which OpenAI's embeddings endpoint rejects - so retrieval came back
        empty with nothing anyone would connect to a missing credential."""
        self._credential(self.groq, "gsk-chat")
        with self.assertRaises(CredentialNotConfigured) as caught:
            embedding_api_key(self.acme)
        self.assertIn("embedding", str(caught.exception))

    def test_the_status_endpoint_reports_the_gap(self):
        self._credential(self.groq, "gsk-chat")
        response = client_for(self.alice, self.acme).get(
            reverse("api-organization:credentials-embedding")
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["data"]["configured"])
        self.assertTrue(response.data["data"]["reason"])
