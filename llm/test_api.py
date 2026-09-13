"""The agent-building API.

Two organizations throughout, for the same reason as organization/test_api.py:
the interesting failures are all cross-tenant, and a single-tenant fixture
cannot catch one.

The knowledge tests patch `.delay`. `CELERY_TASK_ALWAYS_EAGER` is on in
settings_test, so an unpatched `.delay()` would run the indexing task inline
and make real OpenAI and Pinecone calls from the test suite.
"""

from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from llm.models import Agent, KnowledgeDocument, Model, Role, Service, Tool
from neon.testing import add_member, client_for, make_account, make_organization
from neon.utils import crypto
from organization.models import ProviderCredential


def reset_cipher():
    crypto._cipher = None


class TwoOrganizations(TestCase):
    """Shared fixture: Acme (alice) and Beta (bob)."""

    def setUp(self):
        reset_cipher()
        self.alice = make_account("alice")
        self.bob = make_account("bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")
        self.client_acme = client_for(self.alice, self.acme)
        self.client_beta = client_for(self.bob, self.beta)

    def tearDown(self):
        reset_cipher()


class AgentAPITests(TwoOrganizations):

    def test_creating_an_agent_derives_its_slug(self):
        response = self.client_acme.post(
            reverse("api-llm:agents"), {"name": "Support Helper"}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["slug"], "support-helper")

        agent = Agent.objects.get(uuid=response.data["data"]["uuid"])
        self.assertEqual(agent.organization, self.acme)
        self.assertEqual(agent.created_by, self.alice)

    def test_two_organizations_can_use_the_same_slug(self):
        self.client_acme.post(
            reverse("api-llm:agents"), {"name": "Helper"}, format="json"
        )
        response = self.client_beta.post(
            reverse("api-llm:agents"), {"name": "Helper"}, format="json"
        )
        self.assertEqual(response.status_code, 201)

    def test_a_repeated_slug_is_a_field_error_not_a_crash(self):
        self.client_acme.post(
            reverse("api-llm:agents"), {"name": "Helper"}, format="json"
        )
        response = self.client_acme.post(
            reverse("api-llm:agents"), {"name": "Helper"}, format="json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("slug", response.data)

    def test_listing_shows_only_your_own_agents(self):
        Agent.objects.create(organization=self.beta, name="Theirs", slug="theirs")
        Agent.objects.create(organization=self.acme, name="Ours", slug="ours")

        response = self.client_acme.get(reverse("api-llm:agents"))
        self.assertEqual([a["name"] for a in response.data["data"]], ["Ours"])

    def test_another_organizations_agent_is_not_found(self):
        theirs = Agent.objects.create(
            organization=self.beta, name="Theirs", slug="theirs"
        )
        response = self.client_acme.get(
            reverse("api-llm:agent-detail", args=[theirs.uuid])
        )
        self.assertEqual(response.status_code, 404)

    def test_an_agent_cannot_be_given_another_organizations_role(self):
        """The serializer's queryset is the authorization check.

        Filtering only what you READ is not enough: an unrestricted
        PrimaryKeyRelatedField would attach Beta's role - and with it Beta's
        system prompt and tools - to an agent in Acme.
        """
        theirs = Role.objects.create(
            organization=self.beta, name="Theirs", system_prompt="secret"
        )
        response = self.client_acme.post(
            reverse("api-llm:agents"),
            {"name": "Helper", "role_id": theirs.pk},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("role_id", response.data)

    def test_an_agent_can_be_given_its_own_organizations_role(self):
        ours = Role.objects.create(
            organization=self.acme, name="Support", system_prompt="be helpful"
        )
        response = self.client_acme.post(
            reverse("api-llm:agents"),
            {"name": "Helper", "role_id": ours.pk},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["role"]["name"], "Support")

    def test_deleting_an_agent_deactivates_it(self):
        """Messages carry an FK to the agent that wrote them, so a hard delete
        would leave a conversation history unable to say who answered."""
        agent = Agent.objects.create(
            organization=self.acme, name="Helper", slug="helper"
        )
        response = self.client_acme.delete(
            reverse("api-llm:agent-detail", args=[agent.uuid])
        )
        self.assertEqual(response.status_code, 200)
        agent.refresh_from_db()
        self.assertFalse(agent.is_active)


class RoleAPITests(TwoOrganizations):

    def test_tools_can_be_attached_by_id(self):
        tool = Tool.objects.create(organization=self.acme, name="search")
        response = self.client_acme.post(
            reverse("api-llm:roles"),
            {"name": "Support", "system_prompt": "be helpful", "tool_ids": [tool.pk]},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            [t["name"] for t in response.data["data"]["tools"]], ["search"]
        )

    def test_another_organizations_tool_cannot_be_attached(self):
        """The leak this prevents is the endpoint AND the credential: a role
        can call any tool attached to it."""
        theirs = Tool.objects.create(organization=self.beta, name="internal")
        response = self.client_acme.post(
            reverse("api-llm:roles"),
            {
                "name": "Support",
                "system_prompt": "be helpful",
                "tool_ids": [theirs.pk],
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("tool_ids", response.data)

    def test_a_role_in_use_cannot_be_deleted(self):
        """Agent.role is SET_NULL, so deleting a role silently strips the
        system prompt from every agent using it."""
        role = Role.objects.create(
            organization=self.acme, name="Support", system_prompt="x"
        )
        Agent.objects.create(
            organization=self.acme, name="Helper", slug="helper", role=role
        )

        response = self.client_acme.delete(
            reverse("api-llm:role-detail", args=[role.pk])
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("Helper", response.data["message"])
        self.assertTrue(Role.objects.filter(pk=role.pk).exists())

    def test_an_unused_role_can_be_deleted(self):
        role = Role.objects.create(
            organization=self.acme, name="Support", system_prompt="x"
        )
        response = self.client_acme.delete(
            reverse("api-llm:role-detail", args=[role.pk])
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Role.objects.filter(pk=role.pk).exists())

    def test_roles_are_scoped(self):
        Role.objects.create(organization=self.beta, name="Theirs", system_prompt="x")
        response = self.client_acme.get(reverse("api-llm:roles"))
        self.assertEqual(response.data["data"], [])


class ToolAPITests(TwoOrganizations):

    def test_authentication_can_be_written_but_never_read(self):
        """This serializer was `fields = "__all__"`, and its output was
        json.dumps'd into the system prompt - putting every tool's credential
        into the provider's logs and anything the model could be talked into
        repeating."""
        response = self.client_acme.post(
            reverse("api-llm:tools"),
            {
                "name": "search",
                "api_endpoint": "https://example.com/search",
                "authentication": "Bearer super-secret",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertNotIn("authentication", response.data["data"])
        self.assertNotIn("super-secret", str(response.data))
        self.assertTrue(response.data["data"]["has_authentication"])

        stored = Tool.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(stored.authentication, "Bearer super-secret")

    def test_the_credential_is_absent_from_listings_too(self):
        Tool.objects.create(
            organization=self.acme, name="search", authentication="Bearer secret"
        )
        response = self.client_acme.get(reverse("api-llm:tools"))
        self.assertNotIn("secret", str(response.data))

    def test_an_enabled_tool_needs_an_endpoint(self):
        """Otherwise the model picks the tool and the call fails - which reads
        as a broken agent rather than a misconfigured tool."""
        response = self.client_acme.post(
            reverse("api-llm:tools"),
            {"name": "search", "is_enabled": True},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("api_endpoint", response.data)

    def test_a_malformed_parameters_schema_is_rejected(self):
        """A bad schema makes the provider reject the whole completion, so one
        broken tool takes out every conversation in the organization."""
        response = self.client_acme.post(
            reverse("api-llm:tools"),
            {"name": "search", "parameters_schema": ["not", "an", "object"]},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("parameters_schema", response.data)

    def test_two_organizations_can_own_a_tool_called_search(self):
        self.client_acme.post(
            reverse("api-llm:tools"), {"name": "search"}, format="json"
        )
        response = self.client_beta.post(
            reverse("api-llm:tools"), {"name": "search"}, format="json"
        )
        self.assertEqual(response.status_code, 201)

    def test_a_repeated_name_within_one_organization_is_a_conflict(self):
        self.client_acme.post(
            reverse("api-llm:tools"), {"name": "search"}, format="json"
        )
        response = self.client_acme.post(
            reverse("api-llm:tools"), {"name": "search"}, format="json"
        )
        self.assertEqual(response.status_code, 409)

    def test_another_organizations_tool_is_not_found(self):
        theirs = Tool.objects.create(organization=self.beta, name="internal")
        response = self.client_acme.get(
            reverse("api-llm:tool-detail", args=[theirs.pk])
        )
        self.assertEqual(response.status_code, 404)


class CatalogueTests(TwoOrganizations):
    """Services and models are platform data, shared by every organization."""

    def test_services_are_listed(self):
        Service.objects.create(name="OpenAI")
        response = self.client_acme.get(reverse("api-llm:services"))
        self.assertEqual([s["name"] for s in response.data["data"]], ["OpenAI"])

    def test_models_can_be_filtered_by_service(self):
        openai = Service.objects.create(name="OpenAI")
        groq = Service.objects.create(name="Groq")
        Model.objects.create(service=openai, model="gpt-4o-mini")
        Model.objects.create(service=groq, model="llama-3.1-8b")

        response = self.client_acme.get(
            reverse("api-llm:models"), {"service": openai.uuid}
        )
        self.assertEqual(
            [m["model"] for m in response.data["data"]], ["gpt-4o-mini"]
        )

    def test_services_are_read_only(self):
        response = self.client_acme.post(
            reverse("api-llm:services"), {"name": "Mine"}, format="json"
        )
        self.assertEqual(response.status_code, 405)


class KnowledgeAPITests(TwoOrganizations):

    def setUp(self):
        super().setUp()
        self.openai = Service.objects.create(name="OpenAI")
        credential = ProviderCredential(
            organization=self.acme, service=self.openai, is_embedding_default=True
        )
        credential.api_key = "sk-test"
        credential.save()

    def test_uploading_text_queues_indexing(self):
        with patch(
            "llm.views.index_knowledge_document_task.delay"
        ) as queued:
            response = self.client_acme.post(
                reverse("api-llm:knowledge"),
                {"title": "Refund policy", "content": "Refunds within 30 days."},
                format="json",
            )

        self.assertEqual(response.status_code, 201)
        document = KnowledgeDocument.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(document.status, KnowledgeDocument.STATUS_PENDING)
        self.assertEqual(document.organization, self.acme)
        queued.assert_called_once_with(document.pk)

    def test_uploading_a_text_file_works(self):
        upload = SimpleUploadedFile(
            "refund_policy.md", b"# Refunds\n\nWithin 30 days.", content_type="text/markdown"
        )
        with patch("llm.views.index_knowledge_document_task.delay"):
            response = self.client_acme.post(
                reverse("api-llm:knowledge"), {"file": upload}, format="multipart"
            )

        self.assertEqual(response.status_code, 201)
        document = KnowledgeDocument.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(document.source_name, "refund_policy.md")
        self.assertEqual(document.title, "refund policy")
        self.assertIn("Within 30 days", document.content)

    def test_a_binary_document_is_refused_rather_than_indexed_as_noise(self):
        """Indexing the raw bytes of a PDF fills the index with binary that
        quietly degrades every answer - worse than refusing the upload."""
        upload = SimpleUploadedFile(
            "handbook.pdf", b"%PDF-1.4\x00\x01binary", content_type="application/pdf"
        )
        response = self.client_acme.post(
            reverse("api-llm:knowledge"), {"file": upload}, format="multipart"
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("converted to text", response.data["message"])
        self.assertFalse(KnowledgeDocument.objects.exists())

    def test_a_file_that_is_not_utf8_is_named_rather_than_mangled(self):
        upload = SimpleUploadedFile(
            "notes.txt", b"\xff\xfe\x00broken", content_type="text/plain"
        )
        response = self.client_acme.post(
            reverse("api-llm:knowledge"), {"file": upload}, format="multipart"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("UTF-8", response.data["message"])

    def test_an_organization_with_no_embedding_key_is_told_up_front(self):
        """Checked before the row is written. Otherwise the document is stored,
        queued, and fails on the worker - so the user sees "failed" with a
        provider error instead of the thing they can fix."""
        response = self.client_beta.post(
            reverse("api-llm:knowledge"),
            {"title": "Policy", "content": "text"},
            format="json",
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("embedding", response.data["message"])
        self.assertFalse(KnowledgeDocument.objects.exists())

    def test_an_empty_upload_is_rejected(self):
        response = self.client_acme.post(
            reverse("api-llm:knowledge"), {"title": "Nothing"}, format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_documents_are_scoped_to_the_organization(self):
        KnowledgeDocument.objects.create(
            organization=self.beta, title="Theirs", content="secret"
        )
        response = self.client_acme.get(reverse("api-llm:knowledge"))
        self.assertEqual(response.data["data"], [])

    def test_another_organizations_document_is_not_found(self):
        theirs = KnowledgeDocument.objects.create(
            organization=self.beta, title="Theirs", content="secret"
        )
        response = self.client_acme.get(
            reverse("api-llm:knowledge-detail", args=[theirs.pk])
        )
        self.assertEqual(response.status_code, 404)

    def test_the_listing_omits_content_and_the_detail_includes_it(self):
        document = KnowledgeDocument.objects.create(
            organization=self.acme, title="Policy", content="the full text"
        )
        listing = self.client_acme.get(reverse("api-llm:knowledge"))
        self.assertNotIn("content", listing.data["data"][0])

        detail = self.client_acme.get(
            reverse("api-llm:knowledge-detail", args=[document.pk])
        )
        self.assertEqual(detail.data["data"]["content"], "the full text")

    def test_deleting_a_document_queues_its_vectors_for_removal(self):
        document = KnowledgeDocument.objects.create(
            organization=self.acme,
            title="Policy",
            content="text",
            vector_ids=["doc_1", "doc_2"],
        )

        with patch("llm.views.delete_knowledge_vectors_task.delay") as queued:
            response = self.client_acme.delete(
                reverse("api-llm:knowledge-detail", args=[document.pk])
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeDocument.objects.filter(pk=document.pk).exists())
        # The organization id travels with it: vectors live in that
        # organization's Pinecone namespace now, and a delete that did not
        # name it would look successful and remove nothing.
        queued.assert_called_once_with(["doc_1", "doc_2"], str(self.acme.pk))

    def test_re_indexing_resets_the_status_and_queues_again(self):
        document = KnowledgeDocument.objects.create(
            organization=self.acme,
            title="Policy",
            content="text",
            status=KnowledgeDocument.STATUS_FAILED,
            error="boom",
        )

        with patch("llm.views.index_knowledge_document_task.delay") as queued:
            response = self.client_acme.post(
                reverse("api-llm:knowledge-detail", args=[document.pk])
            )

        self.assertEqual(response.status_code, 200)
        document.refresh_from_db()
        self.assertEqual(document.status, KnowledgeDocument.STATUS_PENDING)
        self.assertEqual(document.error, "")
        queued.assert_called_once_with(document.pk)


class ScopingRequiresMembershipTests(TestCase):
    """A signed-in account with no organization gets a clear 403, not a 500."""

    def test_every_scoped_route_refuses_an_account_with_no_organization(self):
        account = make_account("stranger")
        client = client_for(account)
        for name in ("api-llm:agents", "api-llm:roles", "api-llm:tools",
                     "api-llm:knowledge", "api-organization:members"):
            with self.subTest(route=name):
                response = client.get(reverse(name))
                self.assertEqual(response.status_code, 403)
                self.assertFalse(response.data["status"])
