"""Which agents can read which documents.

The thing under test is a confidentiality boundary, so most of these assert a
NEGATIVE: that a document does not reach an agent it was not given to. A test
that only proves the happy path would pass just as well against the old
behaviour, where every document reached every agent.
"""

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from llm.knowledge import document_filter_terms, restricted_document_ids
from llm.models import Agent, KnowledgeDocument, Role
from llm.services.rag import CustomerServiceRAG
from llm.test_rag import CONVERSATION, FakeIndex, build_rag
from neon.testing import client_for, make_account, make_organization


class ScopingTestCase(TestCase):
    def setUp(self):
        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")
        self.client_acme = client_for(self.alice, self.org)

        role = Role.objects.create(
            organization=self.org, name="Support", system_prompt="Help."
        )
        self.billing = Agent.objects.create(
            organization=self.org, name="Billing", slug="billing", role=role
        )
        self.engineering = Agent.objects.create(
            organization=self.org, name="Engineering", slug="eng", role=role
        )

    def _document(self, title, agents=(), vector_ids=("v1", "v2")):
        document = KnowledgeDocument.objects.create(
            organization=self.org,
            title=title,
            content="text",
            status=KnowledgeDocument.STATUS_INDEXED,
            vector_ids=list(vector_ids),
        )
        if agents:
            document.agents.set(agents)
        return document


class FilterTests(ScopingTestCase):
    """What the query actually asks Pinecone for."""

    def _doc_terms(self, terms):
        return [t for t in terms if t.get("type") == "doc"]

    def test_the_shared_corpus_is_always_in_scope(self):
        terms = document_filter_terms(self.billing, CONVERSATION)
        self.assertIn({"type": "doc", "shared": True}, terms)

    def test_a_restricted_document_reaches_the_agent_it_was_given_to(self):
        document = self._document("Price list", agents=[self.billing])
        terms = document_filter_terms(self.billing, CONVERSATION)
        ids = [
            t["document_id"]["$in"]
            for t in self._doc_terms(terms)
            if "document_id" in t
        ]
        self.assertEqual(ids, [[str(document.pk)]])

    def test_a_restricted_document_does_not_reach_another_agent(self):
        """The whole point. Engineering must not be able to match it at all."""
        self._document("Price list", agents=[self.billing])
        terms = document_filter_terms(self.engineering, CONVERSATION)
        self.assertEqual(
            [t for t in self._doc_terms(terms) if "document_id" in t], []
        )

    def test_no_agent_means_shared_only_rather_than_everything(self):
        """An unidentified caller is the one case where failing open would be
        invisible: retrieval would simply return more than it should."""
        self._document("Price list", agents=[self.billing])
        terms = document_filter_terms(None, CONVERSATION)
        self.assertEqual(
            [t for t in self._doc_terms(terms) if "document_id" in t], []
        )
        self.assertIn({"type": "doc", "shared": True}, terms)

    def test_an_unrestricted_document_is_not_enumerated(self):
        """Shared documents are matched by their flag. Listing them too would
        make the filter grow with the corpus, which is what the size limit on
        a Pinecone filter eventually refuses."""
        self._document("Handbook")
        self.assertEqual(restricted_document_ids(self.billing), [])

    def test_another_organizations_assignment_is_not_honoured(self):
        other = make_organization(make_account("bob"), "Beta", "beta")
        stray = Agent.objects.create(
            organization=other, name="Theirs", slug="theirs"
        )
        document = self._document("Price list")
        document.agents.set([stray])
        self.assertEqual(restricted_document_ids(stray), [])

    def test_one_document_can_be_given_to_several_agents(self):
        """Assignment is many-to-many, not an owner field: a pricing sheet that
        billing and sales both need must not have to be uploaded twice, or the
        two copies drift and half the agents answer from the stale one."""
        document = self._document("Price list", agents=[self.billing, self.engineering])

        for agent in (self.billing, self.engineering):
            self.assertEqual(
                restricted_document_ids(agent),
                [str(document.pk)],
                f"{agent.name} should see it",
            )
        self.assertEqual(document.agents.count(), 2)

    def test_one_agent_can_hold_several_documents(self):
        first = self._document("Price list", agents=[self.billing])
        second = self._document("Refunds", agents=[self.billing, self.engineering])
        third = self._document("Architecture", agents=[self.engineering])

        self.assertEqual(
            sorted(restricted_document_ids(self.billing)),
            sorted([str(first.pk), str(second.pk)]),
        )
        self.assertNotIn(str(third.pk), restricted_document_ids(self.billing))

        terms = document_filter_terms(self.billing, CONVERSATION)
        named = [t for t in self._doc_terms(terms) if "document_id" in t]
        self.assertEqual(len(named), 1, "one $in term, not one term per document")
        self.assertEqual(len(named[0]["document_id"]["$in"]), 2)

    def test_removing_one_agent_leaves_the_others(self):
        document = self._document("Price list", agents=[self.billing, self.engineering])
        document.agents.remove(self.engineering)

        self.assertEqual(restricted_document_ids(self.billing), [str(document.pk)])
        self.assertEqual(restricted_document_ids(self.engineering), [])
        self.assertFalse(document.is_shared, "still restricted, just to fewer")

    def test_the_conversation_is_still_in_scope(self):
        terms = document_filter_terms(self.billing, CONVERSATION)
        self.assertIn({"conversation_id": str(CONVERSATION)}, terms)


class RetrievalTests(ScopingTestCase):
    """The filter reaches the index."""

    def setUp(self):
        super().setUp()
        self.index = FakeIndex()
        self.rag = build_rag(self.index)
        patcher = patch.object(
            CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _filter_for(self, agent):
        self.rag.retrieve("question", CONVERSATION, str(self.org.pk), "sk", agent=agent)
        return self.index.queries[0]["filter"]

    def test_an_agents_own_document_is_named_in_the_query(self):
        document = self._document("Price list", agents=[self.billing])
        sent = self._filter_for(self.billing)
        self.assertIn(
            {"type": "doc", "document_id": {"$in": [str(document.pk)]}}, sent["$or"]
        )

    def test_another_agents_document_is_absent_from_the_query(self):
        self._document("Price list", agents=[self.billing])
        sent = self._filter_for(self.engineering)
        self.assertNotIn(
            "document_id", " ".join(str(term) for term in sent["$or"])
        )


class IndexingTests(ScopingTestCase):
    """Vectors carry what the filter matches on."""

    def setUp(self):
        super().setUp()
        self.index = FakeIndex()
        self.rag = build_rag(self.index)

    def _index(self, **kwargs):
        with patch.object(CustomerServiceRAG, "_client") as client:
            client.return_value.embeddings.create.return_value = type(
                "R", (), {"data": [type("E", (), {"embedding": [0.0] * 8})()]}
            )()
            self.rag.bulk_index_docs(["a policy"], "sk", str(self.org.pk), **kwargs)
        return self.index.upserts[0]["vectors"][0]["metadata"]

    def test_a_shared_document_is_flagged_shared(self):
        metadata = self._index(document_id="doc-1", shared=True)
        self.assertTrue(metadata["shared"])
        self.assertEqual(metadata["document_id"], "doc-1")

    def test_a_restricted_document_is_not(self):
        metadata = self._index(document_id="doc-1", shared=False)
        self.assertFalse(metadata["shared"])

    def test_indexing_without_a_document_id_omits_it(self):
        """Chat vectors and ad-hoc indexing have no row to point at, and a
        `document_id` of "None" would be a filterable value that matches
        nothing while looking like it might."""
        metadata = self._index()
        self.assertNotIn("document_id", metadata)


class ApiTests(ScopingTestCase):

    def _upload(self, payload):
        with patch("llm.views.index_knowledge_document_task"):
            return self.client_acme.post(
                reverse("api-llm:knowledge"), payload, format="json"
            )

    def test_an_upload_defaults_to_shared(self):
        with patch("llm.views.embedding_api_key", return_value="sk"):
            response = self._upload({"title": "Handbook", "content": "hello"})
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.data["data"]["is_shared"])

    def test_an_upload_can_be_restricted_to_agents(self):
        with patch("llm.views.embedding_api_key", return_value="sk"):
            response = self._upload(
                {
                    "title": "Price list",
                    "content": "hello",
                    "agent_uuids": [self.billing.uuid],
                }
            )
        self.assertEqual(response.status_code, 201)
        self.assertFalse(response.data["data"]["is_shared"])
        self.assertEqual(
            [a["name"] for a in response.data["data"]["agents"]], ["Billing"]
        )

    def test_the_assignment_is_set_before_indexing_is_queued(self):
        """Otherwise the worker stamps `shared: True` onto a document that was
        meant to be restricted, and the vectors disagree with the row."""
        seen = {}

        def capture(document_id):
            seen["shared"] = KnowledgeDocument.objects.get(pk=document_id).is_shared

        with patch("llm.views.embedding_api_key", return_value="sk"), \
             patch("llm.views.index_knowledge_document_task") as task:
            task.delay.side_effect = capture
            self.client_acme.post(
                reverse("api-llm:knowledge"),
                {"title": "P", "content": "x", "agent_uuids": [self.billing.uuid]},
                format="json",
            )
        self.assertFalse(seen["shared"])

    def test_another_organizations_agent_cannot_be_assigned(self):
        other = make_organization(make_account("bob"), "Beta", "beta")
        stray = Agent.objects.create(organization=other, name="T", slug="t")

        with patch("llm.views.embedding_api_key", return_value="sk"):
            response = self._upload(
                {"title": "P", "content": "x", "agent_uuids": [stray.uuid]}
            )
        self.assertEqual(response.status_code, 404)
        self.assertFalse(KnowledgeDocument.objects.exists())

    def test_a_document_can_be_restricted_after_the_fact(self):
        document = self._document("Handbook")
        with patch("llm.views.sync_document_scoping_task") as task:
            response = self.client_acme.patch(
                reverse("api-llm:knowledge-detail", args=[str(document.pk)]),
                {"agent_uuids": [self.billing.uuid]},
                format="json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["data"]["is_shared"])
        task.delay.assert_called_once_with(document.pk)

    def test_restricting_is_not_a_one_way_door(self):
        document = self._document("Handbook", agents=[self.billing])
        with patch("llm.views.sync_document_scoping_task") as task:
            response = self.client_acme.patch(
                reverse("api-llm:knowledge-detail", args=[str(document.pk)]),
                {"agent_uuids": []},
                format="json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["data"]["is_shared"])
        task.delay.assert_called_once_with(document.pk)

    def test_changing_which_agents_costs_no_pinecone_write(self):
        """Restricted stays restricted, so the `shared` flag on the vectors is
        unchanged and the only thing that moved is a Postgres row."""
        document = self._document("Price list", agents=[self.billing])
        with patch("llm.views.sync_document_scoping_task") as task:
            self.client_acme.patch(
                reverse("api-llm:knowledge-detail", args=[str(document.pk)]),
                {"agent_uuids": [self.engineering.uuid]},
                format="json",
            )
        task.delay.assert_not_called()
        self.assertEqual(restricted_document_ids(self.engineering), [str(document.pk)])

    def test_a_patch_without_the_field_is_refused(self):
        """An empty list means "share with everyone". A missing field must not
        be read as the same thing, or a PATCH of an unrelated field would
        silently unrestrict the document."""
        document = self._document("Price list", agents=[self.billing])
        response = self.client_acme.patch(
            reverse("api-llm:knowledge-detail", args=[str(document.pk)]),
            {"title": "Renamed"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        document.refresh_from_db()
        self.assertFalse(document.is_shared)
