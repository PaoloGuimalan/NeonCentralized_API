"""Retrieval: namespaces, deduplication, ordering and the settings it honours.

No Pinecone and no OpenAI. The index is a fake that records every call, which
is what lets these assert the things that matter here - which NAMESPACE a query
went to, and whether a second query happened at all.
"""

from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.utils.timezone import now

from llm.services import rag as rag_module
from llm.services.rag import (
    LEGACY_NAMESPACE,
    MODE_DUAL,
    MODE_NAMESPACED,
    CustomerServiceRAG,
    _kind,
    _normalise,
)
from messenger.models import Conversation, Message
from neon.testing import make_account, make_organization

CONVERSATION = "11111111-1111-4111-8111-111111111111"

RAG_SETTINGS = {
    "PINECONE_API_KEY": "pcsk-test",
    "PINECONE_INDEX": "neon-test",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "DIMENSION": "1536",
    "CHUNK_SIZE": "2000",
    "RERANKER_MODEL": "bge-reranker-v2-m3",
    "NAMESPACE_MODE": MODE_DUAL,
}


def match(text, kind="doc", score=0.9, **extra):
    metadata = {"text": text, "type": kind}
    metadata.update(extra)
    return {"id": f"v-{abs(hash(text)) % 10000}", "score": score, "metadata": metadata}


class FakeIndex:
    """Records what it was asked, and by which namespace."""

    def __init__(self, by_namespace=None, legacy_count=0):
        self.by_namespace = by_namespace or {}
        self.legacy_count = legacy_count
        self.queries = []
        self.upserts = []
        self.deletes = []

    def query(self, vector, top_k, filter, include_metadata, namespace):
        self.queries.append({"namespace": namespace, "filter": filter, "top_k": top_k})
        return {"matches": list(self.by_namespace.get(namespace, []))}

    def upsert(self, vectors, namespace=None):
        self.upserts.append({"namespace": namespace, "vectors": vectors})

    def delete(self, ids, namespace=None):
        self.deletes.append({"namespace": namespace, "ids": list(ids)})

    def describe_index_stats(self):
        return {"namespaces": {LEGACY_NAMESPACE: {"vector_count": self.legacy_count}}}


class FakeInference:
    """Reranks by keeping the order it was given - enough to assert plumbing."""

    def __init__(self):
        self.calls = []

    def rerank(self, model, query, documents, top_n):
        self.calls.append({"model": model, "documents": documents, "top_n": top_n})

        class Item:
            def __init__(self, document):
                self.document = document

        class Result:
            def __init__(self, data):
                self.data = data

        return Result([Item(doc) for doc in documents[:top_n]])


def build_rag(index=None, mode=MODE_DUAL, index_dimension=1536, **overrides):
    """A RAG client with Pinecone entirely replaced.

    `index_dimension` is what the FAKE INDEX reports, which is a different
    thing from the DIMENSION setting - the point of having both is that they
    can disagree.
    """
    index = index or FakeIndex()
    dimension = index_dimension

    class FakeDescribed:
        status = {"ready": True}

        def __init__(self, dimension):
            self.dimension = dimension

    class FakePinecone:
        def __init__(self, api_key=None):
            self.inference = FakeInference()

        def list_indexes(self):
            class Named:
                name = "neon-test"

            return [Named()]

        def describe_index(self, name):
            return FakeDescribed(dimension)

        def Index(self, name):
            return index

        def create_index(self, **kwargs):
            raise AssertionError("should not create an index that exists")

    settings_now = dict(RAG_SETTINGS)
    settings_now["NAMESPACE_MODE"] = mode
    settings_now.update(overrides)

    with patch.object(rag_module.pinecone, "Pinecone", FakePinecone):
        with override_settings(RAG=settings_now):
            client = CustomerServiceRAG()
    client.index = index
    return client


class NamespaceTests(TestCase):
    """Tenancy is a partition now, not a metadata filter."""

    def setUp(self):
        self.index = FakeIndex()
        self.rag = build_rag(self.index)
        self.embedding = patch.object(
            CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8
        )
        self.embedding.start()
        self.addCleanup(self.embedding.stop)

    def test_documents_are_written_to_the_organizations_namespace(self):
        with patch.object(CustomerServiceRAG, "_client") as client:
            client.return_value.embeddings.create.return_value = type(
                "R", (), {"data": [type("E", (), {"embedding": [0.0] * 8})()]}
            )()
            self.rag.bulk_index_docs(["a policy"], "sk", "org-1")

        self.assertEqual(self.index.upserts[0]["namespace"], "org-1")

    def test_chat_is_written_to_the_organizations_namespace(self):
        self.rag.index_chat_message("u1", CONVERSATION, "text", "hello", "sk", "org-1")
        self.assertEqual(self.index.upserts[0]["namespace"], "org-1")

    def test_a_query_reads_only_its_own_namespace_once_migrated(self):
        """The whole point: a namespace is a real partition, so the query
        touches one organization's vectors rather than filtering across every
        organization's."""
        index = FakeIndex(by_namespace={"org-1": [match("a doc")]}, legacy_count=0)
        rag = build_rag(index, mode=MODE_NAMESPACED)

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.retrieve("question", CONVERSATION, "org-1", "sk")

        self.assertEqual([q["namespace"] for q in index.queries], ["org-1"])

    def test_the_filter_inside_a_namespace_narrows_by_kind_not_tenant(self):
        index = FakeIndex(by_namespace={"org-1": [match("a doc")]})
        rag = build_rag(index, mode=MODE_NAMESPACED)

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.retrieve("question", CONVERSATION, "org-1", "sk")

        # No organization_id term is needed - every vector in here is already
        # this organization's.
        self.assertNotIn("organization_id", str(index.queries[0]["filter"]))

    def test_the_legacy_query_still_filters_by_organization(self):
        """In the old shared namespace the organization_id term is the ONLY
        thing keeping one tenant out of another's vectors."""
        index = FakeIndex(legacy_count=500)
        rag = build_rag(index, mode=MODE_DUAL)

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.retrieve("question", CONVERSATION, "org-1", "sk")

        legacy = [q for q in index.queries if q["namespace"] == LEGACY_NAMESPACE]
        self.assertEqual(len(legacy), 1)
        self.assertIn("org-1", str(legacy[0]["filter"]))

    def test_dual_mode_reads_both_while_legacy_data_remains(self):
        """Nothing written before the migration may silently disappear."""
        index = FakeIndex(
            by_namespace={
                "org-1": [match("new doc")],
                LEGACY_NAMESPACE: [match("old doc")],
            },
            legacy_count=42,
        )
        rag = build_rag(index, mode=MODE_DUAL)

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            context = rag.retrieve("question", CONVERSATION, "org-1", "sk")

        self.assertEqual(
            sorted(q["namespace"] for q in index.queries), sorted(["", "org-1"])
        )
        texts = [row["text"] for row in context]
        self.assertIn("new doc", texts)
        self.assertIn("old doc", texts)

    def test_a_fresh_deployment_pays_nothing_for_dual_mode(self):
        """The legacy namespace is empty, so the second query is never made."""
        index = FakeIndex(by_namespace={"org-1": [match("a doc")]}, legacy_count=0)
        rag = build_rag(index, mode=MODE_DUAL)

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.retrieve("question", CONVERSATION, "org-1", "sk")

        self.assertEqual([q["namespace"] for q in index.queries], ["org-1"])

    def test_the_legacy_check_costs_one_call_per_process(self):
        index = FakeIndex(legacy_count=0)
        rag = build_rag(index, mode=MODE_DUAL)

        calls = []
        original = index.describe_index_stats

        def counted():
            calls.append(1)
            return original()

        index.describe_index_stats = counted

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            for _ in range(5):
                rag.retrieve("question", CONVERSATION, "org-1", "sk")

        self.assertEqual(len(calls), 1)

    def test_deleting_removes_from_the_organizations_namespace(self):
        index = FakeIndex(legacy_count=0)
        rag = build_rag(index, mode=MODE_DUAL)

        rag.delete_vectors(["doc_1", "doc_2"], organization_id="org-1")

        self.assertEqual([d["namespace"] for d in index.deletes], ["org-1"])

    def test_deleting_also_clears_legacy_while_it_still_has_data(self):
        """A document indexed before the migration has its vectors in the
        default namespace, and a delete that missed them would leave the old
        version answering queries forever."""
        index = FakeIndex(legacy_count=10)
        rag = build_rag(index, mode=MODE_DUAL)

        rag.delete_vectors(["doc_1"], organization_id="org-1")

        self.assertEqual(
            sorted(d["namespace"] for d in index.deletes), sorted(["", "org-1"])
        )


class SettingsTests(TestCase):
    """Settings that existed and were ignored."""

    def test_the_configured_dimension_is_used(self):
        rag = build_rag(FakeIndex(), index_dimension=3072, DIMENSION="3072")
        self.assertEqual(rag.dimension, 3072)

    def test_a_dimension_that_disagrees_with_the_index_is_refused(self):
        """Otherwise every upsert fails with an error naming neither the
        setting nor the index - and on the indexing path that surfaces as
        documents stuck in "failed" for a reason nobody can act on."""
        with self.assertRaises(ValueError) as caught:
            build_rag(FakeIndex(), index_dimension=1536, DIMENSION="768")
        message = str(caught.exception)
        self.assertIn("768", message)
        self.assertIn("1536", message)

    def test_the_configured_embedding_model_is_used(self):
        rag = build_rag(FakeIndex(), EMBEDDING_MODEL="text-embedding-3-large")
        self.assertEqual(rag.embedding_model, "text-embedding-3-large")

    def test_a_blank_setting_falls_back_rather_than_crashing(self):
        rag = build_rag(FakeIndex(), DIMENSION="", EMBEDDING_MODEL=None)
        self.assertEqual(rag.dimension, 1536)
        self.assertEqual(rag.embedding_model, "text-embedding-3-small")


class HistoryTests(TestCase):

    def setUp(self):
        account = make_account("alice")
        org = make_organization(account, "Acme", "acme")
        self.conversation = Conversation.objects.create(
            organization=org, name="Chat", created_by=account
        )
        # Stamped explicitly: auto_now_add gives every row in a tight loop the
        # same timestamp, and "the most recent three" then has no deterministic
        # answer to assert against.
        base = now()
        for offset, content in enumerate(["first", "second", "third", "fourth", "fifth"]):
            message = Message.objects.create(
                conversation=self.conversation, message_type="text", content=content
            )
            Message.objects.filter(pk=message.pk).update(
                created_at=base + timedelta(seconds=offset)
            )
        self.rag = build_rag(FakeIndex())

    def test_history_is_oldest_first(self):
        """The query needs a descending sort to take the most RECENT rows, but
        handing the result to the model in that order made it read the
        conversation backwards - the answer before the question, every turn
        inverted."""
        history = self.rag.get_history(self.conversation.conversation_id, 3)
        self.assertEqual([row["text"] for row in history], ["third", "fourth", "fifth"])

    def test_history_is_limited_to_the_most_recent(self):
        history = self.rag.get_history(self.conversation.conversation_id, 2)
        self.assertEqual([row["text"] for row in history], ["fourth", "fifth"])


class DeduplicationTests(TestCase):
    """The history window and retrieval overlap, and the model was shown both."""

    def setUp(self):
        account = make_account("alice")
        org = make_organization(account, "Acme", "acme")
        self.conversation = Conversation.objects.create(
            organization=org, name="Chat", created_by=account
        )
        Message.objects.create(
            conversation=self.conversation,
            message_type="text",
            content="We agreed on tiered pricing.",
        )

    def _retrieve(self, matches):
        index = FakeIndex(by_namespace={"org-1": matches}, legacy_count=0)
        rag = build_rag(index, mode=MODE_NAMESPACED)
        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            return rag.retrieve(
                "pricing?", self.conversation.conversation_id, "org-1", "sk"
            )

    def test_a_chunk_the_history_already_carries_is_dropped(self):
        """Recent turns are indexed, so a query about the current topic
        reliably retrieves them - and the model was shown the same text twice,
        which wastes context and makes one source look like two."""
        context = self._retrieve(
            [match("We agreed on tiered pricing.", kind="text"), match("Other context")]
        )
        texts = [row["text"] for row in context]
        self.assertEqual(texts.count("We agreed on tiered pricing."), 1)
        self.assertIn("Other context", texts)

    def test_deduplication_ignores_whitespace_and_case(self):
        context = self._retrieve([match("  we agreed on TIERED pricing.  ", kind="text")])
        self.assertEqual(len(context), 1)

    def test_retrieval_appears_after_the_history(self):
        context = self._retrieve([match("Other context")])
        self.assertEqual(context[0]["text"], "We agreed on tiered pricing.")
        self.assertEqual(context[-1]["text"], "Other context")

    def test_duplicate_chunks_within_retrieval_are_collapsed(self):
        context = self._retrieve([match("Same thing"), match("Same thing")])
        self.assertEqual([row["text"] for row in context].count("Same thing"), 1)


class ResilienceTests(TestCase):
    """Retrieval failing should cost the answer its memory, not its existence."""

    def setUp(self):
        account = make_account("alice")
        org = make_organization(account, "Acme", "acme")
        self.conversation = Conversation.objects.create(
            organization=org, name="Chat", created_by=account
        )
        Message.objects.create(
            conversation=self.conversation, message_type="text", content="recent turn"
        )

    def test_an_embedding_failure_still_returns_the_history(self):
        rag = build_rag(FakeIndex())
        with patch.object(
            CustomerServiceRAG, "get_embedding", side_effect=RuntimeError("no key")
        ):
            context = rag.retrieve(
                "q", self.conversation.conversation_id, "org-1", "sk"
            )
        self.assertEqual([row["text"] for row in context], ["recent turn"])

    def test_a_failed_query_does_not_take_down_the_other_namespace(self):
        index = FakeIndex(
            by_namespace={LEGACY_NAMESPACE: [match("old doc")]}, legacy_count=5
        )
        rag = build_rag(index, mode=MODE_DUAL)

        original = index.query

        def flaky(vector, top_k, filter, include_metadata, namespace):
            if namespace == "org-1":
                raise RuntimeError("pinecone hiccup")
            return original(vector, top_k, filter, include_metadata, namespace)

        index.query = flaky

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            context = rag.retrieve(
                "q", self.conversation.conversation_id, "org-1", "sk"
            )

        self.assertIn("old doc", [row["text"] for row in context])

    def test_a_rerank_failure_falls_back_to_vector_order(self):
        """An unranked answer is worse than a ranked one and much better than
        none."""
        index = FakeIndex(by_namespace={"org-1": [match("a doc")]}, legacy_count=0)
        rag = build_rag(index, mode=MODE_NAMESPACED)
        rag.pc.inference.rerank = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("rerank down")
        )

        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            context = rag.retrieve(
                "q", self.conversation.conversation_id, "org-1", "sk"
            )

        self.assertIn("a doc", [row["text"] for row in context])


class MetadataTests(TestCase):

    def test_chat_vectors_record_where_the_exchange_happened(self):
        index = FakeIndex()
        rag = build_rag(index)
        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.index_chat_message(
                "u1", CONVERSATION, "text", "hi", "sk", "org-1", source="chatterloop"
            )
        metadata = index.upserts[0]["vectors"][0]["metadata"]
        self.assertEqual(metadata["source"], "chatterloop")

    def test_the_message_kind_is_written_once(self):
        """`type` and `msg_type` used to carry the same value, so a doc and a
        chat message disagreed about which field held the answer."""
        index = FakeIndex()
        rag = build_rag(index)
        with patch.object(CustomerServiceRAG, "get_embedding", return_value=[0.0] * 8):
            rag.index_chat_message("u1", CONVERSATION, "ai_reply", "hi", "sk", "org-1")
        metadata = index.upserts[0]["vectors"][0]["metadata"]
        self.assertEqual(metadata["type"], "ai_reply")
        self.assertNotIn("msg_type", metadata)

    def test_older_vectors_carrying_msg_type_are_still_read(self):
        self.assertEqual(_kind({"msg_type": "text"}), "text")
        self.assertEqual(_kind({"type": "ai_reply", "msg_type": "ai_reply"}), "ai_reply")
        self.assertEqual(_kind({}), "doc")

    def test_normalising_collapses_whitespace_and_case(self):
        self.assertEqual(_normalise("  Hello   World "), "hello world")
        self.assertEqual(_normalise(None), "")
