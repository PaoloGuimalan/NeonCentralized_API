"""Retrieval: indexing an organization's documents and chat, and reading them back.

TENANCY IS A NAMESPACE, NOT A FILTER
------------------------------------
This used to keep every organization's vectors in one namespace and separate
them with an `organization_id` metadata filter. That is correct but slow in a
specific way: a filtered query still scans the whole index and discards what
does not match, so every organization pays for every other organization's
data on every query, and the cost grows with the platform rather than with the
tenant.

A Pinecone namespace is a real partition. Querying `namespace="<org id>"`
touches only that organization's vectors. It is also a stronger boundary than
a filter: a filter that is forgotten or built wrong returns another tenant's
rows, while a missing namespace returns nothing at all - the safe direction to
fail in.

MIGRATING WITHOUT LOSING WHAT IS ALREADY THERE
----------------------------------------------
Vectors written before this change live in the default namespace. Switching
reads to namespaces would make them invisible - silently, with retrieval simply
returning less and nobody able to tell. So `NAMESPACE_MODE` has two settings:

  * "dual" (the default) - write to the organization's namespace, read BOTH,
    and merge. Nothing disappears, and new writes land in the right place.
  * "namespaced" - read only the namespace. Set this once
    `manage.py migrate_rag_namespaces` reports nothing left behind.

A fresh deployment pays nothing for "dual": the legacy read is skipped
automatically when the default namespace is empty, which is checked once per
process rather than per query.
"""

import logging
import time
import uuid
from datetime import datetime, timezone

import pinecone
from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import ServerlessSpec

from messenger.models import Message

logger = logging.getLogger(__name__)

# Vectors written before tenancy moved to namespaces. Pinecone's default.
LEGACY_NAMESPACE = ""

MODE_DUAL = "dual"
MODE_NAMESPACED = "namespaced"

DEFAULT_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# How many candidates the vector search returns before reranking. Wider than
# the final top_k on purpose: the reranker is what picks well, and it can only
# pick from what it is given.
CANDIDATE_POOL = 30
UPSERT_BATCH = 100


def _setting(key, default):
    value = settings.RAG.get(key)
    return default if value in (None, "") else value


class CustomerServiceRAG:
    def __init__(self):
        # Honoured, rather than hardcoded next to a setting that was ignored.
        # Getting this wrong is not subtle - Pinecone rejects every upsert whose
        # vector length disagrees with the index - but the error names neither
        # the setting nor the index, so `ensure_index` checks it explicitly.
        self.dimension = int(_setting("DIMENSION", DEFAULT_DIMENSION))
        self.embedding_model = _setting("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.index_name = settings.RAG["PINECONE_INDEX"]
        self.namespace_mode = _setting("NAMESPACE_MODE", MODE_DUAL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(_setting("CHUNK_SIZE", 2000)), chunk_overlap=200
        )

        self.pc = pinecone.Pinecone(api_key=settings.RAG["PINECONE_API_KEY"])
        self.ensure_index()
        # Built ONCE. `_get_index()` used to construct a fresh Pinecone client
        # on every call - on the read path, the write path and the delete path -
        # which meant a new client object and its connection setup for every
        # message the platform handled.
        self.index = self.pc.Index(self.index_name)
        self._legacy_checked = False
        self._legacy_has_vectors = True

    # ------------------------------------------------------------- index

    def ensure_index(self):
        """Create the index if it is missing, and refuse a dimension mismatch."""
        existing = {idx.name for idx in self.pc.list_indexes()}

        if self.index_name not in existing:
            logger.info("creating Pinecone index %s", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                logger.info("waiting for index %s", self.index_name)
                time.sleep(5)
            # A small buffer for DNS propagation.
            time.sleep(10)
            return

        # The index already exists, so its dimension is fixed and ours has to
        # match. Checked here because the alternative is every upsert failing
        # with a message that mentions neither DIMENSION nor which index - and
        # on the indexing path that surfaces as documents stuck in "failed"
        # for a reason nobody can act on.
        described = self.pc.describe_index(self.index_name)
        actual = int(getattr(described, "dimension", 0) or 0)
        if actual and actual != self.dimension:
            raise ValueError(
                f"Pinecone index {self.index_name!r} has dimension {actual}, but "
                f"RAG['DIMENSION'] is {self.dimension}. They must agree - an "
                f"index's dimension cannot be changed after creation, so either "
                f"set DIMENSION={actual} or use a different index."
            )

    def namespace_for(self, organization_id):
        """One organization, one partition."""
        return str(organization_id)

    def _read_legacy(self):
        """Whether the default namespace is still worth querying.

        Checked once per process rather than per query: index stats are a
        network call, and the answer only changes when somebody runs the
        migration. A fresh deployment therefore pays nothing for "dual" mode -
        the legacy namespace is empty, so the second query is never made.
        """
        if self.namespace_mode != MODE_DUAL:
            return False
        if self._legacy_checked:
            return self._legacy_has_vectors

        self._legacy_checked = True
        try:
            stats = self.index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
            legacy = namespaces.get(LEGACY_NAMESPACE)
            count = 0
            if legacy is not None:
                count = int(
                    getattr(legacy, "vector_count", None)
                    or (legacy.get("vector_count", 0) if isinstance(legacy, dict) else 0)
                )
            self._legacy_has_vectors = count > 0
            if self._legacy_has_vectors:
                logger.info(
                    "%s vectors remain in the legacy namespace; retrieval reads "
                    "both. Run migrate_rag_namespaces, then set "
                    "RAG['NAMESPACE_MODE'] = 'namespaced'.",
                    count,
                )
        except Exception:
            # Being wrong the safe way: assume there IS legacy data, so the
            # extra read still happens and nothing silently disappears.
            logger.warning("could not read index stats; assuming legacy vectors exist")
            self._legacy_has_vectors = True
        return self._legacy_has_vectors

    # --------------------------------------------------------- embedding

    def _client(self, api_key):
        return OpenAI(api_key=api_key)

    def get_embedding(self, text, api_key):
        response = self._client(api_key).embeddings.create(
            input=[text.replace("\n", " ")], model=self.embedding_model
        )
        return response.data[0].embedding

    # ----------------------------------------------------------- writing

    def index_chat_message(
        self,
        user_id,
        conversationID,
        message_type,
        message,
        user_openai_key,
        organization_id,
        source="neon",
    ):
        """Index one chat message into its organization's namespace.

        `source` says where the exchange happened - "neon" for the platform's
        own chat, "chatterloop" for a bot conversation mirrored in. Queryable,
        so "what has this bot been answering?" is answerable without joining
        back to the conversation table.
        """
        vector = self.get_embedding(message, user_openai_key)

        self.index.upsert(
            vectors=[
                {
                    "id": f"chat_{uuid.uuid4()}",
                    "values": vector,
                    "metadata": {
                        # `type` distinguishes chat from docs. This used to be
                        # written twice, as `type` AND `msg_type`, with the
                        # same value - so a doc and a chat message disagreed
                        # about which field held the answer, and `retrieve`
                        # had to guess.
                        "type": message_type,
                        "user_id": str(user_id),
                        "conversation_id": str(conversationID),
                        "organization_id": str(organization_id),
                        "source": source,
                        "text": message,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ],
            namespace=self.namespace_for(organization_id),
        )

    def bulk_index_docs(
        self,
        documents,
        user_openai_key,
        organization_id,
        source=None,
        document_id=None,
        shared=True,
    ):
        """Index documents into the organization's namespace.

        Returns the ids of the vectors written. The caller needs them: they are
        minted here and appear nowhere else, so without returning them an
        indexed document can never be deleted or replaced - the vectors would
        sit in the index answering queries with nothing able to name them.
        """
        client = self._client(user_openai_key)
        all_vectors = []

        for doc in documents:
            chunks = self.splitter.split_text(doc)
            if not chunks:
                continue

            response = client.embeddings.create(input=chunks, model=self.embedding_model)

            for i, (chunk, enc) in enumerate(zip(chunks, response.data)):
                all_vectors.append(
                    {
                        "id": f"doc_{uuid.uuid4()}",
                        "values": enc.embedding,
                        "metadata": {
                            "type": "doc",
                            "organization_id": str(organization_id),
                            "text": chunk,
                            "source": source or doc[:100],
                            "chunk_id": i,
                            # Which agents may read this. `shared` carries the
                            # common case in one flag so the query never has to
                            # enumerate the whole corpus; `document_id` is what
                            # a restricted document is named by, and is stable
                            # for the life of the row - reassigning a document
                            # is then a Postgres change and nothing else.
                            "shared": bool(shared),
                            **(
                                {"document_id": str(document_id)}
                                if document_id is not None
                                else {}
                            ),
                        },
                    }
                )

        namespace = self.namespace_for(organization_id)
        for i in range(0, len(all_vectors), UPSERT_BATCH):
            self.index.upsert(
                vectors=all_vectors[i : i + UPSERT_BATCH], namespace=namespace
            )

        return [vector["id"] for vector in all_vectors]

    def stamp_scoping(self, vector_ids, organization_id, document_id, shared):
        """Write `document_id` and `shared` onto vectors that already exist.

        Two callers, one reason each.

        Changing who may read a document is otherwise a re-embed: the answer
        lives in vector metadata, and the only other way to change it is to
        throw the vectors away and pay the provider to make them again. This
        updates the metadata in place, so reassignment costs one pass over the
        document's own chunks and nothing else.

        The second caller is the backfill. Documents indexed before scoping
        existed carry neither field, and retrieval matches the shared corpus on
        `shared: True` - so until they are stamped they are invisible rather
        than shared, which is the safe direction to be wrong in but not one to
        leave in place.

        Per-vector because Pinecone's update takes one id at a time; a document
        is a handful of chunks, and this runs on a worker rather than in a
        request.
        """
        if not vector_ids:
            return 0

        metadata = {"shared": bool(shared)}
        if document_id is not None:
            metadata["document_id"] = str(document_id)

        namespace = self.namespace_for(organization_id)
        stamped = 0
        for vector_id in vector_ids:
            try:
                self.index.update(
                    id=vector_id, set_metadata=metadata, namespace=namespace
                )
                stamped += 1
            except Exception:
                # One chunk failing must not abandon the rest: a half-stamped
                # document is still better scoped than an unstamped one, and
                # the command is safe to run again.
                logger.exception("could not stamp scoping onto vector %s", vector_id)
        return stamped

    def delete_vectors(self, vector_ids, organization_id=None):
        """Remove vectors by id, from the organization's namespace.

        Also from the legacy namespace while anything remains there: a document
        indexed before the migration has its vectors in the default one, and a
        delete that missed them would leave the old version answering queries
        forever. Deleting an id that is not present is a no-op, so trying both
        costs a round trip and never does harm.
        """
        if not vector_ids:
            return

        namespaces = []
        if organization_id is not None:
            namespaces.append(self.namespace_for(organization_id))
        if self._read_legacy() or organization_id is None:
            namespaces.append(LEGACY_NAMESPACE)

        for namespace in namespaces:
            for i in range(0, len(vector_ids), UPSERT_BATCH):
                try:
                    self.index.delete(
                        ids=vector_ids[i : i + UPSERT_BATCH], namespace=namespace
                    )
                except Exception:
                    logger.warning(
                        "could not delete vectors from namespace %r", namespace
                    )

    # ----------------------------------------------------------- reading

    def get_history(self, conversationID, limit):
        """The last `limit` messages, OLDEST FIRST.

        Ordering matters and was previously reversed. The query takes the most
        recent `limit` rows - which needs a descending sort - but the result
        was handed to the model in that order, so it read the conversation
        backwards: the answer before the question, every turn inverted.
        """
        recent = list(
            Message.objects.filter(conversation_id=conversationID)
            .order_by("-created_at")[:limit]
        )
        recent.reverse()
        return [{"msg_type": msg.message_type, "text": msg.content} for msg in recent]

    def retrieve(
        self,
        query,
        conversationID,
        organization_id,
        user_openai_key,
        top_k=5,
        agent=None,
        include_history=True,
    ):
        """Context for one question: recent turns, plus whatever else is relevant.

        `agent` decides which documents are in scope. Omitting it is not a way
        to see everything - it means the caller could not say who is asking,
        and only the shared corpus answers.
        """
        # `include_history=False` is for a caller with a BETTER transcript than
        # this one. The chatterloop bots have it: they read the conversation
        # from chatterloop, reply chain walked, where this reads four rows of
        # Neon's mirror - which holds only what a bot already answered.
        history = self.get_history(conversationID, 4) if include_history else []

        try:
            query_vec = self.get_embedding(query, user_openai_key)
        except Exception:
            # Retrieval failing should cost the answer its memory, not its
            # existence. The caller gets the conversation's recent turns, which
            # is what it would have had before any of this existed.
            logger.exception("could not embed the query; returning history only")
            return history

        matches = self._search(query_vec, conversationID, organization_id, agent)
        if not matches:
            return history

        retrieved = self._rerank(query, matches, top_k)
        return history + self._without_history(retrieved, history)

    def _search(self, query_vec, conversationID, organization_id, agent=None):
        """Query the organization's namespace, and the legacy one if it still has data."""
        from llm.knowledge import document_filter_terms

        namespace = self.namespace_for(organization_id)

        # Inside a namespace every vector already belongs to this organization,
        # so the filter narrows by KIND rather than by tenant: the documents
        # this AGENT may read, plus chat from this conversation only.
        scoped_filter = {"$or": document_filter_terms(agent, conversationID)}

        matches = list(
            self._query(query_vec, namespace=namespace, filters=scoped_filter)
        )

        if self._read_legacy():
            # The old layout: one namespace for everyone, separated by
            # metadata. The organization_id term is what keeps this from
            # reading another tenant's vectors, and is not optional here.
            #
            # Deliberately NOT agent-scoped. These vectors predate assignment
            # and carry neither `shared` nor `document_id`, so scoping them
            # would hide every one of them rather than restrict any of them.
            # They are shared by definition - there was no other kind when they
            # were written - and `migrate_rag_namespaces` is what empties this.
            legacy_filter = {
                "$or": [
                    {"type": "doc", "organization_id": str(organization_id)},
                    {
                        "conversation_id": str(conversationID),
                        "organization_id": str(organization_id),
                    },
                ]
            }
            matches.extend(
                self._query(
                    query_vec, namespace=LEGACY_NAMESPACE, filters=legacy_filter
                )
            )

        return matches

    def _query(self, query_vec, namespace, filters):
        try:
            results = self.index.query(
                vector=query_vec,
                top_k=CANDIDATE_POOL,
                filter=filters,
                include_metadata=True,
                namespace=namespace,
            )
        except Exception:
            logger.exception("Pinecone query failed for namespace %r", namespace)
            return []

        # Pinecone's response type has changed shape across SDK versions -
        # sometimes a dict, sometimes an object with attributes. Read it both
        # ways rather than pinning a version, because the failure if it changes
        # again is silent: no matches, and an answer with no context.
        matches = getattr(results, "matches", None)
        if matches is None and hasattr(results, "get"):
            matches = results.get("matches")
        return list(matches or [])

    def _rerank(self, query, matches, top_k):
        documents = [match["metadata"] for match in matches if match.get("metadata")]
        if not documents:
            return []

        try:
            reranked = self.pc.inference.rerank(
                model=settings.RAG["RERANKER_MODEL"],
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception:
            # Falling back to vector order rather than returning nothing: an
            # unranked answer is worse than a ranked one and much better than
            # none.
            logger.exception("rerank failed; falling back to vector order")
            documents = documents[:top_k]
            return [
                {"msg_type": _kind(doc), "text": doc.get("text") or ""}
                for doc in documents
            ]

        return [
            {"msg_type": _kind(item.document), "text": item.document.get("text") or ""}
            for item in reranked.data
        ]

    @staticmethod
    def _without_history(retrieved, history):
        """Drop retrieved chunks the history window already carries.

        `get_history` pulls the last few turns of THIS conversation, and those
        same messages are indexed - so a query about the current topic reliably
        retrieves them, and the model is shown the same text twice. That wastes
        context and, worse, makes recent turns look corroborated by a second
        source.
        """
        seen = {_normalise(row["text"]) for row in history}
        deduped = []
        for row in retrieved:
            key = _normalise(row["text"])
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped


def _kind(document):
    """What sort of thing a retrieved chunk is.

    Reads `type` first and `msg_type` second: new vectors carry only `type`,
    while ones written before this file stopped duplicating the value carry
    both. Anything else is a document.
    """
    return document.get("type") or document.get("msg_type") or "doc"


def _normalise(text):
    return " ".join((text or "").split()).lower()
