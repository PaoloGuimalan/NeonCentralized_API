import logging

from celery import shared_task

from messenger.models import Message
from organization.credentials import CredentialNotConfigured, embedding_api_key

from ..services.rag import CustomerServiceRAG

logger = logging.getLogger(__name__)

_rag = None


def get_rag():
    """The RAG client, built on first use and then reused by this worker.

    Built lazily rather than at import: CustomerServiceRAG.__init__ calls
    ensure_index(), which hits Pinecone's list_indexes() over the network.
    At module level that ran on IMPORT - and this module is imported by
    messenger.views, so every `manage.py` invocation paid for it, migrations
    included, and a Pinecone outage stopped Django from starting.

    Reuse only helps if the worker process survives more than one task. Django
    settings must keep CELERY_WORKER_MAX_TASKS_PER_CHILD high enough for that;
    at 1 the process is recycled after every task and this caches nothing.
    """
    global _rag
    if _rag is None:
        _rag = CustomerServiceRAG()
    return _rag


@shared_task
def index_chat_message_task(message_id):
    message = Message.objects.select_related(
        "conversation__organization", "sender"
    ).get(message_id=message_id)
    organization = message.conversation.organization

    sender_id = message.sender.id if message.sender else "ai_reply"

    try:
        api_key = embedding_api_key(organization)
    except CredentialNotConfigured as ex:
        # Was `organization.llm_api_key` read directly, which for an
        # organization using Groq for chat was an unusable key - the embedding
        # call then failed inside the task with a provider error that said
        # nothing about the real cause. Now it names it, and stops.
        logger.warning(
            "not indexing message %s: %s", message_id, ex, extra={"org": organization.id}
        )
        return

    # Where the exchange happened. A bot conversation is mirrored into Neon
    # under a `chatterloop:` footprint, and labelling its vectors means "what
    # has this bot been answering?" is answerable from the index rather than by
    # joining back to the conversation table.
    source = (
        "chatterloop"
        if (message.conversation.footprint or "").startswith("chatterloop:")
        else "neon"
    )

    get_rag().index_chat_message(
        sender_id,
        message.conversation.conversation_id,
        message.message_type,
        message.content,
        api_key,
        organization.id,
        source=source,
    )


@shared_task
def index_knowledge_document_task(document_id):
    """Embed and index one uploaded document.

    On Celery rather than in the request because embedding a long document is
    several provider round trips and the upload should not hold a connection
    open for them. The status column on the row is what makes that visible -
    without it, an upload that failed to embed would look identical to one
    that succeeded.
    """
    # Imported here rather than at module level: messenger.views imports this
    # module, and llm.models importing back would be a cycle.
    from llm.models import KnowledgeDocument

    document = KnowledgeDocument.objects.select_related("organization").get(
        pk=document_id
    )

    # Re-indexing replaces rather than adds. Without this, correcting a
    # document would leave the previous version's chunks in the index,
    # answering queries alongside the correction.
    previous = list(document.vector_ids or [])

    document.status = KnowledgeDocument.STATUS_INDEXING
    document.error = ""
    document.save(update_fields=["status", "error", "updated_at"])

    try:
        api_key = embedding_api_key(document.organization)
        rag = get_rag()
        vector_ids = rag.bulk_index_docs(
            [document.content],
            api_key,
            document.organization_id,
            source=document.title,
        )
        if previous:
            rag.delete_vectors(previous, organization_id=document.organization_id)
    except Exception as ex:
        logger.exception("failed to index knowledge document %s", document_id)
        document.status = KnowledgeDocument.STATUS_FAILED
        # The message is shown to the user, who is the only person able to fix
        # the most common cause (a missing or rejected provider key).
        document.error = str(ex)[:1000]
        document.save(update_fields=["status", "error", "updated_at"])
        return

    document.status = KnowledgeDocument.STATUS_INDEXED
    document.vector_ids = vector_ids
    document.chunk_count = len(vector_ids)
    document.error = ""
    document.save(
        update_fields=["status", "vector_ids", "chunk_count", "error", "updated_at"]
    )


@shared_task
def delete_knowledge_vectors_task(vector_ids, organization_id=None):
    """Remove vectors for a document that has already been deleted in Neon.

    Split from the delete request so a Pinecone outage cannot stop a user from
    removing a document. The row goes immediately; the vectors follow. The
    failure mode if this never runs is orphaned vectors, which is recoverable -
    a row that refused to delete is not.

    `organization_id` names the namespace to delete from. Optional so a task
    queued by the previous version still runs after a deploy; without it the
    delete falls back to the legacy namespace only.
    """
    if not vector_ids:
        return
    get_rag().delete_vectors(list(vector_ids), organization_id=organization_id)
