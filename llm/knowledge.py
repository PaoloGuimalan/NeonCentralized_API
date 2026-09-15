"""Which knowledge documents an agent is allowed to retrieve.

THE SHAPE OF THE ANSWER, AND WHY IT IS NOT JUST A LIST OF IDS
-------------------------------------------------------------
The obvious implementation is "ask Postgres for every document this agent may
see, send the ids to Pinecone as a filter". It works and it stops working
quietly: a Pinecone query filter has a size limit, a document id is 36
characters, and an organization with a few hundred documents overruns it. The
failure is not an exception - it is a filter that no longer says what you meant.

So the split is by DEFAULT rather than by enumeration. A document with no
agents named on it is shared, and its vectors carry `shared: True`; one match
covers the whole shared corpus however large it grows. Only RESTRICTED
documents are enumerated, and only the ones belonging to the agent asking -
which is bounded by what that agent privately owns, not by the platform.

WHY THIS FAILS CLOSED
---------------------
A document matches on `shared: True` or on being named in this agent's own
list. A restricted document assigned to somebody else matches neither, so the
way this breaks is an agent losing access to something it should have had -
visible immediately, as an agent that stops citing its documents.

The alternative (match everything, exclude what is hidden) fails the other
way: one document whose vectors were never stamped stays readable by every
bot, and nothing anywhere says so until it is quoted to a customer.
"""

import logging

logger = logging.getLogger(__name__)


def restricted_document_ids(agent):
    """Ids of restricted documents this agent may read.

    Shared documents are deliberately absent - they are matched by their
    `shared` metadata instead, which is what keeps this list proportional to
    the agent rather than to the organization.

    `None` means no agent was supplied, which is not the same as an agent with
    nothing assigned: it means the caller could not say who is asking, so only
    shared documents are reachable.
    """
    if agent is None:
        return []

    from .models import KnowledgeDocument

    return list(
        KnowledgeDocument.objects.filter(
            agents=agent, organization_id=agent.organization_id
        )
        .distinct()
        .values_list("id", flat=True)
    )


def document_filter_terms(agent, conversation_id):
    """The `$or` terms a retrieval query is scoped by.

    Built here rather than inline in `rag.py` so the two callers that matter -
    a real query and the test that proves an unassigned agent cannot read
    another agent's document - are looking at the same construction.
    """
    terms = [
        # The shared corpus: one term, regardless of how many documents it is.
        {"type": "doc", "shared": True},
    ]

    allowed = restricted_document_ids(agent)
    if allowed:
        terms.append({"type": "doc", "document_id": {"$in": [str(x) for x in allowed]}})

    # This conversation's own turns. Already scoped as tightly as it gets -
    # a conversation belongs to one organization and one thread.
    terms.append({"conversation_id": str(conversation_id)})
    return terms
