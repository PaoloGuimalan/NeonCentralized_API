"""What the model is shown before it answers.

TWO HISTORIES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
------------------------------------------------------
A flat window of the newest messages answers "what is going on right now". It
is the right thing for a busy channel and it is useless for a reply: somebody
answering a question from forty turns back produces a trigger whose subject is
nowhere in the window, and the model reads a sentence with no referent.

The reply CHAIN answers "what is this about". `/messages/{id}/thread` walks
`replyingTo` upward server-side, so the lineage of that specific thread comes
back however far it starts - reaching past anything a window could carry.

Neither subsumes the other, so both are gathered and merged:

  * recency alone loses the subject of an old reply
  * lineage alone loses everything said since, including a correction, or
    somebody else answering first

They overlap heavily in a quiet conversation, which costs nothing: the merge
deduplicates, so the two collapse to one short list exactly when there is
little to say.

ORDERED AS A TRANSCRIPT
-----------------------
Merged chronologically rather than chain-then-recent. A model handed the same
conversation out of order answers the wrong turn - the bug the `get_history`
ordering fix was about - and an ancestor at its real position in time reads as
the beginning of the thread, which is what it is.
"""

import logging

logger = logging.getLogger(__name__)

# How much of each to send.
#
# The chain gets the larger allowance because it is already narrowed to one
# thread, so every message in it is on topic. The flat window is whatever
# happened to be said, and past a dozen turns it is mostly noise that crowds
# out the part the reply is actually about.
RECENT_LIMIT = 12
CHAIN_LIMIT = 20


def merge(recent, chain, exclude_id=None):
    """Both histories as one transcript: deduplicated, oldest first.

    `exclude_id` drops the triggering message, which is passed to the model
    separately as the question. Leaving it in shows it twice and invites an
    answer to the turn before it.
    """
    combined = {}
    for message in list(chain) + list(recent):
        message_id = str(message.get("message_id") or "")
        if not message_id or message_id == str(exclude_id or ""):
            continue
        combined[message_id] = message

    return sorted(combined.values(), key=lambda m: (m.get("created_at") or 0))


def build(recent, chain, trigger_message_id=None, identity=None,
          recent_limit=RECENT_LIMIT, chain_limit=CHAIN_LIMIT):
    """The turns to show the model, in the shape the LLM services expect.

    Each history is trimmed to its own tail before merging - the newest of the
    flat window, the newest of the lineage. Trimming after the merge would let
    a long chain crowd out recency, or the reverse, depending on which happened
    to be longer.

    `identity` decides which turns were the bot's own. Without it everything
    reads as `user`, which is the safe direction: a model told it said
    something it did not will build on the invention.
    """
    turns = []
    for message in merge(
        list(recent)[-recent_limit:] if recent_limit else [],
        list(chain)[-chain_limit:] if chain_limit else [],
        exclude_id=trigger_message_id,
    ):
        content = (message.get("content") or "").strip()
        if not content:
            # Deleted, or a type with no text. Nothing to show.
            continue
        is_self = bool(identity and identity.is_self(message.get("sender_entity_id")))
        turns.append({"role": "assistant" if is_self else "user", "content": content})
    return turns


# How much of the replied-to message to quote back. Long enough to identify it,
# short enough that quoting a wall of text does not become the prompt.
QUOTE_LIMIT = 400


def parent_of(chain, message_id):
    """The message `message_id` is a reply to, from its own lineage.

    The chain comes back oldest first and ENDS with the anchor, so the parent
    is the element before it. Located by id rather than by taking `chain[-2]`,
    because an assumption about position is the kind that holds until the day
    something is filtered out of the middle.
    """
    ids = [str(m.get("message_id") or "") for m in chain]
    try:
        index = ids.index(str(message_id or ""))
    except ValueError:
        return None
    return chain[index - 1] if index > 0 else None


def question_for(query, chain, message_id, turns=None):
    """The question, with the replied-to message quoted when it is not obvious.

    WHY THIS EXISTS
    ---------------
    Handing the model the lineage is not the same as telling it WHICH message
    is being answered. "Can you explain what you did here?" replying to
    something thirteen turns back arrives as a flat transcript plus the word
    "here", and the model resolves it against the nearest thing - the most
    recent turn, which is not what was pointed at.

    A person replying sees the parent quoted above their own message. This is
    the same thing, said to the model.

    ONLY WHEN IT IS NOT ALREADY OBVIOUS
    -----------------------------------
    Replying to the last thing said is the common case and needs no quote - the
    parent is already the final turn the model read. Quoting it anyway would
    change every prompt to solve a problem that only appears when a reply
    points somewhere else.

    ONCE, ON THE QUESTION
    ---------------------
    Deliberately not a per-turn prefix. Prefixing every turn with "History:"
    taught the model to copy it, and replies went out starting with the word.
    One quote on the final question is a shape it has no reason to imitate.
    """
    parent = parent_of(chain, message_id)
    if parent is None:
        return query

    content = (parent.get("content") or "").strip()
    if not content:
        return query

    if turns and (turns[-1].get("content") or "").strip() == content:
        return query

    handle = (parent.get("sender_handle") or "").strip()
    who = f"@{handle}" if handle else "them"
    if len(content) > QUOTE_LIMIT:
        content = content[:QUOTE_LIMIT].rstrip() + "..."
    return f'Replying to {who}: "{content}"\n\n{query}'
