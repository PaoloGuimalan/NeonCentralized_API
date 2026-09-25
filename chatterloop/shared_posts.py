"""What a post sent into a chat IS, in words the model can read.

THE PROBLEM THIS SOLVES
-----------------------
Somebody sends a bot a post. It arrives in one of two shapes:

    sent alone         message_type "post", content = the post's id
    sent with a note   a text message whose reply_target is
                       {"type": "post", "id": <the post's id>}

Sent alone, what the model read was a thirty-digit number - and since that
message is the one addressed to the bot, the number was the QUESTION. Sent
with a note, it read "what do you think of this?" about something it could not
see. moderation_service has already read the post's caption and what is in its
photos and videos; `GET /v1/moderation/posts/{id}` hands that over, and this
module turns it into what the model reads instead - the same prose `media`
writes for an upload, which is also what a comment's post context uses.

A reply to a Moment or a Thought arrives in the second shape, with reply_target
type "moment" / "thought". Both are posts underneath and the same route answers
for them, so they are handled the same way.

Messages only. A comment's post is read by `tasks._post_context`, which already
does this for comments and is deliberately left alone.

Pure, like `media`: no I/O, no Django, no token. The fetch is
`client.fetch_moderation` and the wiring is in `tasks`.
"""

from . import media

# What each reply_target type is called in the bracket. A post sent alone is
# always "post": the message does not say which kind it was, and finding out
# would be another read for one word.
KINDS = {"post": "post", "moment": "Moment", "thought": "Thought"}

# Every post is its own call - the route takes one id - so a window full of
# shared posts is bounded. The triggering message's post always comes first;
# the rest are the newest, which is what the conversation is about.
MAX_POSTS = 3

# The route said the post is gone (deleted), as opposed to "not read".
GONE = object()

# Set on a message once it has been rewritten, so the same dict passed twice
# is not rewritten twice - the second pass would read the bracket as the id.
_REWRITTEN = "_post_context"


def post_of(message):
    """`(kind, post_id)` for the post a message carries, or None."""
    if not isinstance(message, dict) or message.get(_REWRITTEN):
        return None
    if _sent_alone(message):
        post_id = str(message.get("content") or "").strip()
        return ("post", post_id) if post_id else None
    target = message.get("reply_target")
    if isinstance(target, dict):
        kind = str(target.get("type") or "").strip().lower()
        post_id = str(target.get("id") or "").strip()
        if kind in KINDS and post_id:
            return (kind, post_id)
    return None


def wanted(messages, first=None, limit=MAX_POSTS):
    """The post ids worth reading: the triggering message's first, then newest.

    Deduplicated, because the recent window and the reply chain overlap and the
    same shared post arrives in both.
    """
    rows = [message for message in messages or [] if isinstance(message, dict)]
    rows.sort(key=lambda message: message.get("created_at") or 0, reverse=True)
    if first:
        # Stable, so everything else keeps its newest-first order.
        rows.sort(key=lambda message: str(message.get("message_id") or "") != str(first))

    ids = []
    for message in rows:
        found = post_of(message)
        if found and found[1] not in ids:
            ids.append(found[1])
            if len(ids) >= limit:
                break
    return ids


def header(kind, known, sent_alone):
    """The bracket saying there is a post here, and what is in it.

    `known` is the post's moderation records, GONE, or None when it was not
    read (no access, a failed read, or past MAX_POSTS). Not read still says a
    post is there: the model learns more from that than from a bare id or a
    note about nothing.
    """
    verb = "Shared" if sent_alone else "About"
    label = KINDS.get(kind, "post")
    if known is GONE:
        return f"[{verb} a {label} that is no longer available]"
    summary = media.summarise(known) if isinstance(known, list) else ""
    bracket = f"[{verb} a {label}]"
    return f"{bracket}\n{summary}" if summary else bracket


def apply(messages, known):
    """Rewrite every message carrying a post with what the post is. IN PLACE.

    Sent alone, the id is replaced by the post. With a note, the post goes
    above the note, the way the app shows it.

    Returns `{message_id: (header, sent_alone)}` for what was rewritten, so the
    caller can put the post into the QUESTION too: the triggering message is
    left out of the turns (`context.build`), so rewriting it alone would never
    reach the model.
    """
    known = known or {}
    rewritten = {}
    for message in messages or []:
        found = post_of(message)
        if found is None:
            continue
        kind, post_id = found
        alone = _sent_alone(message)
        head = header(kind, known.get(post_id), alone)
        note = "" if alone else str(message.get("content") or "").strip()
        message["content"] = f"{head}\n\n{note}" if note else head
        message[_REWRITTEN] = True
        message_id = str(message.get("message_id") or "")
        if message_id:
            rewritten[message_id] = (head, alone)
    return rewritten


def ask(query, carried):
    """The question for a triggering message that carried a post.

    Sent alone, the post IS the question - the query was its id. With a note,
    the note is the question and the post is what it is about.
    """
    head, alone = carried
    query = (query or "").strip()
    if alone or not query:
        return head
    return f"{head}\n\n{query}"


def _sent_alone(message):
    return str(message.get("message_type") or "").strip().lower() == "post"
