"""Talking to developer_service as a bot.

WHAT THIS IS FOR NOW
--------------------
`GET /v1/whoami` and the retry policy around it. The bot runtime adds the rest
(events, messages, comments) on top of `request()`; the retry rule below is the
part that must not be re-decided per call site, so it lives here from the start.

NEVER RETRY AN AUTHENTICATION FAILURE
-------------------------------------
Timeouts and 5xx are retried; 401 and 403 are not, and that is not a
performance nicety. A revoked token retried on a loop is a bot hammering an
endpoint that will never say yes, and the retries bury the one log line that
would explain it. 403 in particular is usually a MISSING GRANT rather than a
bad token, and no amount of retrying creates a permission row.

Ported from `rag_service/chatterloop/platform/client.py`, which learned this.
"""

import json
import logging
import uuid

import httpx
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15
MAX_ATTEMPTS = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}

# The stream's own opening frame, not a platform event.
READY_EVENT = "ready"

# Comfortably longer than the server's heartbeat, so a quiet stream is not
# mistaken for a dead one - but finite, so a connection that has silently gone
# away is eventually noticed and remade.
SSE_READ_TIMEOUT = 90.0


class ChatterloopAPIError(Exception):
    """A call to developer_service failed."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class TokenRejected(ChatterloopAPIError):
    """401 or 403.

    Distinct because the remedy is different from every other failure: a
    retry cannot fix it, and the two causes - a dead token, or a scope with no
    matching grant - are both things a person has to act on.
    """


def _base_url():
    url = getattr(settings, "DEVELOPER_SERVICE_BASE_URL", "") or ""
    if not url:
        raise ChatterloopAPIError(
            "DEVELOPER_SERVICE_BASE_URL is not configured, so Neon cannot "
            "reach Chatterloop's developer API."
        )
    return url.rstrip("/")


def request(method, path, token, *, params=None, json=None, timeout=DEFAULT_TIMEOUT):
    """One call, with the retry policy above."""
    url = f"{_base_url()}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        # Identifiable on purpose: an operator reading developer_service's
        # logs should be able to tell Neon's traffic from anyone else's.
        "User-Agent": "Neon/1.0 (+https://neonsystems.net)",
    }

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.request(
                method, url, headers=headers, params=params, json=json, timeout=timeout
            )
        except requests.RequestException as ex:
            last_error = ChatterloopAPIError(f"Could not reach Chatterloop: {ex}")
            logger.warning("chatterloop %s %s attempt %s failed: %s", method, path, attempt, ex)
            continue

        if response.status_code in (401, 403):
            # Not retried. See the module docstring.
            raise TokenRejected(
                _message_from(response)
                or (
                    "Chatterloop rejected this token."
                    if response.status_code == 401
                    else "This token is missing a scope, or its entity is missing the "
                    "matching permission grant."
                ),
                response.status_code,
            )

        if response.status_code in RETRY_STATUSES:
            last_error = ChatterloopAPIError(
                _message_from(response) or f"Chatterloop returned {response.status_code}.",
                response.status_code,
            )
            logger.warning(
                "chatterloop %s %s attempt %s returned %s",
                method,
                path,
                attempt,
                response.status_code,
            )
            continue

        if not response.ok:
            raise ChatterloopAPIError(
                _message_from(response) or f"Chatterloop returned {response.status_code}.",
                response.status_code,
            )

        try:
            return response.json()
        except ValueError as ex:
            raise ChatterloopAPIError(
                "Chatterloop returned a response that was not JSON."
            ) from ex

    raise last_error or ChatterloopAPIError("Chatterloop could not be reached.")


def _message_from(response):
    try:
        body = response.json()
    except ValueError:
        return ""
    return body.get("message", "") if isinstance(body, dict) else ""


def whoami(token):
    """What developer_service thinks this credential is.

    Worth calling after minting rather than assuming, for one specific reason:
    the handle comes back RESOLVED from the database rather than echoed from
    what Neon sent. Since `bot_bot.handle` is unique only among bots, and
    handles are resolved by UNION across accounts, realms and bots with the
    first match winning, a bot can end up answering to a handle it did not
    expect - and the symptom is that mention matching never fires, with
    nothing in any log to say why.

    Returns the parsed body: entity_id, handle, realm_id, scopes, token{...}.
    """
    return request("GET", "/v1/whoami", token)


# ---------------------------------------------------------------------------
# READS
#
# Every route answers for the TOKEN'S OWN ENTITY. None of them takes an entity
# id, and that is a design rule of the API rather than an omission: a parameter
# that exists is one somebody eventually passes another entity's value to.
# ---------------------------------------------------------------------------


def fetch_messages(token, conversation_id, limit=40):
    """Recent messages, oldest first so the array reads as a transcript.

    Returns `(messages, conversation_type)`. The type comes back on the same
    response because the route resolves it for its own membership check before
    reading a single message - so asking for it separately would be a second
    call for something already computed.
    """
    if not conversation_id:
        return [], ""
    payload = request(
        "GET",
        f"/v1/conversations/{conversation_id}/messages",
        token,
        params={"limit": max(1, min(int(limit), 200))},
    )
    return _messages_from(payload, conversation_id), str(
        payload.get("conversation_type") or ""
    )


def fetch_thread(token, conversation_id, message_id, limit=20):
    """The reply lineage of one message: what it answers, and what that answered.

    Returns `(messages, truncated)`. Oldest first, ending with the message
    itself - the server walks `replyingTo` upward, which a client cannot: the
    parent of a reply is regularly older than any window worth fetching, and
    there is no route that reads a message by id.

    `truncated` is why this returns a pair. It says the walk stopped at the
    limit or at a broken link rather than at the start of the thread, and a
    partial lineage read as a whole one is exactly the case where an answer
    confidently misses the point.
    """
    if not conversation_id or not message_id:
        return [], False
    payload = request(
        "GET",
        f"/v1/conversations/{conversation_id}/messages/{message_id}/thread",
        token,
        params={"limit": max(1, min(int(limit), 50))},
    )
    return (
        _messages_from(payload, conversation_id),
        bool(payload.get("truncated")),
    )


def fetch_moderation(token, *, message_ids=None, post_id="", comment_ids=None):
    """What the platform already knows about the media in these things.

    A message carrying an upload stores the CDN URL in `content`, so a model
    shown `content` is shown a URL - and a model shown a URL answers the URL.
    The moderation pipeline has already transcribed, captioned and read the
    text out of that same file; this is how the bot gets at it.

    One kind per call, matching the three routes. They are separate because
    each is gated on the scope its underlying read needs - a message is
    `messages.read`, a post or comment is `notifications.read` - and a bot
    holding only one of those should still get the half it can have.

    Returns rows as they arrived, normalised. `status` is on every row and is
    the field that matters most: analysis is asynchronous, so a message seconds
    old comes back `pending` with nothing else, and that is worth saying to the
    model rather than hiding.
    """
    if message_ids:
        ids = [str(value) for value in message_ids if value]
        if not ids:
            return []
        payload = request(
            "GET", "/v1/moderation/messages", token,
            params={"messageID": ids[:MODERATION_ID_CAP]},
        )
    elif post_id:
        payload = request("GET", f"/v1/moderation/posts/{post_id}", token)
    elif comment_ids:
        ids = [str(value) for value in comment_ids if value]
        if not ids:
            return []
        payload = request(
            "GET", "/v1/moderation/comments", token,
            params={"commentID": ids[:MODERATION_ID_CAP]},
        )
    else:
        return []

    return _moderation_from(payload)


# The route's own cap. Asking for more is not an error there - the extra ids
# are dropped - but sending them is a request nobody can answer, and a caller
# that trims knows which ids it gave up on.
MODERATION_ID_CAP = 50


def _moderation_from(payload):
    """Rows into dicts, skipping anything malformed.

    Same rule as `_messages_from`: one unrecognised row costs one row, never
    the batch. This is another service's data and a shape change must degrade
    rather than stop the bot - and here it must degrade to "no context", which
    is exactly where the bot was before this existed.
    """
    records = []
    for row in payload.get("moderation") or []:
        if not isinstance(row, dict):
            continue
        target_id = row.get("target_id")
        if not target_id:
            continue
        records.append(
            {
                "target_id": str(target_id),
                "source_type": str(row.get("source_type") or ""),
                "content_type": str(row.get("content_type") or ""),
                # Absent unless the document said so - `None` means nothing
                # classified the audio, which is not the same as "not music".
                "is_music": row.get("is_music"),
                # Defaulted to "pending" for the same reason the route does:
                # something is there and nothing has said it finished.
                "status": str(row.get("status") or "pending"),
                "text": str(row.get("text") or ""),
                "transcription": str(row.get("transcription") or ""),
                "caption": str(row.get("caption") or ""),
                "shown_text": str(row.get("shown_text") or ""),
                "language": str(row.get("language") or ""),
            }
        )
    return records


def conversation_type(token, conversation_id):
    """Whether this is a DM: "single", something else, or "" if unresolvable.

    `limit=1` because this wants one field, not a history window. The reply
    probe deliberately does NOT resolve this - on that path it answers nothing.
    """
    if not conversation_id:
        return ""
    payload = request(
        "GET",
        f"/v1/conversations/{conversation_id}/messages",
        token,
        params={"limit": 1},
    )
    return str(payload.get("conversation_type") or "")


def fetch_replies_to_me(token, conversation_id, limit=25):
    """Messages here that reply to one the bot wrote. Usually empty.

    WHOSE replies is not a parameter - the route answers for the token's own
    entity. That is what keeps "reply to me without naming me" from widening
    into "reply to anyone", and it is enforced somewhere Neon cannot bypass.
    """
    if not conversation_id:
        return []
    payload = request(
        "GET",
        f"/v1/conversations/{conversation_id}/replies",
        token,
        params={"limit": max(1, min(int(limit), 100))},
    )
    return _messages_from(payload, conversation_id)


def fetch_comment_mentions(token, limit=25):
    """Unread comment mentions, with the comment text already resolved."""
    payload = request(
        "GET",
        "/v1/mentions/comments",
        token,
        params={"limit": max(1, min(int(limit), 100))},
    )
    return _mentions_from(payload, "mentions", "mention")


def fetch_comment_replies(token, limit=25):
    """Unread notifications saying somebody replied to a comment the bot wrote.

    A separate route rather than a flag: Django files "replied to your comment"
    and "commented on your post" under one notification type, and only the
    first is an answer to something the bot said. The route separates them
    structurally, on whether the referenced comment has a parent.
    """
    payload = request(
        "GET",
        "/v1/comments/replies",
        token,
        params={"limit": max(1, min(int(limit), 100))},
    )
    return _mentions_from(payload, "replies", "reply")


def _messages_from(payload, conversation_id):
    """Rows into dicts, skipping anything malformed.

    One unrecognised row costs one skipped message, never the whole batch -
    this is another service's data, and a shape change must degrade rather
    than stop the bot.
    """
    messages = []
    for row in payload.get("messages") or []:
        if not isinstance(row, dict):
            continue
        message_id = row.get("message_id")
        sender = row.get("sender_entity_id")
        content = row.get("content")
        if not message_id or not sender or not isinstance(content, str):
            continue
        messages.append(
            {
                "message_id": str(message_id),
                "conversation_id": str(row.get("conversation_id") or conversation_id),
                "sender_entity_id": str(sender),
                "sender_handle": str(row.get("sender_handle") or ""),
                "content": content,
                # Epoch MILLISECONDS, normalised server-side. 0 means
                # unparseable, and sorts to the start of history so a parsing
                # gap can never make an old message look like the newest.
                "created_at": _as_int(row.get("created_at")),
                "message_type": str(row.get("message_type") or "text"),
                "is_reply": bool(row.get("is_reply")),
                "replying_to": str(row.get("replying_to") or ""),
                "reply_target": _reply_target_from(row.get("reply_target")),
            }
        )
    return messages


def _reply_target_from(value):
    """`{"type", "id"}` for what a message replies to, or None.

    The only field that says a note is about a POST (or a Moment or Thought):
    `replying_to` is a message id, so for those it is empty. Without this, a
    post sent with a note reached the bot as a note about nothing.
    """
    if not isinstance(value, dict):
        return None
    kind = str(value.get("type") or "").strip().lower()
    target_id = str(value.get("id") or "").strip()
    if not kind or not target_id:
        return None
    return {"type": kind, "id": target_id}


def _mentions_from(payload, key, kind):
    pending = []
    for row in payload.get(key) or []:
        if not isinstance(row, dict):
            continue
        comment_id = row.get("comment_id")
        text = row.get("text")
        if not comment_id or not isinstance(text, str) or not text.strip():
            # The route already drops textless rows; this is the belt to that
            # braces, because an unanswerable trigger reaching the policy is
            # worse than one dropped twice.
            continue
        pending.append(
            {
                "comment_id": str(comment_id),
                "post_id": str(row.get("post_id") or ""),
                "author_entity_id": str(row.get("author_entity_id") or ""),
                "author_handle": str(row.get("author_handle") or ""),
                "text": text,
                "created_at": _as_int(row.get("created_at")),
                "kind": str(row.get("kind") or kind),
            }
        )
    return pending


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# WRITES
# ---------------------------------------------------------------------------


def send_message(token, conversation_id, content, reply_to_message_id=""):
    """Post a message as the bot.

    `receivers` is NEVER sent. The route derives it from the conversation,
    which is what stops a token addressing people who are not in it - so
    supplying it is not merely unnecessary, it is the thing the design forbids.

    `conversationType` is absent for a related reason. The API treats it as a
    fallback for a genuinely new non-realm conversation, and an earlier version
    of the reference defaulting it to "single" rewrote a real group's type so
    the UI rendered it as a DM. The bot only ever answers in conversations that
    already exist, so it has nothing to contribute here.
    """
    if not content or not content.strip():
        raise ChatterloopAPIError("Refusing to send an empty message.")

    body = {
        "conversationID": str(conversation_id),
        "content": content.strip(),
        # The optimistic-update key. Echoed back so a sender can reconcile;
        # generated here because nothing upstream has one.
        "pendingID": f"neon-{uuid.uuid4().hex}",
        "messageType": "text",
    }
    # Threading keeps the answer attached to the question. In a busy channel an
    # unthreaded reply arrives detached from what it answers, and by the time
    # the bot has fetched, retrieved and generated, several other messages may
    # have landed in between.
    if reply_to_message_id:
        body["replyingTo"] = str(reply_to_message_id)

    return request("POST", "/v1/messages/send", token, json=body)


def post_comment(token, post_id, text, parent_comment_id=""):
    """Post a comment as the bot.

    `parentID` is what the bot is ANSWERING. The route re-parents a
    reply-to-a-reply onto its top-level ancestor, because threads flatten to
    two levels - and it returns both ids so the flattening is visible rather
    than discovered later from a thread that reads oddly. Neon does not compute
    the parent itself; that would reimplement the rule the route exists to own.
    """
    if not text or not text.strip():
        raise ChatterloopAPIError("Refusing to post an empty comment.")

    body = {"postID": str(post_id), "text": text.strip()}
    if parent_comment_id:
        body["parentID"] = str(parent_comment_id)

    return request("POST", "/v1/comments", token, json=body)


# ---------------------------------------------------------------------------
# EVENTS
# ---------------------------------------------------------------------------


async def stream_events(token, on_envelope, should_continue=None, timeout=None):
    """Consume `GET /v1/events` until the connection ends.

    Async because the supervisor holds one of these per leased bot. An idle SSE
    connection costs a coroutine rather than a thread stack, which is what lets
    hundreds of bots fit in one small container - the whole reason the runtime
    is split between a supervisor and Celery workers.

    Returns normally when the stream ends, which is EXPECTED: the server caps a
    single stream's lifetime (an hour by default), so a clean disconnect is
    routine and the caller reconnects. Raises `TokenRejected` for 401/403,
    which the caller must not retry - no amount of reconnecting creates a
    permission grant.
    """
    url = f"{_base_url()}/v1/events"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
        # Any proxy that buffers would hold frames until its buffer filled,
        # which on a quiet stream means a mention arriving minutes late.
        "Cache-Control": "no-cache",
        "User-Agent": "Neon/1.0 (+https://neonsystems.net)",
    }
    read_timeout = timeout if timeout is not None else SSE_READ_TIMEOUT

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(read_timeout, connect=15.0)
    ) as client:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code in (401, 403):
                await response.aread()
                raise TokenRejected(
                    "Chatterloop rejected this bot's token, or it is missing "
                    "the events.subscribe scope or its matching grant.",
                    response.status_code,
                )
            if response.status_code >= 400:
                await response.aread()
                raise ChatterloopAPIError(
                    f"Event stream returned {response.status_code}.",
                    response.status_code,
                )

            event_name = "message"
            async for raw_line in response.aiter_lines():
                if should_continue is not None and not should_continue():
                    return
                line = raw_line.rstrip("\r\n")

                if line.startswith(":"):
                    # The server's keepalive comment. Proof of life, no payload.
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    name, event_name = event_name, "message"
                    if name == READY_EVENT:
                        continue
                    envelope = decode_frame(payload)
                    if envelope is not None:
                        on_envelope(envelope)
                    continue
                # A blank line terminates an event; anything else (id:, retry:)
                # is valid SSE this consumer has no use for.


def decode_frame(data):
    if not data:
        return None
    try:
        parsed = json.loads(data)
    except ValueError as ex:
        # Another service's firehose. One bad frame is not Neon's problem to
        # solve, and must not interrupt the ones after it.
        logger.warning("undecodable frame on event stream: %s", ex)
        return None
    return parsed if isinstance(parsed, dict) else None
