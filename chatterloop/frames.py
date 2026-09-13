"""The platform's realtime frames, as they arrive on `GET /v1/events`.

THE DOUBLE NESTING IS REAL
--------------------------
The envelope published by `publish(channel, event, message)` in
server/reusables/redis/pubsub.js is:

    {"logType": null, "pod": "...", "event": "<name>",
     "message": <frame>, "dateTime": "..."}

and developer_service forwards it verbatim. The frame inside has a `message`
field of its OWN. So `envelope["message"]["message"]` is the payload, which is
confusing but is what is on the wire.

Plain dicts and dataclasses rather than pydantic - Neon does not depend on it,
and the parsing here is shallow enough that a schema library would be the
larger of the two.

Only the frames the runtime acts on are modelled. Everything else is consumed
and dropped, but IGNORED_EVENTS enumerates the known ones so that "unhandled
event" stays a real signal rather than routine call-signalling noise.
"""

from dataclasses import dataclass

# Events the runtime reacts to.
EVENT_MESSAGES_LIST = "messages_list"
EVENT_NOTIFICATIONS = "notifications"
EVENT_NOTIFICATIONS_RELOAD = "notifications_reload"

HANDLED_EVENTS = frozenset(
    {EVENT_MESSAGES_LIST, EVENT_NOTIFICATIONS, EVENT_NOTIFICATIONS_RELOAD}
)

# Consumed and dropped without comment. Enumerated from the webapp's sse.ts so
# that an event the platform adds later shows up in the logs as genuinely
# unknown, rather than being lost among the call-signalling traffic.
IGNORED_EVENTS = frozenset(
    {
        "coordinates_broadcast",
        "profile_relationship_updated",
        "istyping_broadcast",
        "incomingcall",
        "callreject",
        "contactslist",
        "active_users",
        "voice-joined",
        "join-room-response",
        "create-transport-response",
        "transport-connect-response",
        "produce-response",
        "new_producer",
        "participant-joined",
        "participant-left",
        "update_participants",
        "participant-status",
        "producer-closed",
        "consume-response",
        "consume-transport-error",
        "consume-error",
        "conference_requests_changed",
        "conference_members_changed",
        "conference_access_changed",
        "server_channels_changed",
        "realm_membership_changed",
        "removed_user_notif",
    }
)


@dataclass(slots=True)
class Mentioner:
    """Who mentioned us.

    Present on a `messages_list` frame only when THIS recipient was mentioned:
    the server resolves handles against the conversation's member list and
    passes `isMentioned ? mentioner : null` per receiver. That per-recipient
    resolution is why the runtime can trust this instead of re-parsing.
    """

    entity_id: str = ""
    # Arrives with the "@" already on it.
    username: str = ""
    realm_name: str = ""
    is_single: bool = False


@dataclass(slots=True)
class MessagesListPayload:
    """Body of a `messages_list` frame.

    Note what is absent: the message text, and the message id. This event is a
    ping telling a client to refetch, not a carrier of content - which is why
    answering one always costs a read.
    """

    conversation_id: str = ""
    # The ACTING entity that sent the message. Compared against the bot's own
    # entity id for loop prevention.
    entity_id: str = ""
    mentioner: Mentioner | None = None
    # Present instead of a normal delivery when a message was deleted.
    deleted_message_id: str = ""


@dataclass(slots=True)
class Envelope:
    event: str = ""
    date_time: str = ""
    status: bool = False
    auth: bool = False
    # A dict for messages_list, a plain string for notifications. The platform
    # overloads this field, so it stays loose and is narrowed per event type.
    body: object = None


def parse_envelope(raw):
    """Read one envelope, or None if it is not one.

    Tolerant by construction: this is another service's firehose, and one
    malformed frame must not stop the runtime reading the next.
    """
    if not isinstance(raw, dict):
        return None

    frame = raw.get("message")
    if not isinstance(frame, dict):
        frame = {}

    return Envelope(
        event=str(raw.get("event") or ""),
        date_time=str(raw.get("dateTime") or ""),
        status=bool(frame.get("status")),
        auth=bool(frame.get("auth")),
        body=frame.get("message"),
    )


def parse_messages_list(envelope):
    """Narrow a `messages_list` frame body, or None."""
    body = envelope.body
    if not isinstance(body, dict):
        return None

    raw_mentioner = body.get("mentioner")
    mentioner = None
    if isinstance(raw_mentioner, dict):
        mentioner = Mentioner(
            entity_id=str(raw_mentioner.get("entityID") or ""),
            username=str(raw_mentioner.get("username") or ""),
            realm_name=str(raw_mentioner.get("realmName") or ""),
            is_single=bool(raw_mentioner.get("isSingle")),
        )

    return MessagesListPayload(
        conversation_id=str(body.get("conversationID") or ""),
        entity_id=str(body.get("entityID") or ""),
        mentioner=mentioner,
        deleted_message_id=str(body.get("deletedMessageID") or ""),
    )
