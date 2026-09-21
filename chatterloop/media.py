"""What a piece of media IS, in words the model can read.

THE PROBLEM THIS SOLVES
-----------------------
A message carrying an upload stores the CDN URL in `content`. `context.build`
takes `content` verbatim, so the turn handed to the model is literally

    https://cdn.chatterloop.app/.../clip.mp4

and a model shown a URL answers the URL - or invents what is in it. Meanwhile
moderation_service has already transcribed, captioned and read the text out of
that exact file. This module turns that record into the line that replaces the
URL.

PROSE, NOT A FIELD DUMP
-----------------------
The result is a message, so it has to read like one. A model shown JSON starts
answering in JSON, and a model shown `caption=... transcription=...` starts
writing field names into its replies. What goes in is a sentence with a bracket
in front of it saying what kind of thing it describes.

EVERYTHING HERE IS PURE
-----------------------
No I/O, no Django, no token. The fetch lives in `client.fetch_moderation` and
the wiring in `tasks`, so every rule about what the model ends up reading is
testable without a network or a database - which is the difference between a
rule that is checked and one that is hoped for.
"""

# How much of each field survives into a turn.
#
# A four-minute video's transcript is longer than the conversation it is in,
# and a turn that is mostly one attachment crowds out the thread the bot is
# supposed to be answering. Generous enough to carry what was said, short
# enough that it stays context rather than becoming the prompt.
TRANSCRIPT_LIMIT = 600
CAPTION_LIMIT = 300

# What each status is worth saying out loud.
#
# Analysis is asynchronous: a photo sent seconds ago is usually `pending`, and
# the bot answers before it finishes. Saying so lets the model tell somebody it
# has not read the attachment yet - which is the honest answer, and better than
# both ignoring the attachment and guessing at it.
STATUS_LINES = {
    "pending": "Still being processed.",
    "processing": "Still being processed.",
    "failed": "Could not be read.",
    "skipped": "Not analysed.",
}


def needs_context(messages):
    """The ids of messages whose content is not text, in order, deduplicated.

    Keyed on `message_type`, the same field the publisher keys on when it
    decides what to analyse - so what this asks about and what was analysed are
    decided by one value rather than by two rules that can disagree.

    "notif" is excluded: those are strings this platform wrote, not anything
    somebody uploaded.
    """
    ids = []
    seen = set()
    for message in messages or []:
        if not _is_media(message):
            continue
        message_id = str(message.get("message_id") or "")
        if message_id and message_id not in seen:
            seen.add(message_id)
            ids.append(message_id)
    return ids


def apply(messages, records):
    """Replace every media message's URL with what the media actually is.

    IN PLACE. `recent` and `chain` are freshly fetched lists that nothing else
    holds, and copying them to rewrite one field per row would be a copy for
    its own sake.

    A media message with NO record still gets its URL replaced - with the bare
    kind, `[image]`. Nothing has analysed it, or nothing ever will, and either
    way the model learns more from "there is an image here" than from a URL it
    cannot open.
    """
    by_target = {}
    for record in records or []:
        # `client._moderation_from` already drops a row that is not a dict.
        # Checked again here because this function is also called with records
        # from a test, a fixture or a future caller, and one bad row must cost
        # one row rather than every attachment in the window.
        if not isinstance(record, dict):
            continue
        target_id = str(record.get("target_id") or "")
        if target_id:
            by_target.setdefault(target_id, []).append(record)

    for message in messages or []:
        if not _is_media(message):
            continue
        rows = by_target.get(str(message.get("message_id") or ""))
        line = summarise(rows) if rows else f"[{_label_from_type(message)}]"
        if line:
            message["content"] = line


def summarise(records):
    """One line per record, in the order they arrived.

    A post is the case with several: its caption is one document and each
    attachment is another, and a reader wants all of them.
    """
    lines = [describe(record) for record in records or []]
    return "\n".join(line for line in lines if line)


def describe(record):
    """One record as a line of prose.

    Never empty for a record that exists: a document with nothing in it still
    renders as `[image]`, because "there is an image here and nothing is known
    about it" is information and silence is not.
    """
    if not isinstance(record, dict):
        return ""

    label = _label(record)
    status = str(record.get("status") or "pending").lower()
    if status != "done":
        return f"[{label}] {STATUS_LINES.get(status, 'Not available.')}"

    parts = []

    caption = _clip(record.get("caption"), CAPTION_LIMIT)
    if caption:
        parts.append(_sentence(caption))

    transcription = _clip(record.get("transcription"), TRANSCRIPT_LIMIT)
    if transcription:
        # Quoted, because it is somebody's words rather than a description of
        # them - and a model that cannot tell the two apart will paraphrase a
        # quote as if it were the platform's own summary.
        parts.append(f'Says: "{transcription}"')

    shown_text = _clip(record.get("shown_text"), CAPTION_LIMIT)
    if shown_text:
        parts.append(f'Text shown: "{shown_text}"')

    if not parts:
        # A text unit - a post's own caption, say - arrives in `text` rather
        # than in any of the above.
        authored = _clip(record.get("text"), CAPTION_LIMIT)
        if authored:
            parts.append(_sentence(authored))

    return f"[{label}] " + " ".join(parts) if parts else f"[{label}]"


def _is_media(message):
    message_type = str(message.get("message_type") or "text").lower()
    return message_type not in ("", "text", "notif")


def _label(record):
    """What to call this in the bracket.

    Audio splits on `is_music`, because "voice note" and "audio" set completely
    different expectations about whether there is anything to reply to. `None`
    means nothing classified it, so it stays the neutral word rather than being
    guessed into one of the two.
    """
    content_type = str(record.get("content_type") or "").lower()
    if content_type == "audio":
        is_music = record.get("is_music")
        if is_music is True:
            return "audio, music"
        if is_music is False:
            return "voice note"
        return "audio"
    return content_type or "attachment"


def _label_from_type(message):
    """The same bracket, derived from the mime when there is no record.

    `messageType` mixes bare kinds ("image") with real mimes ("video/mp4"),
    which is why this takes the part before the slash.
    """
    message_type = str(message.get("message_type") or "").lower()
    top = message_type.split("/")[0].strip()
    return top or "attachment"


def _clip(value, limit):
    """A field, trimmed and bounded, or "".

    Cut at a word boundary where there is one: a transcript ending mid-word
    reads as corruption, and a model will try to complete it.
    """
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    spaced = cut.rsplit(" ", 1)[0]
    return (spaced or cut).rstrip(",;:") + "..."


def _sentence(text):
    """Punctuated, so two parts do not run into each other as one clause."""
    return text if text[-1:] in ".!?\"'" else text + "."
