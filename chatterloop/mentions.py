"""@mention parsing.

Ported from the implementations already in the platform:

  * server/reusables/hooks/transformers.js -> extractMentionUsernames()
  * user_service/newsfeed/services/comment_mentions.py
  * rag_service/chatterloop/mentions.py

Those are deliberately identical to each other, and this is the fourth copy. A
mention this file counts that the messenger does not - or the reverse - is a bug
in whichever drifted, so the pattern is reproduced verbatim rather than
"improved".

Note what this is NOT for. The platform resolves handle -> entity server-side
and tells us the answer: a `messages_list` frame carries a non-null `mentioner`
only for recipients who were actually mentioned. That is authoritative and the
runtime trusts it. This parser is for what the event does not answer: which
handle was used, and what the message says once the address is stripped off.
"""

import re

# Identical to MENTION_PATTERN in comment_mentions.py and the inline regex in
# transformers.js.
#
# The leading (?:^|\s) is what stops "you@example.com" from mentioning
# @example. The character class includes "." and is greedy, so "thanks @ana."
# captures "ana." - the lookahead is already satisfied by end-of-string with
# the dot consumed. The other implementations handle that by also emitting the
# dot-stripped form; so does this one.
MENTION_PATTERN = re.compile(r"(?:^|\s)@([A-Za-z0-9._-]{1,30})(?=$|\s|[.,!?;:])")

# Same bound as MAX_MENTIONS_PER_COMMENT. Past this the input is a spam vector.
MAX_MENTIONS = 20


def extract_handles(text):
    """Candidate handles in `text`, lowercased, deduplicated, order preserved.

    A handle ending in "." also yields its stripped form, so "thanks @ana."
    offers both "ana." and "ana". Only one can be a real handle, so resolution
    stays unambiguous.
    """
    if not text:
        return []

    seen = []
    for raw_handle in MENTION_PATTERN.findall(text):
        handle = raw_handle.lower()
        for candidate in (handle, handle.rstrip(".")):
            if candidate and candidate not in seen:
                seen.append(candidate)
        if len(seen) >= MAX_MENTIONS:
            break
    return seen


def normalise_handle(handle):
    """Strip a leading @ and lowercase.

    The platform is inconsistent about which form it hands out - `mentioner.
    username` arrives as "@ana" while `user_account.username` is stored bare -
    so every comparison goes through here rather than guessing at the call site.
    """
    return (handle or "").strip().lstrip("@").lower()


def is_addressed_to(text, handles):
    """Whether any of `handles` is mentioned in `text`.

    A fallback for surfaces where the platform has not already resolved the
    mention. Prefer the server's own signal where it exists.
    """
    if not text or not handles:
        return False
    wanted = {normalise_handle(h) for h in handles}
    return bool(wanted & set(extract_handles(text)))


def strip_mentions(text, handles):
    """Remove the bot's own @handle from the text.

    "@assistant what did we decide about pricing?" is a question about pricing,
    not about the assistant. Leaving the address in drags retrieval toward the
    bot's own name - the one term guaranteed to be irrelevant to the answer.

    Only the addressed handles are removed; mentions of OTHER people are
    content and stay.
    """
    if not text:
        return ""
    wanted = {normalise_handle(h) for h in handles if h}
    if not wanted:
        return text.strip()

    def replace(match):
        handle = match.group(1).lower()
        if handle in wanted or handle.rstrip(".") in wanted:
            # Keep the leading separator so neighbouring words do not fuse.
            return match.group(0)[: match.start(1) - match.start(0) - 1]
        return match.group(0)

    stripped = MENTION_PATTERN.sub(replace, text)
    return re.sub(r"\s{2,}", " ", stripped).strip()


# The command token, as chatterloop's `commandParser.js` matches it.
#
# ANYWHERE A WORD STARTS, like a mention: "@juanlazy /summarize the thread" is
# one thought, and the parser stopped anchoring to the start of the message for
# exactly that case. `(^|\s)` is what keeps a slash inside a word out, so
# "and/or" is untouched.
#
# NOT a second copy of that grammar. The server has already decided whether a
# message IS a command and told us its name; this only removes the token it
# parsed, so the worst a disagreement can do is leave a word in the query
# rather than change which command ran. The escape hatch "//summarize" is a
# literal, not a command, and the negative lookbehind keeps it that way.
COMMAND_PATTERN = re.compile(
    r"(^|\s)/(?<!//)([A-Za-z0-9-]{1,32})(?::([A-Za-z0-9._-]{1,50}))?(?=$|\s)"
)


def strip_command(text):
    """Remove a leading `/command` or `/command:target`.

    The ARGUMENTS are the question. "/summarize the pricing thread" asks about
    the pricing thread, and leaving the token in sends "summarize" to retrieval
    - a term about the instruction rather than about anything being asked for.

    A message that is only the command leaves an empty string, which is a
    legitimate answer: "/members" carries no arguments at all.

    Only the FIRST token goes, matching the parser: one command per message,
    and a second "/summarize" further in is somebody writing about one.
    """
    if not text:
        return ""
    # The captured leading whitespace goes with it, so "@ana /summarize x"
    # leaves "@ana x" rather than "@ana  x".
    return COMMAND_PATTERN.sub("", text, count=1).strip()
