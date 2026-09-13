"""Resolving which API key an organization uses for what.

TWO DIFFERENT KEYS, ONE OLD FIELD
---------------------------------
`Organization.llm_api_key` was a single plaintext column doing two unrelated
jobs: the chat-completion key and the embedding key. Those are not the same
credential and often not even the same provider - embeddings here are OpenAI
only, so an organization running chat on Groq had no way to express an OpenAI
key at all, and RAG silently did nothing for them. Not badly; nothing. No
error, no log line, just an empty context every time.

`ProviderCredential` splits them per provider, and these resolvers are how
callers ask for the one they actually need.

FALLING BACK TO THE OLD FIELD
-----------------------------
Both resolvers fall back to `Organization.llm_api_key` when no credential row
exists, so nothing breaks between deploying this and an organization moving
its key across. The fallback is what makes `llm_api_key` removable later
without a flag day; it is not meant to be permanent.
"""

import logging

from .models import ProviderCredential

logger = logging.getLogger(__name__)

OPENAI = "openai"


class CredentialNotConfigured(Exception):
    """No usable API key for what the caller is trying to do.

    Distinct from a key that exists and is rejected by the provider: this one
    is fixed in Neon's settings, not by the provider, and the message says so.
    """


def _legacy_key(organization):
    return (organization.llm_api_key or "").strip()


def embedding_api_key(organization):
    """The key RAG embeds with.

    Chosen by the explicit `is_embedding_default` flag rather than inferred
    from whichever provider the chat model happens to use - those are
    independent choices, and inferring one from the other is what produced the
    Groq-org-cannot-embed bug in the first place.

    Order: the flagged credential, then an OpenAI credential (embeddings are
    OpenAI-only today, so this is unambiguous when there is exactly one), then
    the legacy field.
    """
    flagged = (
        ProviderCredential.objects.filter(
            organization=organization, is_embedding_default=True, is_active=True
        )
        .select_related("service")
        .first()
    )
    if flagged is not None and flagged.api_key:
        return flagged.api_key

    openai = (
        ProviderCredential.objects.filter(
            organization=organization,
            is_active=True,
            service__name__iexact=OPENAI,
        )
        .select_related("service")
        # Several OpenAI keys are allowed now, so this is no longer "the one" -
        # prefer whichever is the provider default, then whichever exists.
        .order_by("-is_default", "created_at")
        .first()
    )
    if openai is not None and openai.api_key:
        return openai.api_key

    legacy = _legacy_key(organization)
    if legacy:
        return legacy

    raise CredentialNotConfigured(
        "This organization has no embedding credential. Add an OpenAI "
        "provider credential and mark it as the embedding default."
    )


def default_credential(organization, service):
    """The organization's fallback key for one provider, or None.

    The one flagged `is_default`; failing that, the only one there is. That
    second case is what keeps a single-key organization from having to make a
    choice it does not yet have - the moment a second key appears, the flag
    becomes load-bearing and the API requires it.
    """
    if service is None:
        return None

    rows = ProviderCredential.objects.filter(
        organization=organization, service=service, is_active=True
    )
    flagged = rows.filter(is_default=True).first()
    if flagged is not None:
        return flagged

    found = list(rows[:2])
    return found[0] if len(found) == 1 else None


def chat_api_key(organization, service, credential=None):
    """The key for chat completions against `service`.

    `credential` is an explicit assignment - a bot's own key. Honoured only
    when it belongs to this organization AND to this provider: a key assigned
    before somebody changed the bot's model would otherwise be sent to the
    wrong API, which fails as an authentication error and reads as a revoked
    key rather than a mismatch.

    Falling back rather than refusing, because the fallback is what the
    organization already uses everywhere else.
    """
    if credential is not None and credential.is_active and credential.api_key:
        belongs = str(credential.organization_id) == str(organization.pk)
        matches = service is not None and str(credential.service_id) == str(service.pk)
        if belongs and matches:
            return credential.api_key
        logger.warning(
            "ignoring credential %s: it belongs to another organization or "
            "another provider than %s",
            credential.pk,
            service.name if service is not None else "?",
        )

    fallback = default_credential(organization, service)
    if fallback is not None and fallback.api_key:
        return fallback.api_key

    legacy = _legacy_key(organization)
    if legacy:
        return legacy

    name = service.name if service is not None else "that provider"
    raise CredentialNotConfigured(
        f"This organization has no API key configured for {name}."
    )
