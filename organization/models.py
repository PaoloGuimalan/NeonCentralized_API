from django.db import models
from user.models import Account
from neon.utils.crypto import decrypt, encrypt
from neon.utils.identifiers import new_id
import uuid


class Organization(models.Model):
    id = models.CharField(
        max_length=150, default=new_id, unique=True, blank=True, primary_key=True
    )
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    website = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Account, null=False, on_delete=models.DO_NOTHING)
    updated_at = models.DateTimeField(auto_now=True)
    access_key = models.CharField(max_length=150, default=new_id, unique=True)
    pin = models.CharField(max_length=255, default=new_id, unique=True)
    # Superseded by ProviderCredential. Kept until the data migration has run
    # everywhere, then removable. It stored ONE plaintext key doing two jobs -
    # chat completions and embeddings - which is why an organization using
    # Groq for chat could not use RAG at all: the embedding call needs an
    # OpenAI key and there was nowhere to put a second one.
    llm_api_key = models.TextField(blank=True, default=None, null=True)
    address = models.CharField(max_length=255, blank=True)
    contact_email = models.EmailField(blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Member(models.Model):
    id = models.CharField(
        max_length=150, default=new_id, unique=True, blank=True, primary_key=True
    )
    account = models.ForeignKey(
        Account,
        null=False,
        on_delete=models.DO_NOTHING,
        related_name="user_as_member",
    )
    nickname = models.CharField(max_length=150, null=True, blank=True)
    organization = models.ForeignKey(
        Organization,
        null=False,
        on_delete=models.DO_NOTHING,
    )
    added_by = models.ForeignKey(
        Account,
        null=False,
        on_delete=models.DO_NOTHING,
        related_name="user_as_added_by",
    )
    date_joined = models.DateTimeField(null=True)


class ProviderCredential(models.Model):
    """An organization's API key for one LLM provider.

    REPLACES Organization.llm_api_key, which was a single plaintext field
    doing two different jobs. Splitting it per provider is what lets an
    organization run chat on Groq while still embedding through OpenAI -
    previously impossible, and the reason RAG silently did nothing for a
    Groq-only organization.

    The key is encrypted at rest (neon/utils/crypto.py). It has to be
    RECOVERABLE rather than hashed, because it is presented verbatim to the
    provider on every call - so this protects a leaked database dump, not a
    compromised application, and it is worth being clear-eyed that those are
    different threats.
    """

    id = models.CharField(
        max_length=150, default=new_id, unique=True, blank=True, primary_key=True
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="provider_credentials",
    )
    service = models.ForeignKey(
        "llm.Service",
        on_delete=models.CASCADE,
        related_name="credentials",
    )
    # Never read directly - use the `api_key` property, which decrypts.
    api_key_encrypted = models.TextField(blank=True, default="")

    # How a person tells two keys for the same provider apart. Necessary the
    # moment more than one is allowed: "OpenAI" is no longer an identifier, and
    # the only other thing the API can show is the last four characters, which
    # nobody chose and nobody remembers.
    #
    # NULL rather than blank when unnamed, because the uniqueness constraint
    # below is per (organization, name) and Postgres treats NULLs as DISTINCT
    # while two empty strings collide. A credential created in code without a
    # name should not fail against one created earlier the same way.
    name = models.CharField(max_length=120, null=True, blank=True, default=None)

    # Which key this organization falls back to for this provider when nothing
    # names one specifically. Exactly one per (organization, service).
    #
    # An explicit flag rather than "the oldest" or "the first row": a bot that
    # answers unattended should not change which account it bills to because
    # somebody added a second key.
    is_default = models.BooleanField(default=False)

    # Which credential RAG embeds with. Embeddings are OpenAI-only today, and
    # the provider a chat model happens to use says nothing about which key
    # can embed - so it is an explicit flag rather than something inferred.
    # Separate from `is_default` because they answer different questions: this
    # one is per ORGANIZATION, that one is per provider.
    is_embedding_default = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        constraints = [
            # NOT unique on (organization, service) any more. An organization
            # can hold several keys for one provider - a shared one for most
            # bots, a separate one for a customer-facing bot whose spend has to
            # be attributable, a throwaway for testing - and assign them per
            # bot. What has to stay unambiguous is the FALLBACK, below.
            models.UniqueConstraint(
                fields=["organization", "name"],
                name="unique_credential_name_per_org",
            ),
            # At most one fallback per provider.
            models.UniqueConstraint(
                fields=["organization", "service"],
                condition=models.Q(is_default=True),
                name="one_default_credential_per_org_service",
            ),
            # At most one embedding default per organization, so "which key
            # embeds" can never be ambiguous.
            models.UniqueConstraint(
                fields=["organization"],
                condition=models.Q(is_embedding_default=True),
                name="one_embedding_default_per_org",
            ),
        ]

    @property
    def api_key(self):
        return decrypt(self.api_key_encrypted)

    @api_key.setter
    def api_key(self, value):
        self.api_key_encrypted = encrypt(value)

    @property
    def label(self):
        """What to call this key. Falls back to the provider's own name."""
        return self.name or self.service.name

    def __str__(self):
        return f"{self.organization.name} / {self.label}"
