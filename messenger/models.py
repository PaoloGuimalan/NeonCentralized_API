from django.db import models
from user.models import Account, Token
from llm.models import Agent
from organization.models import Organization
import uuid


class Conversation(models.Model):
    """One thread, wherever it came from.

    WHY `origin` IS A COLUMN AND NOT DERIVED FROM `footprint`
    ---------------------------------------------------------
    The three writers already leave a footprint each side can be told apart
    by - "chatterloop:..." for a mirrored bot exchange, "<org>:<email>:<id>"
    for a third-party app, NULL or whatever the client sent for a native one -
    so a reader COULD sniff the prefix and label the row. Three reasons not to:
    the native case is "none of the above" and so cannot be positively
    identified at all; a client-supplied footprint on the native path is free
    to look like any of them; and a label re-derived per read by string
    matching is a rule living in whichever module last needed it.

    The writers know the answer for certain at the moment they write. This is
    where they record it.
    """

    ORIGIN_NATIVE = "native"
    ORIGIN_EXTERNAL = "external"
    ORIGIN_CHATTERLOOP = "chatterloop"
    ORIGIN_CHOICES = [
        # Neon's own frontend - somebody signed in and started a chat.
        (ORIGIN_NATIVE, "Neon platform"),
        # A third-party app through ExternalChatView, authenticated with a
        # developer token. The person is real, the surface is not Neon's.
        (ORIGIN_EXTERNAL, "External app"),
        # Mirrored from a chatterloop bot answering a mention. Nobody on Neon
        # started it and nobody on Neon is reading it there.
        (ORIGIN_CHATTERLOOP, "Chatterloop bot"),
    ]

    conversation_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, null=False, unique=True
    )
    organization = models.ForeignKey(
        Organization, null=False, on_delete=models.DO_NOTHING
    )
    name = models.TextField(default=None)
    footprint = models.CharField(null=True, blank=True, unique=True, default=None)
    # NULLABLE, and not because a conversation may be unattributed - every
    # writer names somebody. It was relaxed (migration 0014) because the
    # chatterloop mirroring path used to write `bot.created_by` straight
    # through, and that column is SET_NULL, so a deleted minting account
    # killed the exchange on this NOT NULL. `chatterloop/ownership.py` is the
    # rule that replaced it and always resolves to an account.
    #
    # Kept nullable even though migration 0015 backfilled the rows written
    # during that window: a NOT NULL migration over any NULL still left - a
    # conversation whose organization row is gone has nobody to inherit - fails
    # outright, so tightening the column is its own step once 0015 reports
    # nothing remaining.
    created_by = models.ForeignKey(
        Account, null=True, blank=True, on_delete=models.DO_NOTHING
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # Defaulted to native rather than left blank: every row that existed before
    # this column did was classified by migration 0016, and a new row that
    # somehow reaches `.create()` without saying where it came from is far more
    # likely to be Neon's own path than either integration - both of which are
    # single call sites that do say.
    #
    # `db_default` AS WELL AS `default`, AND IT IS NOT REDUNDANT
    # ---------------------------------------------------------
    # `default` is Python-side, so Django names the column in every INSERT it
    # builds - which means an image that predates this field omits it entirely
    # and hits the NOT NULL. Django adds a column's default only to backfill
    # existing rows and then DROPS it, so 0016 on its own left a schema that
    # only the new code could write to, and the database always migrates before
    # the new image is running.
    #
    # The database-level default makes the column additive in the way it was
    # meant to be: old code inserts without it and gets `native`, new code says
    # so explicitly, and the two can be rolled out in either order.
    origin = models.CharField(
        max_length=20,
        choices=ORIGIN_CHOICES,
        default=ORIGIN_NATIVE,
        db_default=ORIGIN_NATIVE,
        db_index=True,
        help_text="Which surface this conversation was started from.",
    )

    @property
    def is_external(self):
        """Whether this thread happened somewhere other than Neon's frontend.

        The distinction the conversation list actually draws; kept here so the
        two integrations can be grouped without every caller enumerating them.
        """
        return self.origin != self.ORIGIN_NATIVE


class Message(models.Model):

    MESSAGE_TYPE_CHOICES = [
        ("ai_reply", "AI Reply"),
        ("text", "Text"),
        ("reply", "Reply"),
    ]

    message_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, null=False, unique=True
    )
    pending_id = models.UUIDField(default=uuid.uuid4, null=False, unique=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.DO_NOTHING)
    sender = models.ForeignKey(
        Account, on_delete=models.DO_NOTHING, null=True, blank=True
    )
    agent = models.ForeignKey(Agent, on_delete=models.DO_NOTHING, null=True, blank=True)
    integration = models.ForeignKey(
        Token,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="messages",
        help_text="The external app/integration this message was sent through, if any. Null means it came from Neon's own native frontend.",
    )
    message_type = models.CharField(choices=MESSAGE_TYPE_CHOICES, null=False)
    content = models.TextField(null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    replying_to = models.ForeignKey(
        "self", on_delete=models.DO_NOTHING, null=True, blank=True
    )
    deleted_by = models.ForeignKey(
        Account,
        related_name="conversation_deleted_by",
        on_delete=models.DO_NOTHING,
        null=True,
        default=None,
        blank=True,
    )
    deleted_at = models.DateTimeField(
        null=True,
        default=None,
        blank=True,
    )
    receivers = models.ManyToManyField(
        Account, related_name="conversation_receivers", blank=True
    )
    seeners = models.ManyToManyField(
        Account, related_name="conversation_seeners", blank=True
    )


class Summary(models.Model):
    summary_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, null=False, unique=True
    )
    conversation = models.ForeignKey(Conversation, on_delete=models.DO_NOTHING)
    context = models.TextField(null=False)
    range = models.IntegerField()
