from django.db import models
from user.models import Account
from llm.models import Agent
from organization.models import Organization
import uuid


class Conversation(models.Model):

    conversation_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, null=False, unique=True
    )
    organization = models.ForeignKey(
        Organization, null=False, on_delete=models.DO_NOTHING
    )
    name = models.TextField(default=None)
    created_by = models.ForeignKey(Account, null=False, on_delete=models.DO_NOTHING)
    created_at = models.DateTimeField(auto_now=True)


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
    sender = models.ForeignKey(Account, on_delete=models.DO_NOTHING)
    agent = models.ForeignKey(Agent, on_delete=models.DO_NOTHING)
    message_type = models.CharField(choices=MESSAGE_TYPE_CHOICES, null=False)
    content = models.TextField(null=False)
    created_at = models.DateTimeField(auto_now=True)
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
    receivers = models.ManyToManyField(Account, related_name="conversation_receivers")
    seeners = models.ManyToManyField(Account, related_name="conversation_seeners")
