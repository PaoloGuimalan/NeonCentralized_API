from django.db import models
from user.models import Account
from llm.models import Agent
from organization.models import Organization
import uuid

class Conversation(models.Model):

    conversation_id = models.UUIDField(default=uuid.uuid4, primary_key=True, null=False, unique=True)
    organization = models.ForeignKey(Organization, null=False)
    name = models.TextField(default=None)
    created_by = models.ForeignKey(Account, null=False)

class Message(models.Model):

    MESSAGE_TYPE_CHOICES = [
        ('ai_reply', 'AI Reply'),
        ('reply', 'Reply'),
    ]

    message_id = models.UUIDField(default=uuid.uuid4, primary_key=True, null=False, unique=True)
    pending_id = models.UUIDField(default=uuid.uuid4, primary_key=True, null=False, unique=True)
    conversation = models.ForeignKey(Conversation)
    sender = models.ForeignKey(Account)
    agent = models.ForeignKey(Agent)
    message_type = models.CharField(choices=MESSAGE_TYPE_CHOICES, null=False)
    content = models.TextField(null=False)
    created_at = models.DateTimeField(auto_now=True)
    replying_to = models.ForeignKey('self', on_delete=models.DO_NOTHING, null=True, blank=True)
    deleted_by = models.ManyToManyField(Account, related_name='conversation_deleted')
    deleted_at = models.DateTimeField(auto_now=True)
    receivers = models.ManyToManyField(Account, related_name='conversation_receivers')
    seeners = models.ManyToManyField(Account, related_name='conversation_seeners')

    
