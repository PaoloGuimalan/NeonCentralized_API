from django.db import models
from user.models import Account
import uuid


class Organization(models.Model):
    id = models.CharField(
        max_length=150, default=uuid.uuid4, unique=True, blank=True, primary_key=True
    )
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    website = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Account, null=False, on_delete=models.DO_NOTHING)
    updated_at = models.DateTimeField(auto_now=True)
    access_key = models.CharField(max_length=150, default=uuid.uuid4, unique=True)
    pin = models.CharField(max_length=255, default=uuid.uuid4, unique=True)
    llm_api_key = models.TextField(blank=True, default=None, null=True)
    address = models.CharField(max_length=255, blank=True)
    contact_email = models.EmailField(blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Member(models.Model):
    id = models.CharField(
        max_length=150, default=uuid.uuid4, unique=True, blank=True, primary_key=True
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
