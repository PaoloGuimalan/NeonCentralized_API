from django.db import models
from user.models import Account
from organization.models import Organization

from neon.utils.identifiers import new_id
import uuid


class Tool(models.Model):
    """
    Represents a callable function/tool agents can use.
    """

    PARAM_TYPE_CHOICES = [
        ("query", "Query"),
        ("route", "Route"),
        ("body", "Body"),
    ]

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="tools",
    )
    # Unique PER ORGANIZATION, not globally. It was globally unique, which
    # meant the first organization to create a tool called "search" took that
    # name away from every other organization on the platform - a tenancy bug
    # that only shows up once there is a second customer.
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    parameters_schema = models.JSONField(
        blank=True, null=True
    )  # JSON schema for parameters
    headers_schema = models.JSONField(
        blank=True, null=True
    )  # JSON schema for parameters
    api_endpoint = models.URLField(
        help_text="Tool callable API endpoint URL", blank=True, null=True
    )
    http_method = models.CharField(
        max_length=10, choices=[("GET", "GET"), ("POST", "POST")], default="POST"
    )
    param_type = models.CharField(
        choices=PARAM_TYPE_CHOICES, null=False, default="query"
    )
    requires_auth = models.BooleanField(default=False)
    is_enabled = models.BooleanField(default=False)
    # Read only by llm/utils/tool_execution.py, never serialized. See
    # ToolSerializer for what happened when it was.
    authentication = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name"], name="unique_tool_name_per_org"
            ),
        ]

    def __str__(self):
        return self.name


class Role(models.Model):
    """
    Defines dynamic roles with system prompts and assigned tools.
    """

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="roles",
    )
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    system_prompt = models.TextField(
        help_text="System prompt or personality template for this role"
    )
    tools = models.ManyToManyField(Tool, blank=True, related_name="roles")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name"], name="unique_role_name_per_org"
            ),
        ]

    def __str__(self):
        return self.name


class Agent(models.Model):
    """
    Represents an agent belonging to an organization with a dynamic role.
    """

    uuid = models.CharField(default=new_id, null=False, unique=True)
    name = models.CharField(max_length=255)
    slug = models.SlugField()
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="agents"
    )
    role = models.ForeignKey(
        Role, on_delete=models.SET_NULL, null=True, blank=True, related_name="agents"
    )
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        Account,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_agents",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            # Per organization, for the same reason as Tool.name. `uuid` stays
            # globally unique because it is the external identifier callers
            # pass, and it is generated rather than chosen.
            models.UniqueConstraint(
                fields=["organization", "slug"], name="unique_agent_slug_per_org"
            ),
        ]

    def __str__(self):
        role_name = self.role.name if self.role else "No Role"
        return f"{self.name} ({self.organization.name}, Role: {role_name})"


class Service(models.Model):
    """An LLM provider. Global: "OpenAI" means the same thing to everyone."""

    uuid = models.CharField(default=new_id, null=False, unique=True)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class Model(models.Model):
    """A model offered by a Service. Global, for the same reason."""

    uuid = models.CharField(default=new_id, null=False, unique=True)
    service = models.ForeignKey(
        Service,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_service",
    )
    model = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.service.name if self.service else '?'}/{self.model}"


def new_uuid():
    """Default for KnowledgeDocument.id. See neon.utils.identifiers.new_id.

    Delegates rather than being deleted: migration 0010 names this function by
    import path, and a migration whose default cannot be imported will not
    load.
    """
    return new_id()


class KnowledgeDocument(models.Model):
    """A document an organization has indexed for retrieval.

    WHY A ROW AT ALL, WHEN PINECONE HOLDS THE VECTORS
    -------------------------------------------------
    Because nothing else can answer "what did I upload?" or "remove that one".
    `bulk_index_docs` mints vector ids internally and returns them; without
    somewhere to keep them, every upload is permanent and invisible - the
    vectors stay in the index forever with no way to name them, and re-uploading
    a corrected document leaves the wrong version answering alongside it.

    The row also carries indexing STATUS, because embedding happens on a Celery
    worker: the upload responds long before the vectors exist, and a failure
    there would otherwise be silent.
    """

    STATUS_PENDING = "pending"
    STATUS_INDEXING = "indexing"
    STATUS_INDEXED = "indexed"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_INDEXING, "Indexing"),
        (STATUS_INDEXED, "Indexed"),
        (STATUS_FAILED, "Failed"),
    ]

    id = models.CharField(max_length=150, primary_key=True, default=new_uuid)
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="knowledge_documents"
    )
    title = models.CharField(max_length=255)
    source_name = models.CharField(
        max_length=255, blank=True, default="", help_text="Original filename, if any."
    )
    content_type = models.CharField(max_length=100, blank=True, default="")
    size_bytes = models.PositiveIntegerField(default=0)

    # The text as indexed. Kept so a document can be re-indexed after a change
    # to chunking or embedding model without asking the user to upload again.
    content = models.TextField(blank=True, default="")

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING
    )
    error = models.TextField(blank=True, default="")
    chunk_count = models.PositiveIntegerField(default=0)
    # What to delete from Pinecone. The only handle on those vectors that
    # exists anywhere.
    vector_ids = models.JSONField(default=list, blank=True)

    uploaded_by = models.ForeignKey(
        Account,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="uploaded_documents",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.organization.name})"
