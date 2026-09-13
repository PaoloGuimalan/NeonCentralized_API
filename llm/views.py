"""Agents, roles, tools, models and knowledge.

These were manageable only through Django admin, which meant building an agent
required staff access and the frontend hardcoded the uuids it needed. Every
queryset here filters by the caller's organization via
`neon.api.OrganizationScopedView`.

Where a serializer accepts a related id - a role's tools, an agent's role - the
choices are restricted to the same organization inside the serializer, because
filtering the queryset you READ is not enough on its own: an unrestricted
`PrimaryKeyRelatedField` will happily attach another tenant's tool, credential
included, to a role in this one.
"""

import json
import logging

from django.db import IntegrityError
from rest_framework import status

from neon.api import OrganizationScopedView, fail, ok
from organization.credentials import CredentialNotConfigured, embedding_api_key

from .models import Agent, KnowledgeDocument, Model, Role, Service, Tool
from .scripts.tasks import delete_knowledge_vectors_task, index_knowledge_document_task
from .serializers import (
    AgentSerializer,
    KnowledgeDocumentSerializer,
    ModelSerializer,
    RoleSerializer,
    ServiceSerializer,
    ToolSerializer,
)

logger = logging.getLogger(__name__)

# Documents are embedded in full, and every chunk is stored in Pinecone and
# again on the row. A cap belongs here rather than at the web server, where it
# would be a 413 with no explanation.
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024

# Anything else needs parsing before it is text. Silently indexing the raw
# bytes of a PDF produces an index full of binary noise that quietly degrades
# every answer, which is worse than refusing it.
TEXT_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "application/json",
    "application/octet-stream",  # what browsers send for .md on some platforms
}
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".csv", ".json", ".rst", ".log")


class AgentListView(OrganizationScopedView):

    def get(self, request):
        agents = (
            Agent.objects.filter(organization=self.organization)
            .select_related("role", "created_by")
            .order_by("-created_at")
        )
        return ok(AgentSerializer(agents, many=True).data)

    def post(self, request):
        serializer = AgentSerializer(
            data=request.data, context={"organization": self.organization}
        )
        serializer.is_valid(raise_exception=True)
        try:
            agent = serializer.save(
                organization=self.organization, created_by=request.user
            )
        except IntegrityError:
            return fail(
                "An agent in this organization already uses that slug.",
                status.HTTP_409_CONFLICT,
            )
        return ok(AgentSerializer(agent).data, status.HTTP_201_CREATED)


class AgentDetailView(OrganizationScopedView):

    def _agent(self, agent_uuid):
        return (
            Agent.objects.filter(uuid=agent_uuid, organization=self.organization)
            .select_related("role", "created_by")
            .first()
        )

    def get(self, request, agent_uuid):
        agent = self._agent(agent_uuid)
        if agent is None:
            return fail("Agent not found.", status.HTTP_404_NOT_FOUND)
        return ok(AgentSerializer(agent).data)

    def patch(self, request, agent_uuid):
        agent = self._agent(agent_uuid)
        if agent is None:
            return fail("Agent not found.", status.HTTP_404_NOT_FOUND)

        serializer = AgentSerializer(
            agent,
            data=request.data,
            partial=True,
            context={"organization": self.organization},
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return ok(serializer.data)

    def delete(self, request, agent_uuid):
        agent = self._agent(agent_uuid)
        if agent is None:
            return fail("Agent not found.", status.HTTP_404_NOT_FOUND)

        # Deactivated, not deleted. Messages carry an FK to the agent that
        # wrote them (`messenger.Message.agent`, on_delete=DO_NOTHING), so a
        # hard delete would leave rows pointing at nothing and a conversation
        # history that cannot say who answered.
        agent.is_active = False
        agent.save(update_fields=["is_active", "updated_at"])
        return ok(AgentSerializer(agent).data, message="Agent deactivated.")


class RoleListView(OrganizationScopedView):

    def get(self, request):
        roles = (
            Role.objects.filter(organization=self.organization)
            .prefetch_related("tools")
            .order_by("name")
        )
        return ok(RoleSerializer(roles, many=True).data)

    def post(self, request):
        serializer = RoleSerializer(
            data=request.data, context={"organization": self.organization}
        )
        serializer.is_valid(raise_exception=True)
        try:
            role = serializer.save(organization=self.organization)
        except IntegrityError:
            return fail(
                "A role with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )
        return ok(RoleSerializer(role).data, status.HTTP_201_CREATED)


class RoleDetailView(OrganizationScopedView):

    def _role(self, role_id):
        return (
            Role.objects.filter(id=role_id, organization=self.organization)
            .prefetch_related("tools")
            .first()
        )

    def get(self, request, role_id):
        role = self._role(role_id)
        if role is None:
            return fail("Role not found.", status.HTTP_404_NOT_FOUND)
        return ok(RoleSerializer(role).data)

    def patch(self, request, role_id):
        role = self._role(role_id)
        if role is None:
            return fail("Role not found.", status.HTTP_404_NOT_FOUND)

        serializer = RoleSerializer(
            role,
            data=request.data,
            partial=True,
            context={"organization": self.organization},
        )
        serializer.is_valid(raise_exception=True)
        try:
            serializer.save()
        except IntegrityError:
            return fail(
                "A role with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )
        return ok(serializer.data)

    def delete(self, request, role_id):
        role = self._role(role_id)
        if role is None:
            return fail("Role not found.", status.HTTP_404_NOT_FOUND)

        # Agents point at roles with SET_NULL, so deleting one leaves its
        # agents running with no system prompt at all - a behaviour change
        # nobody would connect to this action. Named, and refused.
        agents = list(role.agents.values_list("name", flat=True)[:5])
        if agents:
            return fail(
                "This role is still used by: " + ", ".join(agents) +
                ". Reassign those agents first.",
                status.HTTP_409_CONFLICT,
            )

        role.delete()
        return ok(None, message="Deleted.")


class ToolListView(OrganizationScopedView):

    def get(self, request):
        tools = Tool.objects.filter(organization=self.organization).order_by("name")
        return ok(ToolSerializer(tools, many=True).data)

    def post(self, request):
        serializer = ToolSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            tool = serializer.save(organization=self.organization)
        except IntegrityError:
            return fail(
                "A tool with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )
        return ok(ToolSerializer(tool).data, status.HTTP_201_CREATED)


class ToolDetailView(OrganizationScopedView):

    def _tool(self, tool_id):
        return Tool.objects.filter(id=tool_id, organization=self.organization).first()

    def get(self, request, tool_id):
        tool = self._tool(tool_id)
        if tool is None:
            return fail("Tool not found.", status.HTTP_404_NOT_FOUND)
        return ok(ToolSerializer(tool).data)

    def patch(self, request, tool_id):
        tool = self._tool(tool_id)
        if tool is None:
            return fail("Tool not found.", status.HTTP_404_NOT_FOUND)

        serializer = ToolSerializer(tool, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        try:
            serializer.save()
        except IntegrityError:
            return fail(
                "A tool with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )
        return ok(serializer.data)

    def delete(self, request, tool_id):
        tool = self._tool(tool_id)
        if tool is None:
            return fail("Tool not found.", status.HTTP_404_NOT_FOUND)

        # Unlike a role, a tool detaching from its roles is harmless - the
        # agent simply stops being offered it - so this really does delete.
        tool.delete()
        return ok(None, message="Deleted.")


class ServiceListView(OrganizationScopedView):
    """Providers. Read-only: see ServiceSerializer for why they are global."""

    def get(self, request):
        return ok(ServiceSerializer(Service.objects.order_by("name"), many=True).data)


class ModelListView(OrganizationScopedView):
    """Models, optionally filtered to one provider."""

    def get(self, request):
        models = Model.objects.select_related("service").order_by(
            "service__name", "model"
        )
        service_uuid = request.query_params.get("service")
        if service_uuid:
            models = models.filter(service__uuid=service_uuid)
        return ok(ModelSerializer(models, many=True).data)


class KnowledgeListView(OrganizationScopedView):
    """Documents an organization has indexed, and adding one.

    Accepts either a `file` upload or a `content` string, because both are real:
    a policy document arrives as a file, and a paragraph of context somebody
    typed does not.
    """

    def get(self, request):
        documents = KnowledgeDocument.objects.filter(
            organization=self.organization
        ).select_related("uploaded_by")
        return ok(KnowledgeDocumentSerializer(documents, many=True).data)

    def post(self, request):
        # Checked BEFORE the row is written. Without it the document is stored,
        # queued, and fails on the worker - so the user sees "failed" with a
        # provider error rather than being told up front that no embedding key
        # is configured, which is the thing they can actually fix.
        try:
            embedding_api_key(self.organization)
        except CredentialNotConfigured as ex:
            return fail(str(ex), status.HTTP_409_CONFLICT)

        upload = request.FILES.get("file")
        if upload is not None:
            rejection, parsed = self._read_upload(upload)
            if rejection is not None:
                return rejection
            title, content, content_type, size = parsed
        else:
            content = (request.data.get("content") or "").strip()
            if not content:
                return fail("Provide a file or some content to index.")
            size = len(content.encode("utf-8"))
            if size > MAX_DOCUMENT_BYTES:
                return fail(
                    f"That content is larger than the "
                    f"{MAX_DOCUMENT_BYTES // (1024 * 1024)}MB limit.",
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                )
            title = (request.data.get("title") or "").strip() or "Untitled document"
            content_type = "text/plain"

        title = (request.data.get("title") or "").strip() or title

        document = KnowledgeDocument.objects.create(
            organization=self.organization,
            title=title[:255],
            source_name=getattr(upload, "name", "")[:255],
            content_type=content_type[:100],
            size_bytes=size,
            content=content,
            uploaded_by=request.user,
        )

        # Queued rather than indexed inline: embedding a long document is
        # several provider round trips, and the upload should not hold the
        # connection open for them. The row's status is what makes the result
        # visible afterwards.
        index_knowledge_document_task.delay(document.pk)

        return ok(
            KnowledgeDocumentSerializer(document).data, status.HTTP_201_CREATED
        )

    def _read_upload(self, upload):
        """Decode an uploaded file.

        Returns `(rejection, parsed)` - exactly one of which is None. Two
        return slots rather than one because the success value is itself a
        tuple, so "is it a tuple?" cannot distinguish them.
        """
        if upload.size > MAX_DOCUMENT_BYTES:
            return (
                fail(
                    f"{upload.name} is larger than the "
                    f"{MAX_DOCUMENT_BYTES // (1024 * 1024)}MB limit.",
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                ),
                None,
            )

        name = (upload.name or "").lower()
        content_type = (upload.content_type or "").split(";")[0].strip()
        if content_type not in TEXT_CONTENT_TYPES and not name.endswith(TEXT_SUFFIXES):
            return (
                fail(
                    f"{upload.name} is not a text document. Plain text, Markdown, "
                    "CSV and JSON can be indexed; PDFs and Office documents need "
                    "to be converted to text first.",
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                ),
                None,
            )

        raw = upload.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Named rather than decoded with errors="replace": a file that is
            # not really text would otherwise be indexed as mojibake, and
            # nothing downstream would ever report it.
            return (
                fail(
                    f"{upload.name} is not valid UTF-8 text.",
                    status.HTTP_400_BAD_REQUEST,
                ),
                None,
            )

        if content_type == "application/json" or name.endswith(".json"):
            try:
                json.loads(content)
            except ValueError:
                return (
                    fail(
                        f"{upload.name} is not valid JSON.",
                        status.HTTP_400_BAD_REQUEST,
                    ),
                    None,
                )

        content = content.strip()
        if not content:
            return (
                fail(f"{upload.name} is empty.", status.HTTP_400_BAD_REQUEST),
                None,
            )

        title = upload.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        return None, (title, content, content_type or "text/plain", upload.size)


class KnowledgeDetailView(OrganizationScopedView):

    def _document(self, document_id):
        return (
            KnowledgeDocument.objects.filter(
                id=document_id, organization=self.organization
            )
            .select_related("uploaded_by")
            .first()
        )

    def get(self, request, document_id):
        document = self._document(document_id)
        if document is None:
            return fail("Document not found.", status.HTTP_404_NOT_FOUND)

        payload = KnowledgeDocumentSerializer(document).data
        # The detail view is the one place the full text is returned - the
        # listing omits it because a knowledge screen showing fifty documents
        # would otherwise transfer every character of all of them.
        payload["content"] = document.content
        return ok(payload)

    def post(self, request, document_id):
        """Re-index. Replaces the previous vectors rather than adding to them."""
        document = self._document(document_id)
        if document is None:
            return fail("Document not found.", status.HTTP_404_NOT_FOUND)

        document.status = KnowledgeDocument.STATUS_PENDING
        document.error = ""
        document.save(update_fields=["status", "error", "updated_at"])
        index_knowledge_document_task.delay(document.pk)
        return ok(KnowledgeDocumentSerializer(document).data, message="Re-indexing.")

    def delete(self, request, document_id):
        document = self._document(document_id)
        if document is None:
            return fail("Document not found.", status.HTTP_404_NOT_FOUND)

        vector_ids = list(document.vector_ids or [])
        document.delete()

        # Vectors go on a worker, after the row. A Pinecone outage must not
        # stop somebody removing a document; orphaned vectors are recoverable,
        # a row that refuses to delete is a dead end. The trade is that
        # retrieval can cite a deleted document until the task runs.
        if vector_ids:
            delete_knowledge_vectors_task.delay(vector_ids, str(self.organization.pk))

        return ok(None, message="Deleted.")
