from django.utils.text import slugify
from rest_framework import serializers

from .models import Agent, KnowledgeDocument, Model, Role, Service, Tool


class ToolSerializer(serializers.ModelSerializer):
    """A Tool as the API shows it.

    `authentication` is WRITE-ONLY. This serializer used to be
    `fields = "__all__"`, and its output was `json.dumps`'d into the system
    prompt - putting every tool's credential into the model provider's logs,
    into any prompt tracing, and into anything the model could be talked into
    repeating back.

    It still has to be settable, because somebody has to configure the tool.
    Write-only is what allows that without the value ever travelling back out:
    execution reads the Tool row directly (llm/utils/tool_execution.py), so
    nothing downstream needs it from here.
    """

    authentication = serializers.CharField(
        write_only=True, required=False, allow_blank=True
    )
    has_authentication = serializers.SerializerMethodField()

    class Meta:
        model = Tool
        fields = [
            "id",
            "name",
            "description",
            "parameters_schema",
            "headers_schema",
            "api_endpoint",
            "http_method",
            "param_type",
            "requires_auth",
            "is_enabled",
            "authentication",
            "has_authentication",
        ]
        read_only_fields = ["id"]

    def get_has_authentication(self, obj):
        return bool(obj.authentication)

    def validate_parameters_schema(self, value):
        """The schema goes to the model provider as a JSON Schema object.

        Rejected here rather than at call time because a malformed schema
        makes the provider reject the whole completion - so one bad tool
        breaks every conversation for the organization, not just the tool.
        """
        if value in (None, ""):
            return value
        if not isinstance(value, dict):
            raise serializers.ValidationError(
                "Must be a JSON Schema object, e.g. "
                '{"type": "object", "properties": {...}}.'
            )
        return value

    def validate_headers_schema(self, value):
        if value in (None, ""):
            return value
        if not isinstance(value, dict):
            raise serializers.ValidationError("Must be a JSON object.")
        return value

    def validate(self, attrs):
        # An enabled tool with no endpoint is a tool the model will choose and
        # then fail to call - it looks like the agent is broken rather than
        # misconfigured. Caught at save, where the person who can fix it is.
        enabled = attrs.get(
            "is_enabled", getattr(self.instance, "is_enabled", False)
        )
        endpoint = attrs.get(
            "api_endpoint", getattr(self.instance, "api_endpoint", None)
        )
        if enabled and not endpoint:
            raise serializers.ValidationError(
                {"api_endpoint": "An enabled tool needs an endpoint to call."}
            )
        return attrs


class ToolSummarySerializer(serializers.ModelSerializer):
    """Tools as they appear nested inside a Role."""

    class Meta:
        model = Tool
        fields = ["id", "name", "description", "is_enabled"]
        read_only_fields = fields


class RoleSerializer(serializers.ModelSerializer):
    tools = ToolSummarySerializer(many=True, read_only=True)
    tool_ids = serializers.PrimaryKeyRelatedField(
        many=True,
        write_only=True,
        required=False,
        queryset=Tool.objects.none(),
        source="tools",
    )
    agent_count = serializers.SerializerMethodField()

    class Meta:
        model = Role
        fields = [
            "id",
            "name",
            "description",
            "system_prompt",
            "tools",
            "tool_ids",
            "agent_count",
        ]
        read_only_fields = ["id"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The tool choices are restricted to the caller's own organization.
        # Without this, `tool_ids` would accept ANY tool id on the platform and
        # attach another tenant's tool - including its endpoint and credential -
        # to this organization's role. A PrimaryKeyRelatedField's queryset is
        # the authorization check, not just a lookup.
        organization = self.context.get("organization")
        if organization is not None:
            self.fields["tool_ids"].child_relation.queryset = Tool.objects.filter(
                organization=organization
            )

    def get_agent_count(self, obj):
        return obj.agents.count()

    def validate_name(self, value):
        return value.strip()


class RoleSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = ["id", "name", "description"]
        read_only_fields = fields


class AgentSerializer(serializers.ModelSerializer):
    role = RoleSummarySerializer(read_only=True)
    role_id = serializers.PrimaryKeyRelatedField(
        write_only=True,
        required=False,
        allow_null=True,
        queryset=Role.objects.none(),
        source="role",
    )
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True, default=""
    )
    # Derived from the name when absent. A slug is a URL detail, and asking
    # somebody to invent one before they can create their first agent is a
    # step with no decision in it.
    slug = serializers.SlugField(required=False, allow_blank=True)

    class Meta:
        model = Agent
        fields = [
            "uuid",
            "name",
            "slug",
            "role",
            "role_id",
            "is_active",
            "created_by_username",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["uuid", "created_at", "updated_at"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Same reasoning as RoleSerializer.tool_ids: this queryset is what
        # stops an agent being pointed at another organization's role, and
        # with it that role's system prompt and tools.
        organization = self.context.get("organization")
        if organization is not None:
            self.fields["role_id"].queryset = Role.objects.filter(
                organization=organization
            )

    def validate(self, attrs):
        slug = (attrs.get("slug") or "").strip()
        if not slug:
            name = attrs.get("name") or getattr(self.instance, "name", "")
            slug = slugify(name)
        if not slug:
            raise serializers.ValidationError(
                {"slug": "Could not derive a slug from that name; provide one."}
            )

        organization = self.context.get("organization")
        if organization is not None:
            clash = Agent.objects.filter(organization=organization, slug=slug)
            if self.instance is not None:
                clash = clash.exclude(pk=self.instance.pk)
            if clash.exists():
                # Checked here so the caller gets a field error rather than the
                # 500 an unhandled IntegrityError on unique_agent_slug_per_org
                # would produce.
                raise serializers.ValidationError(
                    {"slug": "An agent in this organization already uses that slug."}
                )

        attrs["slug"] = slug
        return attrs


class ServiceSerializer(serializers.ModelSerializer):
    """Providers. Global and read-only through the API.

    "OpenAI" means the same thing to every organization, so these are platform
    data rather than tenant data - letting one organization rename or delete a
    Service would change it for everybody.
    """

    class Meta:
        model = Service
        # `id` is here because ProviderCredential.service is a foreign key on
        # the numeric pk, so a client building a credential payload needs it.
        # Without it the catalogue is addressable only by uuid and the two
        # endpoints cannot be used together at all.
        fields = ["id", "uuid", "name"]
        read_only_fields = fields


class ModelSerializer(serializers.ModelSerializer):
    service_name = serializers.CharField(source="service.name", read_only=True)
    service_uuid = serializers.CharField(source="service.uuid", read_only=True)

    class Meta:
        model = Model
        fields = ["uuid", "model", "service_uuid", "service_name"]
        read_only_fields = fields


class KnowledgeDocumentSerializer(serializers.ModelSerializer):
    uploaded_by_username = serializers.CharField(
        source="uploaded_by.username", read_only=True, default=""
    )
    # `content` is not listed: a document can be tens of thousands of
    # characters, and returning every one of them in a listing would make the
    # knowledge screen unusable. The detail view returns it explicitly.

    class Meta:
        model = KnowledgeDocument
        fields = [
            "id",
            "title",
            "source_name",
            "content_type",
            "size_bytes",
            "status",
            "error",
            "chunk_count",
            "uploaded_by_username",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields
