from rest_framework import serializers

from .models import ChatterloopBot, ChatterloopToken
from .provisioning import ALL_SCOPES, DEFAULT_SCOPES


class ChatterloopTokenSerializer(serializers.ModelSerializer):
    """A credential as the API shows it.

    The secret is absent and there is no field for it. It is returned exactly
    once, by the endpoint that mints it, and never again - not because the
    server cannot decrypt it, but because a credential that can be re-read
    through the API is one that any XSS or over-broad session can walk off
    with. `prefix` is safe to show: it travels in the clear inside every token,
    which is how verification finds the row.
    """

    is_live = serializers.BooleanField(read_only=True)
    masked = serializers.SerializerMethodField()

    class Meta:
        model = ChatterloopToken
        fields = [
            "id",
            "name",
            "prefix",
            "masked",
            "scopes",
            "expires_at",
            "revoked_at",
            "provisioned_at",
            "last_verified_at",
            "created_at",
            "is_live",
        ]
        read_only_fields = fields

    def get_masked(self, obj):
        return f"clt_{obj.prefix}_{'•' * 8}"


class ChatterloopBotSerializer(serializers.ModelSerializer):
    agent_uuid = serializers.CharField(source="agent.uuid", read_only=True, default="")
    agent_name = serializers.CharField(source="agent.name", read_only=True, default="")
    model_uuid = serializers.CharField(source="model.uuid", read_only=True, default="")
    model_name = serializers.CharField(source="model.model", read_only=True, default="")
    credential_id = serializers.CharField(
        source="provider_credential.id", read_only=True, default=""
    )
    # Blank means the organization default - shown as such rather than as
    # nothing, since "no key" and "the shared key" are very different.
    credential_name = serializers.CharField(
        source="provider_credential.label", read_only=True, default=""
    )
    owner_name = serializers.SerializerMethodField()
    owner_type = serializers.SerializerMethodField()
    tokens = ChatterloopTokenSerializer(many=True, read_only=True)
    is_live = serializers.BooleanField(read_only=True)
    handle_mismatch = serializers.BooleanField(read_only=True)
    # Whether the bot could actually produce a reply. Three things have to
    # hold and each fails silently on its own - see ChatterloopBot.can_answer.
    can_answer = serializers.BooleanField(read_only=True)
    # What SHOULD be true, from the database.
    should_run = serializers.BooleanField(read_only=True)
    # What IS true, read from the lease in Redis. Kept separate because they
    # genuinely disagree in the case that matters: a bot switched on with no
    # supervisor process running is "should_run, not running", and reporting
    # that as simply "online" is how somebody spends an afternoon wondering
    # why their bot is ignoring them.
    running = serializers.SerializerMethodField()
    control_url = serializers.SerializerMethodField()
    control_key = serializers.SerializerMethodField()

    class Meta:
        model = ChatterloopBot
        fields = [
            "id",
            "entity_id",
            "name",
            "handle",
            "description",
            "status",
            "status_reason",
            "agent_uuid",
            "agent_name",
            "model_uuid",
            "model_name",
            "credential_id",
            "credential_name",
            "can_answer",
            "is_online",
            "online_changed_at",
            "allow_bot_conversations",
            "should_run",
            "running",
            "control_url",
            "control_key",
            "owner_name",
            "owner_type",
            "owner_entity_id",
            "verified_handle",
            "handle_mismatch",
            "last_verified_at",
            "tokens",
            "is_live",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_control_key(self, obj):
        """The key the control endpoint accepts.

        ON THE BOT, not behind a second call: the URL is useless without it,
        and a pair of links somebody cannot actually use is worse than not
        showing them. The cost is real and accepted - this credential now
        appears in every bot listing, so it reaches logs, caches and
        screenshots the way any displayed secret does. It is scoped to one bot
        and one power, and `POST control-key` rotates it if a copy escapes.

        `ensure_control_key` WRITES on first use. New bots get theirs at mint
        time so this is normally a pure read; the write is the one-off for
        bots that predate the feature.
        """
        from .presence import ensure_control_key

        return ensure_control_key(obj)

    def get_control_url(self, obj):
        """The bot's control endpoint, ready to copy.

        Built from the REQUEST when there is one, so it names the host
        somebody is actually using, and falls back to NEON_PUBLIC_BASE_URL for
        callers with no request - a management command, a task.
        """
        from django.conf import settings

        from .presence import control_url

        request = self.context.get("request")
        if request is not None:
            return request.build_absolute_uri(control_url(obj))
        return control_url(obj, getattr(settings, "NEON_PUBLIC_BASE_URL", ""))

    def get_running(self, obj):
        """Whether a supervisor currently holds this bot's lease.

        Derived from Redis rather than stored, so it cannot go stale: a
        supervisor that died leaves its lease to expire, and this starts
        answering False on its own with nothing to clean up.

        The set is passed in by the view for a listing, so N bots cost one
        Redis round trip rather than N.
        """
        running = self.context.get("running")
        if running is None:
            return None
        return str(obj.pk) in running

    def get_owner_name(self, obj):
        if obj.connected_account is not None:
            return obj.connected_account.external_name
        # No connected account means the minting user's own identity.
        return obj.created_by.username if obj.created_by else ""

    def get_owner_type(self, obj):
        return "page" if obj.connected_account is not None else "personal"


class CreateBotSerializer(serializers.Serializer):
    """What the client sends to mint a bot.

    `owner` is either the literal "personal" or a ConnectedAccount id. It is
    NOT an entity id: accepting one would mean trusting the client to say which
    identity it may speak as, and that is decided server-side against
    chatterloop's own membership table.
    """

    name = serializers.CharField(max_length=80)
    handle = serializers.RegexField(
        # Same shape as a platform handle. Validated here so an invalid one is
        # a field error rather than a database constraint failure halfway
        # through minting.
        r"^[A-Za-z0-9_.-]+$",
        max_length=50,
        error_messages={
            "invalid": "A handle may contain only letters, numbers, dots, "
            "underscores and hyphens."
        },
    )
    description = serializers.CharField(
        max_length=2000, required=False, allow_blank=True, default=""
    )
    agent_uuid = serializers.CharField(required=False, allow_blank=True, default="")
    model_uuid = serializers.CharField(required=False, allow_blank=True, default="")
    # Which API key this bot bills to. Blank = the organization default.
    credential_id = serializers.CharField(required=False, allow_blank=True, default="")
    owner = serializers.CharField(default="personal")
    scopes = serializers.ListField(
        child=serializers.ChoiceField(choices=ALL_SCOPES),
        required=False,
        default=list(DEFAULT_SCOPES),
    )

    def validate_handle(self, value):
        return value.strip().lstrip("@")

    def validate_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("A name is required.")
        return value


class UpdateBotSerializer(serializers.Serializer):
    """Only what is safe to change after minting.

    `handle` is absent on purpose. Changing it means a write to chatterloop
    plus a fresh uniqueness check across three tables, and anything already
    configured to mention the old one silently stops working. Minting a new bot
    is the honest path.
    """

    name = serializers.CharField(max_length=80, required=False)
    description = serializers.CharField(
        max_length=2000, required=False, allow_blank=True
    )
    agent_uuid = serializers.CharField(required=False, allow_blank=True)
    model_uuid = serializers.CharField(required=False, allow_blank=True)
    credential_id = serializers.CharField(required=False, allow_blank=True)
    allow_bot_conversations = serializers.BooleanField(required=False)


class RotateTokenSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=120, required=False, allow_blank=True)
    scopes = serializers.ListField(
        child=serializers.ChoiceField(choices=ALL_SCOPES), required=False
    )
