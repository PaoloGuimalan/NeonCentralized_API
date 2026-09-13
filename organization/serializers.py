from django.utils.text import slugify
from rest_framework import serializers

from user.models import Account

from .models import Member, Organization, ProviderCredential


class OrganizationSerializer(serializers.ModelSerializer):
    member_count = serializers.SerializerMethodField()
    is_owner = serializers.SerializerMethodField()

    class Meta:
        model = Organization
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "website",
            "address",
            "contact_email",
            "phone_number",
            "is_active",
            "created_at",
            "updated_at",
            "member_count",
            "is_owner",
        ]
        # `access_key` and `pin` are deliberately absent: they authenticate
        # the organization and belong in whatever issues them, not in a
        # listing every member can read. `llm_api_key` is absent for the same
        # reason - it is a provider credential, and ProviderCredential is
        # where credentials are managed now.
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_member_count(self, obj):
        return obj.member_set.count()

    def get_is_owner(self, obj):
        account = self.context.get("account")
        return bool(account) and str(obj.created_by_id) == str(account.pk)


class OrganizationCreateSerializer(serializers.ModelSerializer):
    """Creating an organization, which is the first thing a new user does.

    `slug` is optional and derived from the name when absent, because it is a
    URL detail rather than a decision worth putting in front of somebody who
    has just signed in and wants to build an agent.
    """

    slug = serializers.SlugField(required=False, allow_blank=True)

    class Meta:
        model = Organization
        fields = [
            "name",
            "slug",
            "description",
            "website",
            "address",
            "contact_email",
            "phone_number",
        ]

    def validate_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("A name is required.")
        return value

    def validate(self, attrs):
        slug = (attrs.get("slug") or "").strip() or slugify(attrs["name"])
        if not slug:
            # A name of only punctuation or non-latin characters slugifies to
            # nothing, and a blank slug would collide with the next one.
            raise serializers.ValidationError(
                {"slug": "Could not derive a URL slug from that name; provide one."}
            )

        candidate, suffix = slug, 2
        while Organization.objects.filter(slug=candidate).exists():
            candidate = f"{slug}-{suffix}"
            suffix += 1
        attrs["slug"] = candidate
        return attrs


class MemberSerializer(serializers.ModelSerializer):
    account_id = serializers.CharField(source="account.id", read_only=True)
    username = serializers.CharField(source="account.username", read_only=True)
    email = serializers.EmailField(source="account.email", read_only=True)
    full_name = serializers.SerializerMethodField()
    profile = serializers.CharField(source="account.profile", read_only=True)
    entity_id = serializers.CharField(source="account.entity_id", read_only=True)
    is_owner = serializers.SerializerMethodField()

    class Meta:
        model = Member
        fields = [
            "id",
            "account_id",
            "username",
            "email",
            "full_name",
            "profile",
            "entity_id",
            "nickname",
            "date_joined",
            "is_owner",
        ]
        read_only_fields = [f for f in fields if f != "nickname"]

    def get_full_name(self, obj):
        return f"{obj.account.first_name} {obj.account.last_name}".strip()

    def get_is_owner(self, obj):
        return str(obj.organization.created_by_id) == str(obj.account_id)


class MemberInviteSerializer(serializers.Serializer):
    """Adding an existing Neon account to an organization.

    By email, and only for an account that already exists. Neon cannot create
    one: chatterloop owns identity, so there is nobody for Neon to invite who
    has not signed in here at least once. Saying that plainly beats a silent
    no-op or a half-made account that can never be signed in to.
    """

    email = serializers.EmailField()
    nickname = serializers.CharField(
        required=False, allow_blank=True, max_length=150
    )

    def validate_email(self, value):
        account = Account.objects.filter(email__iexact=value.strip()).first()
        if account is None:
            raise serializers.ValidationError(
                "No Neon account uses that email. They need to sign in to Neon "
                "with their Chatterloop account once before they can be added."
            )
        self.context["account"] = account
        return value


class ProviderCredentialSerializer(serializers.ModelSerializer):
    """A provider credential as the API shows it.

    `api_key` is WRITE-ONLY. A credential that can be read back through the
    API is one that any XSS, any over-broad token and any leaky log can
    exfiltrate, and there is no product reason to read it: the key is used
    server-side and nowhere else. `api_key_hint` gives the last four
    characters, which is enough to tell two keys apart when deciding which to
    replace and not enough to use.
    """

    service_name = serializers.CharField(source="service.name", read_only=True)
    api_key = serializers.CharField(write_only=True, required=False, allow_blank=True)
    api_key_hint = serializers.SerializerMethodField()
    has_api_key = serializers.SerializerMethodField()
    # What to call it when the name is blank - the provider's own name.
    label = serializers.CharField(read_only=True)
    bot_count = serializers.SerializerMethodField()

    class Meta:
        model = ProviderCredential
        fields = [
            "id",
            "name",
            "label",
            "service",
            "service_name",
            "api_key",
            "api_key_hint",
            "has_api_key",
            "is_default",
            "is_embedding_default",
            "is_active",
            "bot_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "label", "bot_count", "created_at", "updated_at"]

    def get_bot_count(self, obj):
        """How many bots are pinned to this key.

        Shown because deleting a key that bots are assigned to silently drops
        them back to the organization default - a billing change nobody asked
        for - so the count is what makes that visible beforehand.
        """
        return obj.bots.count()

    def validate(self, attrs):
        """Fill in a name, and keep it unique within the organization.

        The column permits NULL, so an unnamed key is legal - but one created
        through the API gets a real name anyway, because a list of keys all
        called "OpenAI" is not something anybody can act on. Derived from the
        provider and disambiguated the same way an organization slug is.
        """
        organization = self.context.get("organization")
        service = attrs.get("service") or getattr(self.instance, "service", None)
        supplied = (attrs.get("name") or "").strip()

        if not supplied:
            if self.instance is not None and self.instance.name:
                return attrs
            name, chosen_by_user = (
                service.name if service is not None else "Key"
            ), False
        else:
            name, chosen_by_user = supplied, True

        if organization is None:
            attrs["name"] = name
            return attrs

        taken = ProviderCredential.objects.filter(organization=organization)
        if self.instance is not None:
            taken = taken.exclude(pk=self.instance.pk)

        if taken.filter(name=name).exists():
            if chosen_by_user:
                # They typed it, so they meant it. Quietly renaming somebody's
                # "Billing: Acme" to "Billing: Acme 2" is how two keys end up
                # looking interchangeable when they bill different accounts.
                raise serializers.ValidationError(
                    {"name": f"A credential called {name!r} already exists here."}
                )
            candidate, suffix = name, 2
            while taken.filter(name=candidate).exists():
                candidate = f"{name} {suffix}"
                suffix += 1
            name = candidate

        attrs["name"] = name
        return attrs

    def get_has_api_key(self, obj):
        return bool(obj.api_key_encrypted)

    def get_api_key_hint(self, obj):
        if not obj.api_key_encrypted:
            return ""
        try:
            key = obj.api_key
        except Exception:
            # A key encrypted under a rotated-away TOKEN_ENCRYPTION_KEY. The
            # listing still has to render - telling the user which credential
            # is unreadable is exactly how they find out they need to replace
            # it - so this reports the state instead of 500ing the page.
            return "unreadable"
        return f"…{key[-4:]}" if len(key) > 4 else "…"

    def create(self, validated_data):
        api_key = validated_data.pop("api_key", "")
        credential = ProviderCredential(**validated_data)
        credential.api_key = api_key
        credential.save()
        return credential

    def update(self, instance, validated_data):
        # A blank `api_key` means "leave it alone", not "clear it". Editing
        # the embedding-default flag on a form that cannot show the existing
        # key would otherwise wipe it.
        api_key = validated_data.pop("api_key", None)
        for field, value in validated_data.items():
            setattr(instance, field, value)
        if api_key:
            instance.api_key = api_key
        instance.save()
        return instance
