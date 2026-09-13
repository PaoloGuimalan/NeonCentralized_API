from rest_framework import serializers

from .models import ConnectedAccount


class ConnectedAccountSerializer(serializers.ModelSerializer):
    """A connected identity as the API shows it.

    Everything here is already public within chatterloop - a page's name, slug
    and avatar. `metadata` is exposed because its contents (the realm id and
    the role the connection was granted under) are what a user needs to
    understand why a page is listed and why it might stop being.
    """

    class Meta:
        model = ConnectedAccount
        fields = [
            "id",
            "provider",
            "external_id",
            "external_type",
            "external_username",
            "external_name",
            "external_profile",
            "metadata",
            "connected_at",
            "last_verified_at",
            "is_active",
        ]
        read_only_fields = fields


class AvailablePageSerializer(serializers.Serializer):
    """A chatterloop page the signed-in user could connect but has not.

    Not a ModelSerializer: these are rows read live out of chatterloop's
    database, never stored in Neon unless the user actually connects one.
    """

    realm_id = serializers.CharField()
    entity_id = serializers.CharField()
    name = serializers.CharField()
    slug = serializers.CharField(allow_blank=True)
    profile = serializers.CharField(allow_blank=True)
    role = serializers.CharField()


class ConnectPageSerializer(serializers.Serializer):
    """What the client sends to connect a page.

    Only the realm entity id. Everything else about the page is read from
    chatterloop server-side - a client that could supply the name or the role
    is a client that could lie about them.
    """

    entity_id = serializers.CharField(max_length=40)
