from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Message
        fields = "__all__"


class ConversationSerializer(serializers.ModelSerializer):
    latest_message = serializers.SerializerMethodField()

    # `origin` itself is already in `__all__` as its raw value, which is what a
    # client should branch on. These two are for rendering:
    #
    #   origin_label - the human wording, from ORIGIN_CHOICES, so the platform
    #                  and the frontend cannot disagree about what "external"
    #                  is called in front of a user.
    #   is_external  - the one bit the list really draws on, so grouping the
    #                  two integrations does not mean every client hardcoding
    #                  which values are "not native".
    origin_label = serializers.CharField(source="get_origin_display", read_only=True)
    is_external = serializers.BooleanField(read_only=True)

    class Meta:
        model = Conversation
        fields = "__all__"

    def get_latest_message(self, obj):
        if self.context.get("include_latest_message", False):
            latest = obj.latest_message_list[0] if obj.latest_message_list else None
            return MessageSerializer(latest).data if latest else None

        return None
