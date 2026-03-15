from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Message
        fields = "__all__"


class ConversationSerializer(serializers.ModelSerializer):
    latest_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = "__all__"

    def get_latest_message(self, obj):
        latest = obj.latest_message_list[0] if obj.latest_message_list else None
        return MessageSerializer(latest).data if latest else None
