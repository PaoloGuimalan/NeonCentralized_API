from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from messenger.models import Conversation, Message
from messenger.serializers import ConversationSerializer, MessageSerializer
from llm.models import Agent
from llm.serializers import ToolSerializer
from django.http import StreamingHttpResponse
from neon.utils.parsing_tools import stringify_json
from llm.services.groq_service import stream_groq_chat_completion


class Pagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = "page_size"


class MessagingListView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = Pagination

    def get(self, request):
        try:
            user = self.request.user

            query_set = Conversation.objects.filter(created_by=user).order_by(
                "-created_at"
            )

            paginator = self.pagination_class()
            paginated_queryset = paginator.paginate_queryset(
                query_set, request, view=self
            )

            serialized_result = ConversationSerializer(paginated_queryset, many=True)
            data = paginator.get_paginated_response(serialized_result.data)

            return data
        except Exception as ex:
            return Response(ex, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MessagingView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = Pagination

    def get(self, request, conversation_id):
        try:
            user = self.request.user

            query_set = Message.objects.filter(
                conversation_id=conversation_id
            ).order_by("-created_at")

            paginator = self.pagination_class()
            paginated_queryset = paginator.paginate_queryset(
                query_set, request, view=self
            )

            serialized_result = MessageSerializer(paginated_queryset, many=True)
            data = paginator.get_paginated_response(serialized_result.data)

            return data
        except Exception as ex:
            return Response(ex, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, conversation_id):
        try:
            user = self.request.user
            message_type = request.data.get("message_type")
            content = request.data.get("content")
            agent_uuid = request.data.get("agent_uuid")

            conversation = Conversation.objects.get(conversation_id=conversation_id)
            agent = Agent.objects.get(uuid=agent_uuid)
            tools = ToolSerializer(agent.role.tools, many=True).data

            new_message = Message(
                conversation=conversation,
                sender=user,
                agent=None,
                message_type=message_type,
                content=content,
            )

            new_message.save()

            new_message.receivers.add(user)
            new_message.seeners.add(user)

            combined = []

            def stream_response():
                try:
                    for token in stream_groq_chat_completion(
                        agent.role.system_prompt, content, tools
                    ):
                        if token is not None:
                            combined.append(token)
                            yield f'data: {stringify_json({"status": True, "token": token})}\n\n'

                    full_reply = "".join(combined)

                    ai_reply = Message(
                        conversation=conversation,
                        sender=None,
                        agent=agent,
                        message_type="ai_reply",
                        content=full_reply,
                    )

                    ai_reply.save()

                    ai_reply.receivers.add(user)
                    ai_reply.seeners.add(user)

                except Exception as ex:
                    yield f'data: {stringify_json({"status": False, "message": f"Unexpected error: {str(ex)}"})}\n\n'
                    return

            return StreamingHttpResponse(
                stream_response(),
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                },
            )

        except Exception as ex:
            return Response(ex, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
