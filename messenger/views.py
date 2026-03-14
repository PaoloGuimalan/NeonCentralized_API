from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from messenger.models import Conversation, Message, Summary
from messenger.serializers import ConversationSerializer, MessageSerializer
from llm.models import Agent, Model
from organization.models import Member
from llm.serializers import ToolSerializer
from django.http import StreamingHttpResponse
from neon.utils.parsing_tools import stringify_json
import uuid

# from llm.services.groq_service import GroqService
from llm.services.llm_factory import LLMFactory
from llm.utils.llm_response_parsing import handle_llm_response
from llm.services.rag import CustomerServiceRAG
from llm.scripts.tasks import index_chat_message_task

rag = CustomerServiceRAG()


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
            model_uuid = request.data.get("model_uuid")

            conversation = Conversation.objects.get(conversation_id=conversation_id)
            agent = Agent.objects.get(uuid=agent_uuid)
            enabled_tools_qs = agent.role.tools.filter(is_enabled=True)
            tools = ToolSerializer(enabled_tools_qs, many=True).data
            llm_model = Model.objects.get(uuid=model_uuid)

            llm_service = LLMFactory().create(
                service=llm_model.service.name,
                api_key=conversation.organization.llm_api_key,
                model=llm_model.model,
            )

            history_query = rag.retrieve(
                content,
                conversation.conversation_id,
                conversation.organization.llm_api_key,
                8,
            )
            history = []

            for msg in history_query:
                history.append(
                    {
                        "role": ("user" if msg["msg_type"] == "text" else "assistant"),
                        "content": f'History: {msg["text"]}',
                    }
                )

            def stream_response():
                max_retries = 3
                attempts = 0

                while attempts < max_retries:
                    try:
                        combined = []
                        for token in llm_service.stream_chat_completion(
                            history, agent.role.system_prompt, content, tools
                        ):
                            if token is not None:
                                combined.append(token)
                                yield f'data: {stringify_json({"status": True, "token": token})}\n\n'

                        new_message = Message(
                            conversation=conversation,
                            sender=user,
                            agent=None,
                            message_type=message_type,
                            content=content,
                        )

                        new_message.save()

                        index_chat_message_task.delay(
                            new_message.message_id,
                            conversation.organization.llm_api_key,
                        )

                        new_message.receivers.add(user)
                        new_message.seeners.add(user)

                        full_reply = "".join(combined)

                        ai_reply = Message(
                            conversation=conversation,
                            sender=None,
                            agent=agent,
                            message_type="ai_reply",
                            content=handle_llm_response(full_reply),
                        )
                        ai_reply.save()

                        index_chat_message_task.delay(
                            ai_reply.message_id, conversation.organization.llm_api_key
                        )

                        ai_reply.receivers.add(user)
                        ai_reply.seeners.add(user)

                        # Exit if successful
                        return

                    except Exception as ex:
                        attempts += 1
                        # Provide immediate feedback of failure to client
                        yield f'data: {stringify_json({"status": False, "message": f"Attempt {attempts} failed: {str(ex)}"})}\n\n'
                        if attempts >= max_retries:
                            new_message = Message(
                                conversation=conversation,
                                sender=user,
                                agent=None,
                                message_type=message_type,
                                content=content,
                            )

                            new_message.save()

                            index_chat_message_task.delay(
                                new_message.message_id,
                                conversation.organization.llm_api_key,
                            )

                            new_message.receivers.add(user)
                            new_message.seeners.add(user)

                            # Generate a final message using LLM to inform user there's a persistent problem
                            error_message = "Sorry, there is a problem processing your request. Please try again later."

                            ai_reply = Message(
                                conversation=conversation,
                                sender=None,
                                agent=agent,
                                message_type="ai_reply",
                                content=error_message,
                            )
                            ai_reply.save()

                            index_chat_message_task.delay(
                                ai_reply.message_id,
                                conversation.organization.llm_api_key,
                            )

                            ai_reply.receivers.add(user)
                            ai_reply.seeners.add(user)

                            yield f'data: {stringify_json({"status": False, "token": error_message})}\n\n'
                            return

            return StreamingHttpResponse(
                stream_response(),
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                },
            )

        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConversationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            user = self.request.user
            name = request.data.get("name")
            footprint = request.data.get("footprint", None)

            member = Member.objects.get(account=user)

            if footprint is not None:
                # Only dedupe when footprint has a value
                conversation, created = Conversation.objects.get_or_create(
                    footprint=footprint,
                    defaults={
                        "organization": member.organization,
                        "name": name,
                        "created_by": member.account,
                    },
                )
            else:
                conversation = Conversation.objects.create(
                    organization=member.organization,
                    name=name,
                    footprint=footprint,
                    created_by=member.account,
                )

            return Response(
                {"conversation_id": conversation.conversation_id},
                status=status.HTTP_200_OK,
            )
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
