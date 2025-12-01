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

            history = []

            batch_size = 20

            try:
                messages = Message.objects.filter(conversation=conversation).order_by(
                    "created_at"
                )
                serialized_messages = MessageSerializer(messages, many=True).data

                # Prepare messages for summarization
                def build_messages_for_batch(batch):
                    return [
                        {
                            "role": (
                                "user" if msg["message_type"] == "text" else "assistant"
                            ),
                            "content": f"History: {msg['content']}",
                        }
                        for msg in batch
                    ]

                # Get existing summary range if any
                summary = Summary.objects.get(conversation=conversation)
                last_range = summary.range

                total_messages = len(serialized_messages)

                # Only summarize if total messages >= last range + batch_size
                if total_messages - last_range >= batch_size or last_range == 0:
                    batches = [
                        serialized_messages[i : i + batch_size]
                        for i in range(0, total_messages, batch_size)
                    ]

                    cumulative_summary = ""
                    for batch in batches:
                        batch_messages = build_messages_for_batch(batch)
                        to_summarize = []
                        if cumulative_summary:
                            to_summarize.append(
                                {
                                    "role": "system",
                                    "content": f"Previous summary: {cumulative_summary}",
                                }
                            )
                        to_summarize.extend(batch_messages)

                        cumulative_summary = llm_service.summarize_messages(
                            to_summarize
                        )  # your LLM summarizer

                    # Update summary and range
                    Summary.objects.update_or_create(
                        conversation=conversation,
                        defaults={
                            "context": cumulative_summary,
                            "range": total_messages,
                        },
                    )

                # Build final history with summary and leftover
                history = [
                    {
                        "role": "system",
                        "content": f"Summary so far: {summary.context if summary else ''}",
                    }
                ]

                leftover_start = last_range if last_range else 0
                leftover_messages = serialized_messages[leftover_start:]
                history.extend(build_messages_for_batch(leftover_messages))

            except Summary.DoesNotExist:
                to_summarize = []
                messages = Message.objects.filter(conversation=conversation)
                serialized_messages = MessageSerializer(messages, many=True).data

                if messages.count() >= batch_size:
                    for msg in serialized_messages:
                        to_summarize.append(
                            {
                                "role": (
                                    "user"
                                    if msg["message_type"] == "text"
                                    else "assistant"
                                ),
                                "content": f'History: {msg["content"]}',
                            }
                        )
                        history.append(
                            {
                                "role": (
                                    "user"
                                    if msg["message_type"] == "text"
                                    else "assistant"
                                ),
                                "content": f'History: {msg["content"]}',
                            }
                        )

                    raw_summary = llm_service.summarize_messages(to_summarize)

                    Summary.objects.create(
                        conversation=conversation, context=raw_summary, range=batch_size
                    )

                else:
                    for msg in serialized_messages:
                        history.append(
                            {
                                "role": (
                                    "user"
                                    if msg["message_type"] == "text"
                                    else "assistant"
                                ),
                                "content": f'History: {msg["content"]}',
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
            return Response(ex, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class ConversationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            user = self.request.user
            name = request.data.get("name")
            footprint = request.data.get("footprint", None)

            member = Member.objects.get(account=user)

            query_set = Conversation.objects.create(
                organization=member.organization,
                name=name,
                footprint=footprint,
                created_by=member.account
            )

            return Response({ "conversation_id": query_set.conversation_id }, status=status.HTTP_200_OK)
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
