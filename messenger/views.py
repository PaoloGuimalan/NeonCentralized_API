from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from messenger.models import Conversation, Message, Summary
from messenger.serializers import ConversationSerializer, MessageSerializer
from llm.models import Agent
from llm.serializers import ToolSerializer
from django.http import StreamingHttpResponse
from neon.utils.parsing_tools import stringify_json
from llm.services.groq_service import stream_groq_chat_completion, summarize_messages


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

                        cumulative_summary = summarize_messages(
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

                if messages.count() == batch_size:
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

                    raw_summary = summarize_messages(to_summarize)

                    Summary.objects.create(
                        conversation=conversation, context=raw_summary, range=6
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
                        history, agent.role.system_prompt, content, tools
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
