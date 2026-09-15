from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from messenger.models import Conversation, Message, Summary
from messenger.serializers import ConversationSerializer, MessageSerializer
from llm.models import Agent, Model
from organization.tenancy import (  # noqa: F401  (re-exported, see below)
    OrganizationResolutionError,
    organization_for,
    requested_organization_id,
)
from django.http import StreamingHttpResponse
from neon.utils.parsing_tools import stringify_json
from django.db.models import Prefetch, Subquery, OuterRef
from django.db.models.functions import Coalesce
from django.shortcuts import get_object_or_404
import uuid

from llm.services.llm_factory import LLMFactory
from llm.utils.llm_response_parsing import handle_llm_response
from llm.services.rag import CustomerServiceRAG
from llm.scripts.tasks import index_chat_message_task
from user.utils.user_manipulation import get_or_create_account
from neon.permissions import IsDeveloperToken

_rag = None


def get_rag():
    """The RAG client, built on first use.

    Constructing it at module level ran CustomerServiceRAG.__init__ - and so
    Pinecone's list_indexes() - at IMPORT time. Every `manage.py` command paid
    a network round trip before doing anything, migrations included, and a
    Pinecone outage stopped Django from starting at all rather than degrading
    one feature.
    """
    global _rag
    if _rag is None:
        _rag = CustomerServiceRAG()
    return _rag


EXTERNAL_CHAT_FALLBACK_REPLY = (
    "Sorry, there is a problem processing your request. Please try again later."
)


# Tenant resolution moved to organization/tenancy.py when the platform API
# started scoping agents, roles, tools, credentials and knowledge the same way.
# Re-exported under the names this module already used, so every call site here
# is unchanged - and there is one definition of "which organization is this
# caller in" rather than two that can drift apart into a cross-tenant read.
#
# The behaviour change worth knowing about: belonging to several organizations
# is no longer automatically a conflict. A caller can name one with the
# X-Organization header, and only an unnamed ambiguous case is still a 409.


class Pagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = "page_size"


ORIGIN_PARAM = "origin"

VALID_ORIGINS = {value for value, _ in Conversation.ORIGIN_CHOICES}


class InvalidOrigin(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def requested_origins(request):
    """Which surfaces the caller wants listed, or None for all of them.

    Accepts both spellings - repeated (?origin=native&origin=external) and
    comma-separated (?origin=native,external) - because a hand-written link
    reaches for one and a form serializer for the other, and rejecting either
    would be a rule nobody can guess.

    AN UNKNOWN VALUE IS A 400, NOT "EVERYTHING"
    -------------------------------------------
    Falling back to unfiltered on a typo is the worse failure by a distance:
    the one thing this parameter exists to do is keep a caller from seeing
    threads it did not ask for, and a silent fallback shows them ALL of them -
    bot mirrors included - while reporting success.
    """
    raw = request.query_params.getlist(ORIGIN_PARAM)
    wanted = {
        value.strip() for item in raw for value in item.split(",") if value.strip()
    }
    if not wanted:
        return None

    unknown = sorted(wanted - VALID_ORIGINS)
    if unknown:
        raise InvalidOrigin(
            f"Unknown {ORIGIN_PARAM}: {', '.join(unknown)}. "
            f"One of: {', '.join(sorted(VALID_ORIGINS))}."
        )
    return wanted


class MessagingListView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = Pagination

    def get(self, request):
        # Before the broad handler below, which would turn a bad parameter into
        # a 500 that says nothing about which value was wrong.
        try:
            origins = requested_origins(request)
        except InvalidOrigin as ex:
            return Response(
                {"status": False, "message": ex.message},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = self.request.user

            latest_time_qs = (
                Message.objects.filter(conversation=OuterRef("pk"))
                .order_by("-created_at")
                .values("created_at")[:1]
            )

            query_set = (
                Conversation.objects.annotate(
                    latest_message_time=Subquery(latest_time_qs)
                )
                .prefetch_related(
                    Prefetch(
                        "message_set",
                        queryset=Message.objects.order_by("-created_at")[:1],
                        to_attr="latest_message_list",
                    )
                )
                .filter(created_by=user)
                # Most recently active first. This was ascending on
                # latest_message_time, which floated the STALEST conversations
                # to the top of the list.
                #
                # Coalesce rather than a bare "-latest_message_time" because a
                # conversation with no messages yet has a NULL there, and its
                # real last activity is when it was created - which is what
                # puts a brand-new chat at the top instead of at either
                # extreme by accident of NULL ordering.
                .order_by(
                    Coalesce("latest_message_time", "created_at").desc(),
                    "-created_at",
                )
            )

            if origins is not None:
                # Applied after ordering for readability only - a queryset is
                # lazy, so this is one WHERE on `origin`, which is indexed.
                query_set = query_set.filter(origin__in=origins)

            paginator = self.pagination_class()
            paginated_queryset = paginator.paginate_queryset(
                query_set, request, view=self
            )

            serialized_result = ConversationSerializer(
                paginated_queryset, many=True, context={"include_latest_message": True}
            )
            data = paginator.get_paginated_response(serialized_result.data)

            return data
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MessagingView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = Pagination

    def get(self, request, conversation_id):
        try:
            user = self.request.user

            try:
                organization = organization_for(
                    user, requested_organization_id(request)
                )
            except OrganizationResolutionError as ex:
                return Response(
                    {"status": False, "message": ex.message}, status=ex.status_code
                )

            # Scoped to the caller's organization. Filtering on the id alone
            # served any conversation to any authenticated account.
            query_set = Message.objects.filter(
                conversation_id=conversation_id,
                conversation__organization=organization,
            ).order_by("-created_at")

            paginator = self.pagination_class()
            paginated_queryset = paginator.paginate_queryset(
                query_set, request, view=self
            )

            serialized_result = MessageSerializer(paginated_queryset, many=True)
            data = paginator.get_paginated_response(serialized_result.data)

            return data
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, conversation_id):
        try:
            user = self.request.user
            message_type = request.data.get("message_type")
            content = request.data.get("content")
            agent_uuid = request.data.get("agent_uuid")
            model_uuid = request.data.get("model_uuid")
            pending_id = request.data.get("pending_id") or uuid.uuid4()

            try:
                organization = organization_for(
                    user, requested_organization_id(request)
                )
            except OrganizationResolutionError as ex:
                return Response(
                    {"status": False, "message": ex.message}, status=ex.status_code
                )

            # Both lookups are organization-scoped. Unscoped, these let any
            # authenticated account post into any conversation and drive any
            # organization's agent - and therefore spend its LLM credit.
            try:
                conversation = Conversation.objects.get(
                    conversation_id=conversation_id, organization=organization
                )
            except Conversation.DoesNotExist:
                return Response(
                    {"status": False, "message": "Conversation not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            try:
                agent = Agent.objects.select_related("role").get(
                    uuid=agent_uuid, organization=organization, is_active=True
                )
            except Agent.DoesNotExist:
                return Response(
                    {
                        "status": False,
                        "message": "Agent not found for this organization.",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            if agent.role is None:
                return Response(
                    {
                        "status": False,
                        "message": "Agent has no role/system prompt configured.",
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

            # Tool INSTANCES, not serialized data: execution needs the
            # credential, and the credential must never reach a serializer
            # whose output is handed to a model provider.
            tools = list(agent.role.tools.filter(is_enabled=True))

            try:
                llm_model = Model.objects.get(uuid=model_uuid)
            except Model.DoesNotExist:
                return Response(
                    {"status": False, "message": "Model not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            llm_service = LLMFactory().create(
                service=llm_model.service.name,
                api_key=conversation.organization.llm_api_key,
                model=llm_model.model,
            )

            history_query = get_rag().retrieve(
                content,
                conversation.conversation_id,
                conversation.organization_id,
                conversation.organization.llm_api_key,
                8,
                agent=agent,
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
                            pending_id=pending_id,
                        )

                        new_message.save()

                        index_chat_message_task.delay(new_message.message_id)

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

                        index_chat_message_task.delay(ai_reply.message_id)

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
                                pending_id=pending_id,
                            )

                            new_message.save()

                            index_chat_message_task.delay(new_message.message_id)

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

                            index_chat_message_task.delay(ai_reply.message_id)

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

    def get(self, request, conversation_id):
        try:
            user = self.request.user

            try:
                organization = organization_for(
                    user, requested_organization_id(request)
                )
            except OrganizationResolutionError as ex:
                return Response(
                    {"status": False, "message": ex.message}, status=ex.status_code
                )

            # Scoped to the caller's organization. This lookup was by
            # conversation_id ALONE - no organization filter and no ownership
            # check - so any authenticated account could read any
            # conversation's name and footprint by guessing or harvesting an
            # id. MessagingView was fixed for exactly this; this route was
            # missed because it only returns metadata, which is still another
            # tenant's metadata.
            # .first() rather than get_object_or_404: the method is wrapped in
            # a bare `except Exception` that turns everything into a 500, so
            # Http404 would surface as "server error" - and a 500 on a
            # conversation that simply is not yours reads as a bug in Neon
            # rather than as the access rule doing its job.
            conversation = Conversation.objects.filter(
                conversation_id=conversation_id, organization=organization
            ).first()
            if conversation is None:
                return Response(
                    {"status": False, "message": "Conversation not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            serialized_conv = ConversationSerializer(
                conversation, context={"include_latest_message": False}
            )

            return Response(
                serialized_conv.data,
                status=status.HTTP_200_OK,
            )
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        try:
            user = self.request.user
            name = request.data.get("name")
            footprint = request.data.get("footprint", None)

            try:
                organization = organization_for(
                    user, requested_organization_id(request)
                )
            except OrganizationResolutionError as ex:
                return Response(
                    {"status": False, "message": ex.message}, status=ex.status_code
                )

            if footprint is not None:
                # Only dedupe when footprint has a value
                conversation, created = Conversation.objects.get_or_create(
                    footprint=footprint,
                    defaults={
                        "organization": organization,
                        "name": name,
                        "created_by": user,
                        "origin": Conversation.ORIGIN_NATIVE,
                    },
                )
            else:
                conversation = Conversation.objects.create(
                    organization=organization,
                    name=name,
                    footprint=footprint,
                    created_by=user,
                    origin=Conversation.ORIGIN_NATIVE,
                )

            return Response(
                {"conversation_id": conversation.conversation_id},
                status=status.HTTP_200_OK,
            )
        except Exception as ex:
            return Response(str(ex), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ExternalChatView(APIView):
    """
    Server-to-server chat endpoint for third-party apps, authenticated via
    x-developer-token. External callers own their own conversation UI/state
    and only get back the AI's reply; Neon still records the interaction
    internally (tied to the real person by email) so it shows up as a normal
    conversation on Neon's own native frontend.
    """

    permission_classes = [IsDeveloperToken]

    def post(self, request):
        try:
            email = request.data.get("email")
            first_name = request.data.get("first_name")
            last_name = request.data.get("last_name")
            external_conversation_id = request.data.get("external_conversation_id")
            agent_uuid = request.data.get("agent_uuid")
            model_uuid = request.data.get("model_uuid")
            content = request.data.get("content")

            required = {
                "email": email,
                "external_conversation_id": external_conversation_id,
                "agent_uuid": agent_uuid,
                "model_uuid": model_uuid,
                "content": content,
            }
            missing = [key for key, value in required.items() if not value]
            if missing:
                return Response(
                    {
                        "status": False,
                        "message": f"Missing required field(s): {', '.join(missing)}",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Belonging to several organizations became a normal state once
            # the API could create them, so this resolves through the shared
            # helper - which lets an integration whose owner has more than one
            # name the right one with the X-Organization header instead of
            # being permanently 409'd.
            try:
                organization = organization_for(
                    request.user, requested_organization_id(request)
                )
            except OrganizationResolutionError as ex:
                return Response(
                    {"status": False, "message": ex.message}, status=ex.status_code
                )

            end_user, _ = get_or_create_account(email, first_name, last_name)

            # Namespace the footprint per-organization AND per-person:
            # Conversation.footprint is globally unique, so without this,
            # two orgs reusing the same external conversation id (or two
            # different end-users of the SAME external app both using a
            # non-globally-unique id, e.g. each user's own "conversation 1")
            # would collide onto the same Neon conversation and leak each
            # other's chat history.
            footprint = f"{organization.id}:{email}:{external_conversation_id}"
            conversation, _ = Conversation.objects.get_or_create(
                footprint=footprint,
                defaults={
                    "organization": organization,
                    "name": external_conversation_id,
                    "created_by": end_user,
                    # Recorded so the conversation list can say this thread
                    # happened in somebody else's app. The messages carry
                    # `integration` for WHICH app; this is the row-level answer
                    # to "was this ours", which a conversation with no messages
                    # yet would otherwise have no way to give.
                    "origin": Conversation.ORIGIN_EXTERNAL,
                },
            )
            if conversation.organization_id != organization.id:
                return Response(
                    {"status": False, "message": "Conversation identifier conflict."},
                    status=status.HTTP_409_CONFLICT,
                )

            try:
                agent = Agent.objects.select_related("role").get(
                    uuid=agent_uuid, organization=organization, is_active=True
                )
            except Agent.DoesNotExist:
                return Response(
                    {
                        "status": False,
                        "message": "Agent not found for this organization.",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            if agent.role is None:
                return Response(
                    {
                        "status": False,
                        "message": "Agent has no role/system prompt configured.",
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

            try:
                llm_model = Model.objects.get(uuid=model_uuid)
            except Model.DoesNotExist:
                return Response(
                    {"status": False, "message": "Model not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            if not organization.llm_api_key:
                return Response(
                    {
                        "status": False,
                        "message": "Organization has no LLM API key configured.",
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

            llm_service = LLMFactory().create(
                service=llm_model.service.name,
                api_key=organization.llm_api_key,
                model=llm_model.model,
            )
            if llm_service is None:
                return Response(
                    {
                        "status": False,
                        "message": "Unsupported LLM service configured for this model.",
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

            # Instances, not serialized data - see MessagingView.post.
            tools = list(agent.role.tools.filter(is_enabled=True))

            history_query = get_rag().retrieve(
                content,
                conversation.conversation_id,
                conversation.organization_id,
                organization.llm_api_key,
                8,
                agent=agent,
            )
            history = [
                {
                    "role": ("user" if msg["msg_type"] == "text" else "assistant"),
                    "content": f'History: {msg["text"]}',
                }
                for msg in history_query
            ]

            full_reply = None
            attempts = 0
            max_retries = 3

            while attempts < max_retries:
                try:
                    combined = []
                    for token in llm_service.stream_chat_completion(
                        history, agent.role.system_prompt, content, tools
                    ):
                        if token is not None:
                            combined.append(token)
                    full_reply = "".join(combined)
                    break
                except Exception:
                    attempts += 1

            user_message = Message.objects.create(
                conversation=conversation,
                sender=end_user,
                agent=None,
                integration=request.auth,
                message_type="text",
                content=content,
            )
            index_chat_message_task.delay(user_message.message_id)
            user_message.receivers.add(end_user)
            user_message.seeners.add(end_user)

            reply_content = (
                handle_llm_response(full_reply)
                if full_reply is not None
                else EXTERNAL_CHAT_FALLBACK_REPLY
            )

            ai_reply = Message.objects.create(
                conversation=conversation,
                sender=None,
                agent=agent,
                integration=request.auth,
                message_type="ai_reply",
                content=reply_content,
            )
            index_chat_message_task.delay(ai_reply.message_id)
            ai_reply.receivers.add(end_user)
            ai_reply.seeners.add(end_user)

            return Response(
                {
                    "conversation_id": str(conversation.conversation_id),
                    "reply": reply_content,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as ex:
            return Response(
                {"status": False, "message": str(ex)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
