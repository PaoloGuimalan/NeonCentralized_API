"""Connected identities: what the signed-in user may publish bots as."""

import logging

from django.db import IntegrityError
from django.utils.timezone import now
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from chatterloop.models import Realm

from .models import ConnectedAccount
from .serializers import (
    AvailablePageSerializer,
    ConnectedAccountSerializer,
    ConnectPageSerializer,
)
from .services.chatterloop_identity import available_pages, may_act_as

logger = logging.getLogger(__name__)


def _requires_chatterloop(user):
    """Every route here needs the caller's own chatterloop entity.

    An account with no `entity_id` is an auto-provisioned end-user row from
    ExternalChatView, which cannot sign in - so in practice this guards
    against a stale session rather than a live caller, and says so rather
    than failing further down with an empty page list.
    """
    if not user.entity_id:
        return Response(
            {
                "status": False,
                "message": "This account is not linked to a Chatterloop identity.",
            },
            status=status.HTTP_403_FORBIDDEN,
        )
    return None


class ConnectionsView(APIView):
    """The identities this account can act as."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        guard = _requires_chatterloop(user)
        if guard is not None:
            return guard

        connections = ConnectedAccount.objects.filter(
            account=user, is_active=True
        ).order_by("connected_at")

        return Response(
            {
                "status": True,
                # The personal identity is NOT a ConnectedAccount row - it is
                # established by signing in and lives on the Account itself.
                # Returned alongside so the UI can render one list without
                # having to know that.
                "personal": {
                    "entity_id": user.entity_id,
                    "external_type": "user",
                    "external_username": user.username,
                    "external_name": f"{user.first_name} {user.last_name}".strip(),
                    "external_profile": user.profile,
                },
                "connections": ConnectedAccountSerializer(connections, many=True).data,
            },
            status=status.HTTP_200_OK,
        )


class AvailablePagesView(APIView):
    """Chatterloop pages the user could connect but has not yet."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        guard = _requires_chatterloop(user)
        if guard is not None:
            return guard

        try:
            pages = available_pages(user.entity_id)
        except Exception:
            logger.exception("could not read chatterloop pages")
            return Response(
                {
                    "status": False,
                    "message": "Could not reach Chatterloop to list your pages.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        already = set(
            ConnectedAccount.objects.filter(
                account=user,
                provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
                is_active=True,
            ).values_list("external_id", flat=True)
        )
        pages = [p for p in pages if p["entity_id"] not in already]

        return Response(
            {"status": True, "pages": AvailablePageSerializer(pages, many=True).data},
            status=status.HTTP_200_OK,
        )


class ConnectPageView(APIView):
    """Connect a chatterloop page, or disconnect one."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        guard = _requires_chatterloop(user)
        if guard is not None:
            return guard

        serializer = ConnectPageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        realm_entity_id = serializer.validated_data["entity_id"]

        # Authority is decided against chatterloop's own membership table, not
        # against anything the client sent. The client names a page; whether
        # this person may speak for it is not theirs to assert.
        if not may_act_as(user.entity_id, realm_entity_id):
            return Response(
                {
                    "status": False,
                    "message": "You are not an owner or admin of that page.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            realm = Realm.objects.get(entity_id=realm_entity_id)
        except Realm.DoesNotExist:
            return Response(
                {"status": False, "message": "Page not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        existing = ConnectedAccount.objects.filter(
            provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
            external_id=realm_entity_id,
            is_active=True,
        ).first()
        if existing is not None:
            if existing.account_id == user.id:
                return Response(
                    {
                        "status": True,
                        "connection": ConnectedAccountSerializer(existing).data,
                    },
                    status=status.HTTP_200_OK,
                )
            # Claimed by somebody else. Deliberately not saying who - that
            # would leak which Neon accounts exist and what they administer.
            return Response(
                {
                    "status": False,
                    "message": "That page is already connected to another Neon account.",
                },
                status=status.HTTP_409_CONFLICT,
            )

        try:
            connection = ConnectedAccount.objects.create(
                account=user,
                provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
                external_id=realm_entity_id,
                external_type=ConnectedAccount.TYPE_REALM,
                external_username=realm.slug or "",
                external_name=realm.name,
                external_profile=realm.profile or "none",
                metadata={
                    "realm_id": realm.realm_id or realm.id,
                    # The role this was granted under, recorded so a later
                    # revocation is explainable rather than mysterious.
                    "role": next(
                        (
                            p["role"]
                            for p in available_pages(user.entity_id)
                            if p["entity_id"] == realm_entity_id
                        ),
                        "",
                    ),
                },
                last_verified_at=now(),
            )
        except IntegrityError:
            # Lost a race against a concurrent connect of the same page.
            return Response(
                {
                    "status": False,
                    "message": "That page is already connected to another Neon account.",
                },
                status=status.HTTP_409_CONFLICT,
            )

        return Response(
            {"status": True, "connection": ConnectedAccountSerializer(connection).data},
            status=status.HTTP_201_CREATED,
        )


def _bots_for(connection):
    """Bots published under this identity that are still live."""
    # Imported here rather than at module level: chatterloop.models imports
    # core.models for its ConnectedAccount relation, so a top-level import
    # would be a cycle.
    from chatterloop.models import ChatterloopBot

    return ChatterloopBot.objects.filter(
        connected_account=connection, status=ChatterloopBot.STATUS_ACTIVE
    )


def _stop_bots_for(connection, actor):
    """Deactivate every live bot on this identity and revoke its tokens.

    Returns `(stopped, failed)` as lists of handles. Each bot is stopped
    independently so one unreachable revocation does not leave the rest
    running - the failure mode to avoid is "some of them stopped and we do not
    know which".
    """
    from chatterloop.provisioning import deactivate_bot

    stopped, failed = [], []
    for bot in _bots_for(connection):
        try:
            deactivate_bot(
                bot,
                reason=f"@{actor.username} disconnected {connection.external_name}.",
            )
            stopped.append(bot.handle)
        except Exception:
            logger.exception("could not deactivate bot %s on disconnect", bot.entity_id)
            failed.append(bot.handle)
    return stopped, failed


class ConnectionDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, connection_id):
        """What disconnecting this identity would stop.

        Read before the confirmation dialog, so it can name the bots rather
        than asking "are you sure?" about a consequence the user cannot see.
        """
        connection = ConnectedAccount.objects.filter(
            id=connection_id, account=request.user, is_active=True
        ).first()
        if connection is None:
            return Response(
                {"status": False, "message": "Connection not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        bots = _bots_for(connection).values("id", "handle", "name")
        return Response(
            {
                "status": True,
                "connection": ConnectedAccountSerializer(connection).data,
                "bots": list(bots),
            },
            status=status.HTTP_200_OK,
        )

    def delete(self, request, connection_id):
        user = request.user

        connection = ConnectedAccount.objects.filter(
            id=connection_id, account=user, is_active=True
        ).first()
        if connection is None:
            return Response(
                {"status": False, "message": "Connection not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Soft delete: the row is the audit trail for any bot that was minted
        # under this identity, and hard-deleting it would leave those bots
        # pointing at an owner Neon can no longer explain. The partial unique
        # index is conditioned on is_active, so this does not block a later
        # reconnect of the same page.
        connection.is_active = False
        connection.disconnected_at = now()
        connection.save(update_fields=["is_active", "disconnected_at"])

        # A bot published under this identity keeps working unless it is
        # stopped here: its token authenticates against developer_service,
        # which knows nothing about Neon's connection rows. Leaving it live
        # would mean a bot still speaking as a page the user just unlinked.
        stopped, failed = _stop_bots_for(connection, request.user)

        if failed:
            # Reported rather than swallowed. The connection IS disconnected -
            # that part committed - but a bot whose revocation did not land is
            # still live, and the user is the only one who can escalate it.
            return Response(
                {
                    "status": True,
                    "message": (
                        "Disconnected, but "
                        + ", ".join(failed)
                        + " could not be stopped on Chatterloop. "
                        "Revoke those tokens from the Bots screen."
                    ),
                    "stopped_bots": stopped,
                    "failed_bots": failed,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "status": True,
                "message": "Disconnected.",
                "stopped_bots": stopped,
            },
            status=status.HTTP_200_OK,
        )
