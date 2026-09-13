from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.services.chatterloop_identity import (
    ChatterloopAuthError,
    ChatterloopUnavailable,
    sign_in_with_google,
    sign_in_with_password,
)
from neon.utils.jwt_tools import JWTTools

from .models import Account
from .serializers import AccountSerializer

jwt = JWTTools


class Pagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = "page_size"


def _session_response(user):
    """The token pair Neon's frontend expects, unchanged from before.

    Neon issues its OWN session rather than passing chatterloop's through.
    That is what keeps a chatterloop outage from signing everybody out: it
    blocks new sign-ins and leaves existing ones alone.
    """
    serialized_user = AccountSerializer(user)
    return {
        "status": True,
        "result": {
            "usertoken": jwt.encoder(serialized_user.data),
            "authtoken": jwt.encoder(
                {"userID": user.username, "username": user.username}
            ),
        },
    }


def _auth_error_response(ex):
    return Response(
        {"status": False, "message": ex.message},
        status=ex.status_code,
    )


def _unavailable_response(ex):
    # 503, not 401. Telling somebody their password is wrong when the real
    # problem is an unreachable service sends them to reset a working
    # credential.
    return Response(
        {"status": False, "message": str(ex)},
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


class UserAuthentication(APIView):
    """Sign in, by forwarding to chatterloop.

    Neon no longer stores or checks passwords. Chatterloop owns credentials,
    verification and account standing; this view hands the credential over,
    mirrors the resulting identity into Neon's own Account row, and issues a
    Neon session. See core/services/chatterloop_identity.py for why forwarding
    beats reading the password hash directly, given Neon has the database
    access to do either.
    """

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [AllowAny()]

    def get(self, request, username=None):
        user = get_object_or_404(Account, username=username)
        return Response(AccountSerializer(user).data, status=status.HTTP_200_OK)

    def post(self, request):
        email_username = request.data.get("email_username")
        password = request.data.get("password")

        if not email_username or not password:
            return Response(
                {
                    "status": False,
                    "message": "Chatterloop email/username and password are required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # The credential is handed straight to the forwarder and never stored,
        # logged or echoed. Note that DRF's own exception reporting can render
        # request bodies when DEBUG is on, so a production deployment must keep
        # DEBUG off - which is already how neon/settings.py reads it.
        try:
            user = sign_in_with_password(email_username, password)
        except ChatterloopAuthError as ex:
            return _auth_error_response(ex)
        except ChatterloopUnavailable as ex:
            return _unavailable_response(ex)

        return Response(_session_response(user), status=status.HTTP_200_OK)


class ThirdPartyAuthentication(APIView):
    """Sign in with Google, by forwarding to chatterloop's own Google path.

    Neon's former TPAuthentication table is gone: the `azp` check that used to
    happen here now happens on chatterloop, against its own registry. Neon's
    Google client id therefore has to be registered in chatterloop's
    `core_tpauthentication`, or every Google sign-in fails.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        id_token = request.data.get("token")

        if not id_token:
            return Response(
                {"status": False, "message": "A Google credential is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = sign_in_with_google(id_token)
        except ChatterloopAuthError as ex:
            return _auth_error_response(ex)
        except ChatterloopUnavailable as ex:
            return _unavailable_response(ex)

        return Response(_session_response(user), status=status.HTTP_200_OK)
