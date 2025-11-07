from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.db.models import (
    Q,
)
from .models import Account
from .serializers import (
    AccountSerializer,
)
from neon.utils.jwt_tools import JWTTools
from neon.utils.generators import generate_random_digit
from rest_framework.pagination import PageNumberPagination
from datetime import datetime
from django.shortcuts import get_object_or_404
from django.utils.timezone import localtime
import bcrypt

jwt = JWTTools


class Pagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = "page_size"


class UserAuthentication(APIView):

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        elif self.request.method == "POST":
            return [AllowAny()]
        return super().get_permissions()

    def get(self, request, username=None):
        user = get_object_or_404(Account, username=username)

        # Format birthdate parts
        serialized_user = AccountSerializer(user)

        # Build response JSON matching your example
        data = serialized_user.data

        return Response(data, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            email_username = request.data.get("email_username")
            password = request.data.get("password")

            if not email_username or not password:
                return Response(
                    {
                        "status": False,
                        "message": "Email or Password is missing",
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            user = Account.objects.get(
                Q(email=email_username) | Q(username=email_username)
            )

            if user:
                hashed = user.password.encode("utf-8")
                bytes_password = password.encode("utf-8")
                is_correct = bcrypt.checkpw(bytes_password, hashed)

                if is_correct:

                    serialized_user = AccountSerializer(user)

                    return Response(
                        {
                            "status": True,
                            "result": {
                                "usertoken": jwt.encoder(serialized_user.data),
                                "authtoken": jwt.encoder(
                                    {"userID": user.username, "username": user.username}
                                ),
                            },
                        },
                        status=status.HTTP_200_OK,
                    )

                return Response(
                    {
                        "status": False,
                        "message": "Incorrect email, username, or password",
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )
            else:
                return Response(
                    {
                        "status": False,
                        "message": "Incorrect email, username, or password",
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )
        except Exception as e:
            return Response(
                {"status": False, "message": f"{e}"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
