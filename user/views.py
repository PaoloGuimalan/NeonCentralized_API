from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.db.models import (
    Q,
)
from .models import Account
from core.models import TPAuthentication
from .utils.user_manipulation import create_user
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

            if not email_username:
                return Response(
                    {
                        "status": False,
                        "message": "Email is missing",
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            user = Account.objects.get(
                Q(email=email_username) | Q(username=email_username)
            )

            if user:
                if user.join_type == "system":
                    if not password:
                        return Response(
                            {
                                "status": False,
                                "message": "Password is missing",
                            },
                            status=status.HTTP_401_UNAUTHORIZED,
                        )

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


class ThirdPartyAuthentication(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            token = request.data.get("token")

            decoded_token = JWTTools.decoder(token, options={"verify_signature": False})

            if decoded_token:
                authorization_token = decoded_token["azp"]

                tp_check_query = TPAuthentication.objects.get(
                    service_id=authorization_token
                )

                if tp_check_query:
                    email = decoded_token["email"]
                    user = Account.objects.filter(email=email)

                    if len(user) > 0:
                        user = user[0]
                        serialized_user = AccountSerializer(user)

                        return Response(
                            {
                                "status": True,
                                "result": {
                                    "usertoken": jwt.encoder(serialized_user.data),
                                    "authtoken": jwt.encoder(
                                        {
                                            "userID": user.username,
                                            "username": user.username,
                                        }
                                    ),
                                },
                            },
                            status=status.HTTP_200_OK,
                        )
                    else:
                        # Automatic registration

                        first_name = decoded_token["given_name"]
                        middle_name = "N/A"
                        last_name = decoded_token.get("family_name", None)
                        email = decoded_token["email"]

                        if last_name is None:
                            split_name = first_name.split(" ")

                            first_name = split_name[0]
                            last_name = split_name[1]

                        create_user_query = create_user(
                            first_name,
                            middle_name,
                            last_name,
                            email,
                            email,
                            None,
                            None,
                            None,
                            None,
                            "google",
                        )

                        if create_user_query:
                            serialized_user = AccountSerializer(create_user_query)

                            return Response(
                                {
                                    "status": True,
                                    "result": {
                                        "usertoken": jwt.encoder(serialized_user.data),
                                        "authtoken": jwt.encoder(
                                            {
                                                "userID": create_user_query.username,
                                                "username": create_user_query.username,
                                            }
                                        ),
                                    },
                                },
                                status=status.HTTP_200_OK,
                            )

                return Response(
                    {"status": False, "message": f"User cannot be logged in"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            return Response(
                {"status": False, "message": f"Cannot proceed with login"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        except Exception as e:
            print(str(e))
            return Response(
                {"status": False, "message": f"{e}"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
