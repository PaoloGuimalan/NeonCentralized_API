from django.contrib.auth.backends import BaseBackend
from user.models import Account
from .utils.jwt_tools import JWTTools

jwt = JWTTools


class AutheticationBackend(BaseBackend):

    def authenticate(self, request):
        try:
            token = request.headers.get("x-access-token")

            if not token:
                return None

            decoded_header = jwt.decoder(token)
            decoded_id = decoded_header["userID"]

            user = Account.objects.get(username=decoded_id)
            return (user, True)
        except Account.DoesNotExist:
            return None
        except:
            return None

    def get_user(self, user_id):
        try:
            return Account.objects.get(pk=user_id)
        except Account.DoesNotExist:
            return None

    def authenticate_header(self, request):
        return "Bearer"
