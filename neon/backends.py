from django.contrib.auth.backends import BaseBackend
from user.models import Account, Token
from .utils.jwt_tools import JWTTools

jwt = JWTTools


class AutheticationBackend(BaseBackend):

    def authenticate(self, request):
        try:
            auth_token = request.headers.get("x-access-token")
            token = request.headers.get("x-developer-token")

            if auth_token:
                decoded_header = jwt.decoder(auth_token)
                decoded_id = decoded_header["userID"]

                user = Account.objects.get(username=decoded_id)
                return (user, True)
            elif token:
                loaded_token = Token.objects.get(token=token)
                user = Account.objects.get(username=loaded_token.account.username)
                return (user, True)
        except Account.DoesNotExist:
            return None
        except Token.DoesNotExist:
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
