from rest_framework.permissions import BasePermission
from user.models import Token


class IsDeveloperToken(BasePermission):
    """Only allow requests authenticated via x-developer-token (not JWT).

    neon.backends.AutheticationBackend sets request.auth to the Token
    instance itself for this path (and to "jwt" for x-access-token), so
    this also lets views identify exactly which integration authenticated.
    """

    message = "This endpoint requires x-developer-token authentication."

    def has_permission(self, request, view):
        return isinstance(request.auth, Token)
