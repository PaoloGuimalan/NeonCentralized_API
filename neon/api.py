"""Shared base classes for the platform API.

The point of these is that org scoping is not something each view remembers to
do. `OrganizationScopedView.organization` raises if it cannot resolve one, and
`handle_exception` turns that into the right status code - so a view that
forgets to scope a queryset is a visible mistake rather than a silent
cross-tenant read, and a view that scopes it gets the 403/409 wording for free.

Responses keep the `{"status": ..., "message": ...}` envelope the existing
`core` and `messenger` views already return, so the frontend has one shape to
handle rather than two.
"""

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from organization.tenancy import (
    OrganizationResolutionError,
    is_owner,
    organization_from_request,
)


def ok(payload=None, code=status.HTTP_200_OK, **extra):
    body = {"status": True}
    if payload is not None:
        body["data"] = payload
    body.update(extra)
    return Response(body, status=code)


def fail(message, code=status.HTTP_400_BAD_REQUEST, **extra):
    body = {"status": False, "message": message}
    body.update(extra)
    return Response(body, status=code)


class OrganizationScopedView(APIView):
    """A view that acts inside exactly one organization."""

    permission_classes = [IsAuthenticated]

    @property
    def organization(self):
        """The organization this request acts in.

        Cached per request: resolving hits `Member`, and a view that reads it
        in both `get_queryset` and `post` should not pay for it twice.
        """
        if not hasattr(self, "_organization"):
            self._organization = organization_from_request(self.request)
        return self._organization

    def require_owner(self):
        """Raise unless the caller may change the organization itself.

        See `organization.tenancy.is_owner` for how narrow "owner" currently
        is and why.
        """
        if not is_owner(self.request.user, self.organization):
            raise OrganizationResolutionError(
                "Only the organization's owner can do that.",
                status.HTTP_403_FORBIDDEN,
            )

    def handle_exception(self, exc):
        if isinstance(exc, OrganizationResolutionError):
            return fail(exc.message, exc.status_code)
        return super().handle_exception(exc)
