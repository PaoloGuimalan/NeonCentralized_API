"""Which organization a request acts in.

WHY THIS LIVES HERE
-------------------
Every list and every detail lookup in the platform API is scoped by this. It
started life inside `messenger/views.py` because that was the only module that
needed it; with agents, roles, tools, credentials and knowledge all scoped the
same way, a second copy would be a second thing to keep correct. Tenant
resolution is exactly the kind of code where a copy that drifts becomes a
cross-tenant read, so there is one of it, next to the `Member` table it reads.

`messenger.views` imports from here, keeping the names it already used.

CHOOSING, WHEN THERE IS MORE THAN ONE
-------------------------------------
Before this phase there was no way to create an organization outside the admin,
so "the caller's organization" could be resolved by simply looking for their
single `Member` row - and resolving to more than one was treated as a conflict
to report. Now that the API can create organizations, belonging to several is
a normal state, not an error, so the caller may name one:

    X-Organization: <organization id>      (header, preferred)
    ?organization=<organization id>        (query string, for links)

Naming one the caller is not a member of is a 403 and NOT a 404: a 404 would
say "no such organization", which tells an outsider whether an id is real.

With nothing named, a sole membership still resolves - so every existing
caller and every single-org user is unaffected - and an ambiguous one is a 409
that names the candidates, because the server picking silently would mean the
same request writing into different tenants depending on join order.
"""

from rest_framework import status

from .models import Member


class OrganizationResolutionError(Exception):
    """The caller's organization could not be determined unambiguously."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


ORGANIZATION_HEADER = "X-Organization"
ORGANIZATION_PARAM = "organization"


def memberships(account):
    return Member.objects.select_related("organization").filter(account=account)


def organization_for(account, requested_id=None):
    """The organization `account` acts in, optionally the one it asked for."""
    rows = memberships(account)

    if requested_id:
        member = rows.filter(organization_id=requested_id).first()
        if member is None:
            raise OrganizationResolutionError(
                "You are not a member of that organization.",
                status.HTTP_403_FORBIDDEN,
            )
        return member.organization

    found = list(rows[:2])
    if not found:
        raise OrganizationResolutionError(
            "This account is not associated with any organization.",
            status.HTTP_403_FORBIDDEN,
        )
    if len(found) == 1:
        return found[0].organization

    names = ", ".join(
        f"{m.organization.id} ({m.organization.name})" for m in memberships(account)
    )
    raise OrganizationResolutionError(
        "This account belongs to more than one organization, so the one to act "
        f"in must be named with the {ORGANIZATION_HEADER} header. One of: {names}",
        status.HTTP_409_CONFLICT,
    )


def requested_organization_id(request):
    """The organization the caller named, if any."""
    return (
        request.headers.get(ORGANIZATION_HEADER)
        or request.query_params.get(ORGANIZATION_PARAM)
        or None
    )


def organization_from_request(request):
    """`organization_for`, reading the caller's choice off the request."""
    return organization_for(request.user, requested_organization_id(request))


def is_owner(account, organization):
    """Whether `account` may change the organization itself.

    `Member` carries no role column, so the only ownership the schema can
    express today is `Organization.created_by`. That is deliberately narrow:
    it means exactly one person can rename an organization, manage its members
    and manage its provider credentials.

    A `Member.role` of owner/admin/member is the obvious next step and would
    replace this function without changing its callers. It is not here because
    it is a schema change nobody has asked for yet, and guessing at a role
    model is how you end up migrating one twice.
    """
    return str(organization.created_by_id) == str(account.pk)
