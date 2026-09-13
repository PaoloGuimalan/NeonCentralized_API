"""Fixtures shared by the API tests.

Here rather than duplicated per app because both `llm` and `organization`
need the same thing: two organizations with different owners, so that every
"does this leak across tenants?" test has a second tenant to leak into. A
scoping test with only one organization in the database cannot fail.
"""

from django.utils.timezone import now
from rest_framework.test import APIClient

from organization.models import Member, Organization
from user.models import Account


def make_account(username, email=None, entity_id=None):
    return Account.objects.create(
        username=username,
        first_name=username.title(),
        last_name="Tester",
        email=email or f"{username}@example.com",
        entity_id=entity_id,
    )


def make_organization(owner, name, slug):
    organization = Organization.objects.create(
        name=name, slug=slug, created_by=owner
    )
    Member.objects.create(
        account=owner,
        organization=organization,
        added_by=owner,
        date_joined=now(),
    )
    return organization


def add_member(organization, account, added_by):
    return Member.objects.create(
        account=account,
        organization=organization,
        added_by=added_by,
        date_joined=now(),
    )


def client_for(account, organization=None):
    """An authenticated client, optionally pinned to one organization.

    `force_authenticate` rather than a real JWT: these tests are about
    authorization, and going through neon.backends would mean every one of
    them also testing token decoding.
    """
    client = APIClient()
    client.force_authenticate(user=account)
    if organization is not None:
        client.credentials(HTTP_X_ORGANIZATION=str(organization.pk))
    return client
