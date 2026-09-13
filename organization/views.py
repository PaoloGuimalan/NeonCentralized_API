"""Organizations, their members, and their provider credentials.

Organizations were creatable only through Django admin, which made the first
step of using Neon something a staff member had to do by hand. These are the
endpoints onboarding needs: create one, see who is in it, and store the API
keys its agents run on.

Everything except organization creation is scoped by
`neon.api.OrganizationScopedView`, so a request acts in exactly one
organization and querysets are filtered by it rather than by an id the caller
supplies.
"""

import logging

from django.db import IntegrityError, transaction
from django.utils.timezone import now
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from neon.api import OrganizationScopedView, fail, ok

from .credentials import CredentialNotConfigured, embedding_api_key
from .models import Member, Organization, ProviderCredential
from .serializers import (
    MemberInviteSerializer,
    MemberSerializer,
    OrganizationCreateSerializer,
    OrganizationSerializer,
    ProviderCredentialSerializer,
)
from .tenancy import is_owner, memberships

logger = logging.getLogger(__name__)


class OrganizationListView(APIView):
    """The organizations the caller belongs to, and creating one.

    Deliberately NOT organization-scoped: this is the endpoint a user hits
    when they have no organization at all, which is every user's first
    request after signing in.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        rows = memberships(request.user).order_by("organization__created_at")
        organizations = [m.organization for m in rows]
        return ok(
            OrganizationSerializer(
                organizations, many=True, context={"account": request.user}
            ).data
        )

    def post(self, request):
        serializer = OrganizationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # One transaction: an organization with no members is unreachable
        # through every other endpoint here, including the one that would let
        # somebody add themselves to it. Half of this committing would strand
        # a row nobody can ever see or delete.
        try:
            with transaction.atomic():
                organization = serializer.save(created_by=request.user)
                Member.objects.create(
                    account=request.user,
                    organization=organization,
                    added_by=request.user,
                    date_joined=now(),
                )
        except IntegrityError:
            # The slug resolver checks for collisions, but two simultaneous
            # creates of the same name both pass that check.
            return fail(
                "That organization could not be created; try a different name.",
                status.HTTP_409_CONFLICT,
            )

        return ok(
            OrganizationSerializer(organization, context={"account": request.user}).data,
            status.HTTP_201_CREATED,
        )


class OrganizationDetailView(OrganizationScopedView):
    """Read or change the organization the caller is acting in."""

    def get(self, request):
        return ok(
            OrganizationSerializer(
                self.organization, context={"account": request.user}
            ).data
        )

    def patch(self, request):
        self.require_owner()

        serializer = OrganizationSerializer(
            self.organization,
            data=request.data,
            partial=True,
            context={"account": request.user},
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return ok(serializer.data)


class MemberListView(OrganizationScopedView):
    """Who is in the organization, and adding somebody."""

    def get(self, request):
        rows = (
            Member.objects.filter(organization=self.organization)
            .select_related("account", "organization")
            .order_by("date_joined")
        )
        return ok(MemberSerializer(rows, many=True).data)

    def post(self, request):
        self.require_owner()

        serializer = MemberInviteSerializer(data=request.data, context={})
        serializer.is_valid(raise_exception=True)
        account = serializer.context["account"]

        existing = Member.objects.filter(
            organization=self.organization, account=account
        ).first()
        if existing is not None:
            return fail(
                "That account is already a member.", status.HTTP_409_CONFLICT
            )

        member = Member.objects.create(
            account=account,
            organization=self.organization,
            added_by=request.user,
            nickname=serializer.validated_data.get("nickname") or None,
            date_joined=now(),
        )
        member = (
            Member.objects.select_related("account", "organization")
            .get(pk=member.pk)
        )
        return ok(MemberSerializer(member).data, status.HTTP_201_CREATED)


class MemberDetailView(OrganizationScopedView):

    def _member(self, member_id):
        return (
            Member.objects.select_related("account", "organization")
            .filter(id=member_id, organization=self.organization)
            .first()
        )

    def patch(self, request, member_id):
        member = self._member(member_id)
        if member is None:
            return fail("Member not found.", status.HTTP_404_NOT_FOUND)

        # A member may set their own nickname; changing anyone else's is the
        # owner's business.
        if str(member.account_id) != str(request.user.pk):
            self.require_owner()

        serializer = MemberSerializer(member, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return ok(serializer.data)

    def delete(self, request, member_id):
        member = self._member(member_id)
        if member is None:
            return fail("Member not found.", status.HTTP_404_NOT_FOUND)

        # Leaving is always allowed; removing somebody else is the owner's.
        leaving = str(member.account_id) == str(request.user.pk)
        if not leaving:
            self.require_owner()

        # The owner cannot leave. `Organization.created_by` is the only thing
        # expressing ownership, so an owner who left would leave an
        # organization nobody can ever administer again - no member could add
        # members, rotate credentials or rename it.
        if is_owner(member.account, self.organization):
            return fail(
                "The organization's owner cannot be removed. Transfer "
                "ownership first.",
                status.HTTP_409_CONFLICT,
            )

        member.delete()
        return ok(None, message="Removed.")


class EmbeddingDefaultMixin:
    """Shared by the credential list and detail views.

    A mixin rather than the detail view subclassing the list view: inheriting
    would also inherit `post(self, request)`, which the detail route calls with
    a `credential_id` it cannot accept.
    """

    def _clear_embedding_default(self, validated_data, exclude_pk=None):
        """Make room for a new embedding default.

        There is a partial unique constraint allowing one per organization, so
        without this, flagging a second credential is an IntegrityError the
        user reads as "something went wrong" rather than the intended "this one
        instead". Unsetting the old one is what they meant.
        """
        if not validated_data.get("is_embedding_default"):
            return
        rows = ProviderCredential.objects.filter(
            organization=self.organization, is_embedding_default=True
        )
        if exclude_pk is not None:
            rows = rows.exclude(pk=exclude_pk)
        rows.update(is_embedding_default=False)

    def _clear_provider_default(self, validated_data, service, exclude_pk=None):
        """The same, for "which key this provider falls back to".

        Per (organization, service) rather than per organization, because an
        organization can now hold several keys for one provider and each
        provider needs its own answer.
        """
        if not validated_data.get("is_default") or service is None:
            return
        rows = ProviderCredential.objects.filter(
            organization=self.organization, service=service, is_default=True
        )
        if exclude_pk is not None:
            rows = rows.exclude(pk=exclude_pk)
        rows.update(is_default=False)

    def _only_key_for(self, service, exclude_pk=None):
        """Whether this would be the organization's first key for a provider.

        The first one is made the default automatically: a single-key
        organization should not have to make a choice it does not yet have.
        """
        rows = ProviderCredential.objects.filter(
            organization=self.organization, service=service
        )
        if exclude_pk is not None:
            rows = rows.exclude(pk=exclude_pk)
        return not rows.exists()


class ProviderCredentialListView(EmbeddingDefaultMixin, OrganizationScopedView):
    """The API keys this organization's agents run on."""

    def get(self, request):
        rows = (
            ProviderCredential.objects.filter(organization=self.organization)
            .select_related("service")
            .order_by("service__name")
        )
        return ok(ProviderCredentialSerializer(rows, many=True).data)

    def post(self, request):
        self.require_owner()

        serializer = ProviderCredentialSerializer(
            data=request.data, context={"organization": self.organization}
        )
        serializer.is_valid(raise_exception=True)

        service = serializer.validated_data.get("service")
        # The first key for a provider becomes its default without being asked.
        first = self._only_key_for(service)

        try:
            with transaction.atomic():
                self._clear_embedding_default(serializer.validated_data)
                self._clear_provider_default(serializer.validated_data, service)
                credential = serializer.save(
                    organization=self.organization,
                    is_default=serializer.validated_data.get("is_default") or first,
                )
        except IntegrityError:
            return fail(
                "A credential with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )

        credential = ProviderCredential.objects.select_related("service").get(
            pk=credential.pk
        )
        return ok(
            ProviderCredentialSerializer(credential).data, status.HTTP_201_CREATED
        )


class ProviderCredentialDetailView(EmbeddingDefaultMixin, OrganizationScopedView):

    def _credential(self, credential_id):
        return (
            ProviderCredential.objects.select_related("service")
            .filter(id=credential_id, organization=self.organization)
            .first()
        )

    def get(self, request, credential_id):
        credential = self._credential(credential_id)
        if credential is None:
            return fail("Credential not found.", status.HTTP_404_NOT_FOUND)
        return ok(ProviderCredentialSerializer(credential).data)

    def patch(self, request, credential_id):
        self.require_owner()

        credential = self._credential(credential_id)
        if credential is None:
            return fail("Credential not found.", status.HTTP_404_NOT_FOUND)

        serializer = ProviderCredentialSerializer(
            credential,
            data=request.data,
            partial=True,
            context={"organization": self.organization},
        )
        serializer.is_valid(raise_exception=True)

        try:
            with transaction.atomic():
                self._clear_embedding_default(
                    serializer.validated_data, exclude_pk=credential.pk
                )
                self._clear_provider_default(
                    serializer.validated_data,
                    credential.service,
                    exclude_pk=credential.pk,
                )
                serializer.save()
        except IntegrityError:
            return fail(
                "A credential with that name already exists in this organization.",
                status.HTTP_409_CONFLICT,
            )
        return ok(serializer.data)

    def delete(self, request, credential_id):
        self.require_owner()

        credential = self._credential(credential_id)
        if credential is None:
            return fail("Credential not found.", status.HTTP_404_NOT_FOUND)

        # Bots pinned to this key fall back to the organization default when it
        # goes (the FK is SET_NULL). That is a billing change nobody asked for,
        # so it is reported rather than discovered from an invoice.
        pinned = list(credential.bots.values_list("handle", flat=True)[:5])

        credential.delete()
        if pinned:
            return ok(
                None,
                message="Deleted. These bots now use the organization default: "
                + ", ".join(f"@{handle}" for handle in pinned),
            )
        return ok(None, message="Deleted.")


class EmbeddingCredentialStatusView(OrganizationScopedView):
    """Whether this organization can index knowledge at all.

    Its own endpoint because the answer is not obvious from the credential
    list: embeddings are OpenAI-only, so an organization with a perfectly good
    Groq key and nothing else has full chat and silently empty retrieval. This
    is what lets the knowledge screen say so before somebody uploads a document
    and wonders why the agent never cites it.
    """

    def get(self, request):
        try:
            embedding_api_key(self.organization)
        except CredentialNotConfigured as ex:
            return ok({"configured": False, "reason": str(ex)})
        return ok({"configured": True, "reason": ""})
