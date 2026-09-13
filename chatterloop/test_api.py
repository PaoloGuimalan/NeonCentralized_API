"""The bots API, and the disconnect that stops them.

Two organizations throughout, as everywhere else: the interesting failures are
cross-tenant, and a single-tenant fixture cannot catch one.
"""

import uuid
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from django.utils.timezone import now

from chatterloop.client import ChatterloopAPIError, TokenRejected
from chatterloop.external_models import Bot, EntityPermission, Token
from chatterloop.models import ChatterloopBot, ChatterloopToken
from chatterloop.provisioning import ALL_SCOPES
from chatterloop.testing import create_chatterloop_schema
from chatterloop.tests import make_chatterloop_account, make_realm
from core.models import ConnectedAccount
from llm.models import Agent
from neon.testing import client_for, make_account, make_organization
from neon.utils import crypto


class BotAPITestCase(TestCase):
    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()

        self.alice = make_account("alice", entity_id="entity-alice")
        self.bob = make_account("bob", entity_id="entity-bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")

        self.client_acme = client_for(self.alice, self.acme)
        self.client_beta = client_for(self.bob, self.beta)

        self.agent = Agent.objects.create(
            organization=self.acme, name="Support Helper", slug="support-helper"
        )

    def tearDown(self):
        crypto._cipher = None

    def create(self, client=None, **payload):
        payload.setdefault("name", "Helper")
        payload.setdefault("handle", "helper")
        return (client or self.client_acme).post(
            reverse("api-chatterloop:bots"), payload, format="json"
        )


class MintEndpointTests(BotAPITestCase):

    def test_minting_returns_the_token_exactly_once(self):
        response = self.create(agent_uuid=self.agent.uuid)
        self.assertEqual(response.status_code, 201)

        token = response.data["data"]["token"]
        self.assertRegex(token, r"^clt_[0-9a-f]{12}_[0-9a-f]{64}$")
        # And the user is told it will not be shown again.
        self.assertIn("not shown again", response.data["message"])

        # Never again, through any other route.
        listing = self.client_acme.get(reverse("api-chatterloop:bots"))
        self.assertNotIn(token, str(listing.data))

        bot_id = response.data["data"]["id"]
        detail = self.client_acme.get(
            reverse("api-chatterloop:bot-detail", args=[bot_id])
        )
        self.assertNotIn(token, str(detail.data))

    def test_the_bot_is_bound_to_the_agent(self):
        response = self.create(agent_uuid=self.agent.uuid)
        bot = ChatterloopBot.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(bot.agent_id, self.agent.id)
        self.assertEqual(response.data["data"]["agent_name"], "Support Helper")

    def test_an_agent_from_another_organization_is_refused(self):
        theirs = Agent.objects.create(
            organization=self.beta, name="Theirs", slug="theirs"
        )
        response = self.create(agent_uuid=theirs.uuid)
        self.assertEqual(response.status_code, 404)
        self.assertFalse(ChatterloopBot.objects.exists())

    def test_a_taken_handle_is_a_conflict_with_a_reason(self):
        make_chatterloop_account("taken")
        response = self.create(handle="taken")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Chatterloop account", response.data["message"])

    def test_an_invalid_handle_is_a_field_error(self):
        response = self.create(handle="not a handle!")
        self.assertEqual(response.status_code, 400)
        self.assertIn("handle", response.data)

    def test_an_unknown_scope_is_refused(self):
        response = self.create(scopes=["messages.send", "messages.delete"])
        self.assertEqual(response.status_code, 400)

    def test_an_account_with_no_chatterloop_identity_cannot_mint(self):
        """An auto-provisioned end-user row from ExternalChatView has no
        entity_id and cannot speak as anybody."""
        stranger = make_account("stranger")
        org = make_organization(stranger, "Gamma", "gamma")
        response = self.create(client=client_for(stranger, org))
        self.assertEqual(response.status_code, 403)

    def test_bots_are_scoped_to_the_organization(self):
        self.create()
        response = self.client_beta.get(reverse("api-chatterloop:bots"))
        self.assertEqual(response.data["data"], [])

    def test_another_organizations_bot_is_not_found(self):
        created = self.create()
        response = self.client_beta.get(
            reverse("api-chatterloop:bot-detail", args=[created.data["data"]["id"]])
        )
        self.assertEqual(response.status_code, 404)


class OwnershipTests(BotAPITestCase):
    """Who a bot may speak as is decided server-side, every time."""

    def setUp(self):
        super().setUp()
        self.realm = make_realm("acmepage")
        self.connection = ConnectedAccount.objects.create(
            account=self.alice,
            provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
            external_id=self.realm.entity_id,
            external_type=ConnectedAccount.TYPE_REALM,
            external_name="Acme Page",
            external_username="acmepage",
        )

    def _grant_admin(self):
        from chatterloop.external_models import RealmMember

        RealmMember.objects.create(
            member_id=str(uuid.uuid4()),
            entity_id=self.alice.entity_id,
            realm_id=self.realm.id,
            role=RealmMember.ADMIN,
            date_joined=now(),
        )

    def test_a_page_can_own_a_bot_when_the_role_still_holds(self):
        self._grant_admin()
        response = self.create(handle="pagebot", owner=str(self.connection.pk))

        self.assertEqual(response.status_code, 201)
        bot = ChatterloopBot.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(bot.owner_entity_id, self.realm.entity_id)
        self.assertEqual(response.data["data"]["owner_type"], "page")
        self.assertEqual(
            Bot.objects.get(id=bot.bot_id).owner_entity_id, self.realm.entity_id
        )

    def test_a_revoked_page_role_is_caught_at_mint_time(self):
        """The connection row alone is not authority. A role removed on
        chatterloop after connecting must stop this here - otherwise a stale
        row is enough to publish a bot under a page you no longer administer.
        """
        # No RealmMember row at all: the role is gone.
        response = self.create(handle="pagebot", owner=str(self.connection.pk))
        self.assertEqual(response.status_code, 403)
        self.assertIn("no longer an owner or admin", response.data["message"])
        self.assertFalse(ChatterloopBot.objects.exists())

    def test_a_connection_belonging_to_someone_else_is_not_found(self):
        theirs = ConnectedAccount.objects.create(
            account=self.bob,
            provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
            external_id=str(uuid.uuid4()),
            external_type=ConnectedAccount.TYPE_REALM,
            external_name="Bob's Page",
        )
        response = self.create(handle="pagebot", owner=str(theirs.pk))
        self.assertEqual(response.status_code, 404)

    def test_personal_ownership_needs_no_connection(self):
        response = self.create()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["owner_type"], "personal")
        bot = ChatterloopBot.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(bot.owner_entity_id, self.alice.entity_id)


class TokenEndpointTests(BotAPITestCase):

    def test_rotating_issues_a_second_live_token(self):
        created = self.create()
        bot_id = created.data["data"]["id"]

        response = self.client_acme.post(
            reverse("api-chatterloop:bot-tokens", args=[bot_id]), {}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertRegex(
            response.data["data"]["token"], r"^clt_[0-9a-f]{12}_[0-9a-f]{64}$"
        )
        # The previous one keeps working until it is explicitly revoked.
        self.assertIn("still live", response.data["message"])
        self.assertEqual(ChatterloopToken.objects.filter(bot_id=bot_id).count(), 2)
        self.assertEqual(Token.objects.filter(is_active=True).count(), 2)

    def test_revoking_cuts_the_credential_off_on_chatterloop(self):
        created = self.create()
        bot_id = created.data["data"]["id"]
        credential = ChatterloopToken.objects.get(bot_id=bot_id)

        response = self.client_acme.delete(
            reverse(
                "api-chatterloop:bot-token-detail", args=[bot_id, str(credential.pk)]
            )
        )
        self.assertEqual(response.status_code, 200)

        token = Token.objects.get(id=credential.token_id)
        self.assertFalse(token.is_active)
        self.assertIsNotNone(token.revoked_at)

    def test_another_organizations_token_cannot_be_revoked(self):
        created = self.create()
        bot_id = created.data["data"]["id"]
        credential = ChatterloopToken.objects.get(bot_id=bot_id)

        response = self.client_beta.delete(
            reverse(
                "api-chatterloop:bot-token-detail", args=[bot_id, str(credential.pk)]
            )
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Token.objects.get(id=credential.token_id).is_active)


class DeactivationTests(BotAPITestCase):

    def test_deleting_a_bot_deactivates_rather_than_deletes(self):
        """The chatterloop entity outlives the Neon row; a hard delete would
        orphan a live identity Neon could no longer name or revoke."""
        created = self.create()
        bot_id = created.data["data"]["id"]

        response = self.client_acme.delete(
            reverse("api-chatterloop:bot-detail", args=[bot_id])
        )
        self.assertEqual(response.status_code, 200)

        bot = ChatterloopBot.objects.get(pk=bot_id)
        self.assertEqual(bot.status, ChatterloopBot.STATUS_DEACTIVATED)
        self.assertFalse(Bot.objects.get(id=bot.bot_id).is_active)
        self.assertFalse(Token.objects.filter(is_active=True).exists())

    def test_reactivating_says_the_tokens_are_still_revoked(self):
        created = self.create()
        bot_id = created.data["data"]["id"]
        self.client_acme.delete(reverse("api-chatterloop:bot-detail", args=[bot_id]))

        response = self.client_acme.post(
            reverse("api-chatterloop:bot-reactivate", args=[bot_id])
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Issue a new token", response.data["message"])
        self.assertEqual(
            ChatterloopBot.objects.get(pk=bot_id).status, ChatterloopBot.STATUS_ACTIVE
        )


class DisconnectStopsBotsTests(BotAPITestCase):
    """Disconnecting an identity must stop what speaks as it."""

    def setUp(self):
        super().setUp()
        self.realm = make_realm("acmepage")
        self.connection = ConnectedAccount.objects.create(
            account=self.alice,
            provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
            external_id=self.realm.entity_id,
            external_type=ConnectedAccount.TYPE_REALM,
            external_name="Acme Page",
            external_username="acmepage",
        )
        from chatterloop.external_models import RealmMember

        RealmMember.objects.create(
            member_id=str(uuid.uuid4()),
            entity_id=self.alice.entity_id,
            realm_id=self.realm.id,
            role=RealmMember.OWNER,
            date_joined=now(),
        )
        self.create(handle="pagebot", owner=str(self.connection.pk))

    def test_the_preview_names_the_affected_bots(self):
        """So the confirmation can say what it will stop, rather than asking
        about a consequence the user cannot see."""
        response = self.client_acme.get(
            reverse("api-core:connections-detail", args=[str(self.connection.pk)])
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([b["handle"] for b in response.data["bots"]], ["pagebot"])

    def test_disconnecting_deactivates_the_bots_and_revokes_their_tokens(self):
        response = self.client_acme.delete(
            reverse("api-core:connections-detail", args=[str(self.connection.pk)])
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["stopped_bots"], ["pagebot"])

        bot = ChatterloopBot.objects.get(handle="pagebot")
        self.assertEqual(bot.status, ChatterloopBot.STATUS_DEACTIVATED)
        self.assertIn("disconnected", bot.status_reason)
        self.assertFalse(Bot.objects.get(id=bot.bot_id).is_active)
        self.assertFalse(Token.objects.filter(is_active=True).exists())

    def test_a_bot_that_could_not_be_stopped_is_reported(self):
        """The connection is gone either way, but a still-live bot is
        something only the user can escalate - so it is said out loud rather
        than swallowed."""
        with patch(
            "chatterloop.provisioning.deactivate_bot",
            side_effect=RuntimeError("pg is down"),
        ):
            response = self.client_acme.delete(
                reverse("api-core:connections-detail", args=[str(self.connection.pk)])
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["failed_bots"], ["pagebot"])
        self.assertIn("could not be stopped", response.data["message"])


class VerificationTests(BotAPITestCase):
    """`/v1/whoami` answers the half a token cannot report about itself."""

    def _bot_id(self):
        return self.create().data["data"]["id"]

    def test_verification_records_the_resolved_handle(self):
        bot_id = self._bot_id()
        with patch(
            "chatterloop.views.whoami",
            return_value={
                "entity_id": "e1",
                "handle": "helper",
                "scopes": list(ALL_SCOPES),
            },
        ):
            response = self.client_acme.post(
                reverse("api-chatterloop:bot-verify", args=[bot_id])
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["data"]["ok"])
        self.assertFalse(response.data["data"]["handle_mismatch"])

        bot = ChatterloopBot.objects.get(pk=bot_id)
        self.assertEqual(bot.verified_handle, "helper")
        self.assertIsNotNone(bot.last_verified_at)

    def test_a_handle_chatterloop_resolves_differently_is_surfaced(self):
        """The silent failure this exists to catch: a bot whose handle resolves
        to somebody else matches no mentions, and no log says why."""
        bot_id = self._bot_id()
        with patch(
            "chatterloop.views.whoami",
            return_value={"entity_id": "e1", "handle": "someone-else", "scopes": []},
        ):
            response = self.client_acme.post(
                reverse("api-chatterloop:bot-verify", args=[bot_id])
            )
        self.assertTrue(response.data["data"]["handle_mismatch"])

    def test_a_rejected_token_is_an_answer_not_an_error(self):
        bot_id = self._bot_id()
        with patch(
            "chatterloop.views.whoami",
            side_effect=TokenRejected("Invalid or expired token.", 401),
        ):
            response = self.client_acme.post(
                reverse("api-chatterloop:bot-verify", args=[bot_id])
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["data"]["ok"])
        self.assertEqual(response.data["data"]["status_code"], 401)

    def test_an_unreachable_service_is_not_reported_as_a_bad_token(self):
        bot_id = self._bot_id()
        with patch(
            "chatterloop.views.whoami",
            side_effect=ChatterloopAPIError("Could not reach Chatterloop"),
        ):
            response = self.client_acme.post(
                reverse("api-chatterloop:bot-verify", args=[bot_id])
            )
        self.assertEqual(response.status_code, 503)

    def test_a_missing_grant_is_reported_alongside_the_scopes(self):
        """A scope on the token with no grant row is a silent 403 at request
        time. whoami shows the token half only, so the grant half is read from
        the table developer_service itself reads."""
        bot_id = self._bot_id()
        bot = ChatterloopBot.objects.get(pk=bot_id)
        EntityPermission.objects.filter(
            entity_id=bot.entity_id, permission="messages.send"
        ).delete()

        with patch(
            "chatterloop.views.whoami",
            return_value={
                "entity_id": bot.entity_id,
                "handle": "helper",
                "scopes": list(ALL_SCOPES),
            },
        ):
            response = self.client_acme.post(
                reverse("api-chatterloop:bot-verify", args=[bot_id])
            )

        self.assertIn("messages.send", response.data["data"]["grants"]["missing"])


class HandleAvailabilityTests(BotAPITestCase):

    def test_a_free_handle_is_available(self):
        response = self.client_acme.get(
            reverse("api-chatterloop:handle-available"), {"handle": "brand-new"}
        )
        self.assertTrue(response.data["data"]["available"])

    def test_a_taken_handle_says_what_took_it(self):
        make_realm("acmepage")
        response = self.client_acme.get(
            reverse("api-chatterloop:handle-available"), {"handle": "@acmepage"}
        )
        self.assertFalse(response.data["data"]["available"])
        self.assertEqual(response.data["data"]["taken_by"], "a Chatterloop page")
