"""The control endpoint as an actual request.

`test_presence.py` covers the helpers - action parsing, url building, the key
lifecycle. This covers the VIEW, which is where the security lives: what
authenticates it, what a wrong key is allowed to reveal, and what a caller can
reach without a Neon session.

NEON ROWS ONLY. A ChatterloopBot lives in Neon's own database, so none of this
needs the chatterloop schema - which is also why these tests pass while the
suites that build that schema currently do not.
"""

from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from chatterloop.models import ChatterloopBot
from chatterloop.presence import ensure_control_key
from neon.testing import client_for, make_account, make_organization
from neon.utils import crypto


class ControlEndpointTests(TestCase):
    def setUp(self):
        crypto._cipher = None
        self.alice = make_account("alice", entity_id="entity-alice")
        self.bob = make_account("bob", entity_id="entity-bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")

        self.bot = ChatterloopBot.objects.create(
            organization=self.acme,
            created_by=self.alice,
            entity_id="entity-bot",
            bot_id="bot-1",
            name="Helper",
            handle="helper",
            owner_entity_id="entity-alice",
            status=ChatterloopBot.STATUS_ACTIVE,
            is_online=True,
        )
        self.key = ensure_control_key(self.bot)
        self.url = reverse("bots:control", args=[self.bot.pk])
        # No session, no organization header: this endpoint is driven by
        # things that hold neither.
        self.anon = APIClient()

    def tearDown(self):
        crypto._cipher = None

    def call(self, action, key=None, url=None):
        request = {}
        if key is not None:
            request["HTTP_AUTHORIZATION"] = f"Bearer {key}"
        return self.anon.post(f"{url or self.url}?action={action}", **request)

    # --- what it does ----------------------------------------------------

    def test_sleep_takes_the_bot_offline(self):
        response = self.call("sleep", self.key)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.bot.refresh_from_db()
        self.assertFalse(self.bot.is_online)
        self.assertIsNotNone(self.bot.online_changed_at)

    def test_wake_brings_it_back(self):
        self.bot.is_online = False
        self.bot.save(update_fields=["is_online"])

        response = self.call("wake", self.key)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_it_is_idempotent(self):
        """A caller retrying after a timeout should not have to care which
        attempt landed."""
        self.assertEqual(self.call("wake", self.key).status_code, status.HTTP_200_OK)
        self.assertEqual(self.call("wake", self.key).status_code, status.HTTP_200_OK)

        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_the_action_may_come_in_the_body(self):
        response = self.anon.post(
            self.url,
            {"action": "sleep"},
            format="json",
            HTTP_AUTHORIZATION=f"Bearer {self.key}",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.bot.refresh_from_db()
        self.assertFalse(self.bot.is_online)

    # --- what it refuses -------------------------------------------------

    def test_no_key_is_a_404(self):
        """Not a 401: telling an unauthenticated caller that this bot id
        exists is itself an answer, and the id travels in a URL meant to be
        pasted into other people's configuration."""
        response = self.call("sleep")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_a_wrong_key_is_a_404_and_changes_nothing(self):
        response = self.call("sleep", "not-the-key")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_another_bots_key_does_not_work(self):
        """The key is scoped to ONE bot. A holder of one must not be able to
        drive another, even inside the same organization."""
        other = ChatterloopBot.objects.create(
            organization=self.acme,
            created_by=self.alice,
            entity_id="entity-bot-2",
            bot_id="bot-2",
            name="Second",
            handle="second",
            owner_entity_id="entity-alice",
            status=ChatterloopBot.STATUS_ACTIVE,
            is_online=True,
        )
        other_key = ensure_control_key(other)

        response = self.call("sleep", other_key)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_an_unknown_bot_is_a_404(self):
        response = self.call(
            "sleep", self.key, reverse("bots:control", args=["no-such-bot"])
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_a_bad_action_is_refused_before_anything_changes(self):
        for action in ("", "delete", "restart", "WAKEUP"):
            response = self.call(action, self.key)
            self.assertEqual(
                response.status_code,
                status.HTTP_400_BAD_REQUEST,
                msg=f"{action!r} should be refused",
            )

        self.bot.refresh_from_db()
        self.assertTrue(self.bot.is_online)

    def test_a_bot_with_no_key_cannot_be_driven(self):
        """An empty stored key must never match an empty presented one."""
        self.bot.control_key_encrypted = ""
        self.bot.save(update_fields=["control_key_encrypted"])

        self.assertEqual(
            self.call("sleep", "").status_code, status.HTTP_404_NOT_FOUND
        )
        self.assertEqual(
            self.call("sleep", "anything").status_code, status.HTTP_404_NOT_FOUND
        )


class ControlKeyRouteTests(TestCase):
    """Issuing and rotating the key. A SESSION route, unlike the endpoint it
    unlocks - who may hold a bot's control key is a Neon question."""

    def setUp(self):
        crypto._cipher = None
        self.alice = make_account("alice", entity_id="entity-alice")
        self.bob = make_account("bob", entity_id="entity-bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")
        self.client_acme = client_for(self.alice, self.acme)
        self.client_beta = client_for(self.bob, self.beta)

        self.bot = ChatterloopBot.objects.create(
            organization=self.acme,
            created_by=self.alice,
            entity_id="entity-bot",
            bot_id="bot-1",
            name="Helper",
            handle="helper",
            owner_entity_id="entity-alice",
            status=ChatterloopBot.STATUS_ACTIVE,
        )
        self.url = reverse("bots:control-key", args=[self.bot.pk])

    def tearDown(self):
        crypto._cipher = None

    def test_it_returns_the_url_and_the_key(self):
        response = self.client_acme.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        body = response.json()["data"]
        self.assertTrue(body["key"])
        self.assertIn("/api/bots/", body["url"])
        self.assertIn("wake", body["usage"]["wake"])

    def test_rotating_invalidates_the_previous_key(self):
        first = self.client_acme.get(self.url).json()["data"]["key"]
        second = self.client_acme.post(self.url).json()["data"]["key"]

        self.assertNotEqual(first, second)

        anon = APIClient()
        stale = anon.post(
            f"{reverse('bots:control', args=[self.bot.pk])}?action=sleep",
            HTTP_AUTHORIZATION=f"Bearer {first}",
        )
        self.assertEqual(stale.status_code, status.HTTP_404_NOT_FOUND)

    def test_another_organization_cannot_read_the_key(self):
        """The cross-tenant case, which is the one that matters: a key is a
        credential for somebody else's bot."""
        self.assertEqual(
            self.client_beta.get(self.url).status_code, status.HTTP_404_NOT_FOUND
        )

    def test_a_session_is_required(self):
        self.assertIn(
            APIClient().get(self.url).status_code,
            (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN),
        )
