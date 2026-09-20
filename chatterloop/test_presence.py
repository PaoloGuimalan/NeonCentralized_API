"""The per-bot control endpoint: `?action=wake|sleep`.

The security properties are the point. This is the one route in the app that
is not behind a Neon session - it is meant to be pasted into somebody else's
configuration - so what authenticates it, and what a wrong key is allowed to
reveal, is pinned here rather than assumed.
"""

from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from chatterloop import presence


class ActionTests(SimpleTestCase):
    def test_only_wake_and_sleep(self):
        self.assertEqual(presence.resolve_action("wake"), "wake")
        self.assertEqual(presence.resolve_action("sleep"), "sleep")
        for junk in ("", None, "WAKEUP", "stop", "delete", "online"):
            self.assertIsNone(presence.resolve_action(junk))

    def test_typed_by_hand_is_tolerated(self):
        """This value goes into a config box as often as it is generated."""
        self.assertEqual(presence.resolve_action("  WAKE "), "wake")
        self.assertEqual(presence.resolve_action("Sleep"), "sleep")


class UrlTests(SimpleTestCase):
    def test_the_path_carries_the_api_prefix(self):
        """From reverse(), never a literal: this path is copied into another
        system's configuration, where a wrong one 404s silently.

        And NOT under /chatterloop/ - a bot is a general integration, so the
        URL people paste elsewhere does not name the platform carrying its
        messages. See bot_api_urls.py."""
        path = presence.control_path(SimpleNamespace(pk="bot-1"))

        self.assertEqual(path, "/api/bots/bot-1/control")

    def test_a_base_is_joined_without_doubling_the_slash(self):
        url = presence.control_url(SimpleNamespace(pk="bot-1"), "https://neon.example/")

        self.assertEqual(
            url, "https://neon.example/api/bots/bot-1/control"
        )

    def test_no_base_yields_the_path_alone(self):
        """A caller with no request and no configured base still gets
        something usable rather than a string starting with None."""
        self.assertEqual(
            presence.control_url(SimpleNamespace(pk="b"), ""),
            "/api/bots/b/control",
        )


class ControlKeyTests(SimpleTestCase):
    def test_an_existing_key_is_reused(self):
        bot = SimpleNamespace(control_key_encrypted="enc", handle="neon")

        with mock.patch.object(presence, "decrypt", return_value="the-key"):
            self.assertEqual(presence.ensure_control_key(bot), "the-key")

    def test_a_missing_key_is_minted_and_saved(self):
        saved = {}
        bot = SimpleNamespace(
            control_key_encrypted="",
            handle="neon",
            save=lambda **kw: saved.update(kw),
        )

        with mock.patch.object(presence, "encrypt", side_effect=lambda v: f"enc:{v}"):
            key = presence.ensure_control_key(bot)

        self.assertTrue(key)
        self.assertEqual(bot.control_key_encrypted, f"enc:{key}")
        self.assertIn("control_key_encrypted", saved["update_fields"])

    def test_an_unreadable_key_is_replaced_rather_than_trusted(self):
        """Whatever the holder has would no longer match anything Neon can
        check, so the only honest outcome is a new key."""
        saved = {}
        bot = SimpleNamespace(
            control_key_encrypted="corrupt",
            handle="neon",
            save=lambda **kw: saved.update(kw),
        )

        with mock.patch.object(
            presence, "decrypt", side_effect=ValueError("bad key")
        ), mock.patch.object(presence, "encrypt", side_effect=lambda v: f"enc:{v}"):
            key = presence.ensure_control_key(bot)

        self.assertTrue(key)
        self.assertEqual(bot.control_key_encrypted, f"enc:{key}")

    def test_keys_are_unguessable_and_distinct(self):
        keys = {presence.generate_control_key() for _ in range(50)}

        self.assertEqual(len(keys), 50)
        for key in keys:
            # token_urlsafe(32) - 256 bits, and url-safe so it survives being
            # pasted into a header, a query string or a YAML file.
            self.assertGreaterEqual(len(key), 40)
