"""Signing in when Neon cannot read chatterloop's database.

The bug these cover: chatterloop accepted the credential, and Neon then threw
on the `user_account` read - either because the row was not there or because
the `chatterloop` database alias is not configured in this deployment. Both
turned a successful sign-in into a 500 instead of importing the account.
"""

import uuid
from unittest.mock import patch

import jwt
from django.db import OperationalError
from django.test import TestCase

from core.services import chatterloop_identity as identity
from core.services.chatterloop_identity import (
    ChatterloopUnavailable,
    sync_account,
)
from chatterloop.testing import create_chatterloop_schema
from chatterloop.tests import make_chatterloop_account
from neon.testing import make_account
from user.models import Account


def usertoken(**overrides):
    """A chatterloop `usertoken`, signed with a secret Neon does not hold.

    Signed with the WRONG key on purpose: the response's trust comes from Neon
    having made the request over TLS, not from a shared JWT secret, and a
    sign-in must not start depending on one.
    """
    payload = {
        "id": str(uuid.uuid4()),
        "username": "carol",
        "first_name": "Carol",
        "middle_name": "N/A",
        "last_name": "Tester",
        "email": "carol@example.com",
        "profile": "none",
        "is_active": True,
        "is_verified": True,
    }
    payload.update(overrides)
    return jwt.encode(payload, "not-neons-secret", algorithm="HS256")


def login_result(entity_id, **overrides):
    return {
        "personal_entity_id": entity_id,
        "usertoken": usertoken(**overrides),
        "authtoken": "irrelevant",
    }


class AutoImportTests(TestCase):
    """`sync_account` with the chatterloop database standing in for one that
    cannot be read. The alias exists in settings_test, so it is the read that
    is made to fail rather than the configuration."""

    databases = {"default"}

    def setUp(self):
        self.entity_id = str(uuid.uuid4())

    def unreadable_database(self):
        return patch.object(
            identity,
            "_identity_from_database",
            wraps=lambda entity_id: None,
        )

    def test_an_account_is_imported_from_the_login_response(self):
        with self.unreadable_database():
            account = sync_account(login_result(self.entity_id))

        self.assertEqual(account.entity_id, self.entity_id)
        self.assertEqual(account.email, "carol@example.com")
        self.assertEqual(account.username, "carol")
        self.assertEqual(account.join_type, "chatterloop")
        self.assertTrue(account.is_verified)
        # Signed in through chatterloop and only ever through chatterloop.
        self.assertIsNone(account.password)

    def test_an_existing_account_is_adopted_by_email_not_duplicated(self):
        existing = make_account("old-carol", email="carol@example.com")

        with self.unreadable_database():
            account = sync_account(login_result(self.entity_id))

        self.assertEqual(account.pk, existing.pk)
        self.assertEqual(Account.objects.filter(email="carol@example.com").count(), 1)
        self.assertEqual(account.entity_id, self.entity_id)

    def test_a_missing_chatterloop_row_falls_through_instead_of_failing(self):
        """`ChatterloopAccount.DoesNotExist` used to end the sign-in. The
        alias in settings_test has no `user_account` table, so this exercises
        the real lookup."""
        result = login_result(self.entity_id)

        with patch.object(
            identity.ChatterloopAccount.objects,
            "get",
            side_effect=identity.ChatterloopAccount.DoesNotExist,
        ):
            account = sync_account(result)

        self.assertEqual(account.entity_id, self.entity_id)

    def test_an_unreachable_chatterloop_database_falls_through(self):
        """The deployment-shaped failure: CHATTERLOOP_DB_* unset or pointed at
        another environment, so the read raises rather than returning nothing."""
        with patch.object(
            identity.ChatterloopAccount.objects,
            "get",
            side_effect=OperationalError("could not connect to server"),
        ):
            account = sync_account(login_result(self.entity_id))

        self.assertEqual(account.entity_id, self.entity_id)

    def test_a_blank_username_is_generated_rather_than_written_empty(self):
        with self.unreadable_database():
            account = sync_account(login_result(self.entity_id, username=""))

        self.assertTrue(account.username)

    def test_neither_source_resolving_is_reported_as_an_outage(self):
        """Not a credential error: chatterloop said yes, and Neon simply
        cannot find out who to."""
        result = login_result(self.entity_id)
        result["usertoken"] = ""

        with self.unreadable_database(), self.assertRaises(ChatterloopUnavailable):
            sync_account(result)

    def test_a_token_with_no_email_is_not_a_usable_identity(self):
        result = {
            "personal_entity_id": self.entity_id,
            "usertoken": usertoken(email=""),
        }

        with self.unreadable_database(), self.assertRaises(ChatterloopUnavailable):
            sync_account(result)


class DatabaseStillWinsTests(TestCase):
    """The fallback is a fallback. When chatterloop's tables ARE readable they
    are what gets projected, because they are current at the moment they are
    read where the login response is a snapshot of one instant."""

    databases = {"default", "chatterloop"}

    def setUp(self):
        create_chatterloop_schema()
        self.remote = make_chatterloop_account("carol")

    def test_the_database_row_is_preferred_over_the_login_response(self):
        result = login_result(
            self.remote.entity_id,
            username="stale-name",
            email="stale@example.com",
        )

        account = sync_account(result)

        self.assertEqual(account.username, self.remote.username)
        self.assertEqual(account.email, self.remote.email)
        self.assertEqual(account.chatterloop_account_id, str(self.remote.id))


class ForwardedResultTests(TestCase):
    """`_forward` hands the whole `result` on, because the account data the
    fallback needs lives in `usertoken` rather than in `personal_entity_id`."""

    databases = {"default"}

    def test_the_whole_result_is_returned(self):
        entity_id = str(uuid.uuid4())
        body = {"status": True, "result": login_result(entity_id)}

        with patch.object(identity.requests, "post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = body
            result = identity.authenticate_with_password("carol", "pw")

        self.assertEqual(result["personal_entity_id"], entity_id)
        self.assertIn("usertoken", result)

    def test_the_password_never_leaves_the_payload(self):
        """Guarding the one thing this module promises: the credential goes to
        chatterloop and nowhere else."""
        with patch.object(identity.requests, "post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {
                "status": True,
                "result": login_result(str(uuid.uuid4())),
            }
            identity.authenticate_with_password("carol", "s3cret")

        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent, {"email_username": "carol", "password": "s3cret"})
        self.assertNotIn("password", post.call_args.kwargs["headers"])
