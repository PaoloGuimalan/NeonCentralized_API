"""The 0015 backfill, run against real rows.

The migration's own function is called with the live app registry rather than
through `migrate`: the resolution is the part with judgement in it, and the
graph applying from zero is already covered by
`manage.py migrate --settings=neon.settings_test`.
"""

import importlib
import uuid

from django.apps import apps as live_apps
from django.test import TestCase
from django.utils.timezone import now

from chatterloop.models import ChatterloopBot
from core.models import ConnectedAccount
from messenger.models import Conversation
from neon.testing import make_account, make_organization

backfill = importlib.import_module(
    "messenger.migrations.0015_backfill_conversation_created_by"
)


def make_bot(organization, handle, created_by=None, owner_entity_id=None, **kwargs):
    return ChatterloopBot.objects.create(
        organization=organization,
        created_by=created_by,
        entity_id=str(uuid.uuid4()),
        bot_id=str(uuid.uuid4()),
        name=handle.title(),
        handle=handle,
        owner_entity_id=owner_entity_id or str(uuid.uuid4()),
        status=ChatterloopBot.STATUS_ACTIVE,
        **kwargs,
    )


def unattributed(organization, name, footprint):
    """A conversation as the buggy mirroring path left it."""
    return Conversation.objects.create(
        organization=organization,
        name=name,
        footprint=footprint,
        created_by=None,
    )


def run():
    backfill.backfill_created_by(live_apps, None)


class HandleParsingTests(TestCase):

    def test_the_name_mirroring_writes_is_parsed(self):
        self.assertEqual(backfill._handle_from_name("@helper: conv-1"), "helper")

    def test_a_post_mirror_parses_the_same_way(self):
        self.assertEqual(backfill._handle_from_name("@helper: post-9"), "helper")

    def test_anything_else_is_declined_rather_than_guessed_at(self):
        for name in [None, "", "helper: conv-1", "@", "@:", "a@b: c"]:
            self.assertIsNone(backfill._handle_from_name(name), name)


class BackfillTests(TestCase):

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")

    def test_the_minting_account_is_used_when_the_bot_still_has_one(self):
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        make_bot(self.org, "helper", created_by=bob)
        row = unattributed(
            self.org, "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, bob)

    def test_a_bot_with_no_minting_account_resolves_through_its_page(self):
        page_entity = str(uuid.uuid4())
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        connection = ConnectedAccount.objects.create(
            account=bob,
            external_id=page_entity,
            external_type=ConnectedAccount.TYPE_REALM,
            connected_at=now(),
        )
        make_bot(
            self.org,
            "helper",
            created_by=None,
            owner_entity_id=page_entity,
            connected_account=connection,
        )
        row = unattributed(
            self.org, "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, bob)

    def test_a_bot_speaking_as_a_person_resolves_to_them(self):
        carol = make_account("carol", entity_id=str(uuid.uuid4()))
        make_bot(
            self.org, "helper", created_by=None, owner_entity_id=carol.entity_id
        )
        row = unattributed(
            self.org, "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, carol)

    def test_a_dead_page_claim_does_not_collect_attribution(self):
        page_entity = str(uuid.uuid4())
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        ConnectedAccount.objects.create(
            account=bob,
            external_id=page_entity,
            external_type=ConnectedAccount.TYPE_REALM,
            connected_at=now(),
            is_active=False,
            disconnected_at=now(),
        )
        make_bot(self.org, "helper", created_by=None, owner_entity_id=page_entity)
        row = unattributed(
            self.org, "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, self.alice)

    def test_a_row_whose_bot_is_gone_falls_back_to_the_organization_owner(self):
        row = unattributed(
            self.org, "@vanished: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, self.alice)

    def test_a_row_that_is_not_a_bot_mirror_still_gets_an_owner(self):
        """Nothing but the mirroring path could write a NULL, but a row whose
        name does not parse must not be left behind on that assumption."""
        row = unattributed(self.org, "Some native conversation", None)

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, self.alice)

    def test_an_already_attributed_row_is_untouched(self):
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        row = Conversation.objects.create(
            organization=self.org,
            name="@helper: conv-1",
            footprint=f"chatterloop:{self.org.id}:conv-1",
            created_by=bob,
        )
        make_bot(self.org, "helper", created_by=self.alice)

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, bob)

    def test_a_handle_resolves_only_within_its_own_organization(self):
        """(organization, handle) is unique, not handle alone - two tenants can
        both have an @helper, and one must not collect the other's rows."""
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        other = make_organization(bob, "Beta", "beta")
        carol = make_account("carol", entity_id=str(uuid.uuid4()))
        make_bot(other, "helper", created_by=carol)

        row = unattributed(
            self.org, "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        run()

        row.refresh_from_db()
        self.assertEqual(row.created_by, self.alice)

    def test_every_null_row_is_repaired_in_one_pass(self):
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        make_bot(self.org, "helper", created_by=bob)
        for index in range(5):
            unattributed(
                self.org,
                f"@helper: conv-{index}",
                f"chatterloop:{self.org.id}:conv-{index}",
            )
        unattributed(self.org, "@gone: conv-9", f"chatterloop:{self.org.id}:conv-9")

        run()

        self.assertFalse(Conversation.objects.filter(created_by__isnull=True).exists())
        self.assertEqual(Conversation.objects.filter(created_by=bob).count(), 5)
        self.assertEqual(Conversation.objects.filter(created_by=self.alice).count(), 1)

    def test_nothing_to_do_is_not_an_error(self):
        run()
        self.assertEqual(Conversation.objects.count(), 0)
