"""Who a bot's conversations get attributed to.

The bug these cover: `ChatterloopBot.created_by` is SET_NULL, so deleting the
account that minted a bot made `_mirror_conversation` write NULL into
`Conversation.created_by` and the whole exchange died on that column's NOT
NULL. Every case here is a bot with no minting account still landing on a real
person.
"""

import uuid

from django.test import TestCase
from django.utils.timezone import now

from core.models import ConnectedAccount
from messenger.models import Conversation
from neon.testing import make_account, make_organization

from .models import ChatterloopBot
from .ownership import conversation_owner
from .tasks import _mirror_conversation
from .triggers import Trigger, TriggerReason, TriggerSource


def make_bot(organization, created_by=None, owner_entity_id=None, **kwargs):
    return ChatterloopBot.objects.create(
        organization=organization,
        created_by=created_by,
        entity_id=str(uuid.uuid4()),
        bot_id=str(uuid.uuid4()),
        name="Helper",
        handle=f"helper-{uuid.uuid4().hex[:6]}",
        owner_entity_id=owner_entity_id or str(uuid.uuid4()),
        status=ChatterloopBot.STATUS_ACTIVE,
        **kwargs,
    )


class ConversationOwnerTests(TestCase):

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")

    def test_the_minting_account_wins(self):
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        bot = make_bot(self.org, created_by=bob, owner_entity_id=bob.entity_id)

        self.assertEqual(conversation_owner(bot), bob)

    def test_falls_back_to_the_account_that_connected_the_page(self):
        """A bot published under a page, whose minting account is gone."""
        page_entity = str(uuid.uuid4())
        connection = ConnectedAccount.objects.create(
            account=self.alice,
            external_id=page_entity,
            external_type=ConnectedAccount.TYPE_REALM,
            external_name="Acme Support",
            connected_at=now(),
        )
        bot = make_bot(
            self.org,
            created_by=None,
            owner_entity_id=page_entity,
            connected_account=connection,
        )

        self.assertEqual(conversation_owner(bot), self.alice)

    def test_falls_back_to_the_person_the_bot_speaks_as(self):
        """A bot published under someone's own chatterloop identity."""
        bot = make_bot(self.org, created_by=None, owner_entity_id=self.alice.entity_id)

        self.assertEqual(conversation_owner(bot), self.alice)

    def test_a_disconnected_pages_dead_claim_does_not_count(self):
        """Disconnecting a page gives up the authority to speak as it, so its
        row must not keep collecting attribution - the organization owner does
        instead."""
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
        bot = make_bot(self.org, created_by=None, owner_entity_id=page_entity)

        self.assertEqual(conversation_owner(bot), self.alice)

    def test_the_organization_owner_is_the_terminating_case(self):
        """Nothing resolves the owner entity, and the chain still names
        somebody - `Organization.created_by` is NOT NULL."""
        bot = make_bot(self.org, created_by=None, owner_entity_id=str(uuid.uuid4()))

        self.assertEqual(conversation_owner(bot), self.alice)


class MirroredConversationTests(TestCase):

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")
        self.bot = make_bot(self.org, created_by=None)

    def trigger(self):
        return Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.MENTION,
            author_entity_id="entity-human",
            conversation_id="conv-1",
            message_id="m1",
            text="@helper hello",
            query="hello",
            dedupe_key=f"msg:{uuid.uuid4().hex}",
        )

    def test_a_bot_with_no_minting_account_still_attributes_its_conversation(self):
        conversation = _mirror_conversation(self.bot, self.trigger())

        self.assertEqual(conversation.created_by, self.alice)

    def test_a_row_left_unattributed_is_repaired_on_the_next_exchange(self):
        """Migration 0014 relaxed the column, so rows written in between have
        a NULL that nothing else would ever fill in."""
        trigger = self.trigger()
        Conversation.objects.create(
            organization=self.org,
            name="stale",
            footprint=f"chatterloop:{self.org.id}:conv-1",
            created_by=None,
        )

        conversation = _mirror_conversation(self.bot, trigger)

        self.assertEqual(conversation.created_by, self.alice)
        conversation.refresh_from_db()
        self.assertEqual(conversation.created_by, self.alice)

    def test_an_existing_attribution_is_not_rewritten(self):
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        Conversation.objects.create(
            organization=self.org,
            name="theirs",
            footprint=f"chatterloop:{self.org.id}:conv-1",
            created_by=bob,
        )

        conversation = _mirror_conversation(self.bot, self.trigger())

        self.assertEqual(conversation.created_by, bob)
