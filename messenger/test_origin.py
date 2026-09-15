"""Telling a conversation's surface apart in the list.

The distinction exists because of the `created_by` fix: attributing a bot's
mirrored threads to a real person is what puts them in that person's
conversation list at all, and an unlabelled bot thread sitting among somebody's
own chats is worse than it not being there.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from django.apps import apps as live_apps
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from chatterloop.models import ChatterloopBot
from chatterloop.tasks import _mirror_conversation
from chatterloop.triggers import Trigger, TriggerReason, TriggerSource
from llm.models import Agent, Model, Role, Service
from messenger.models import Conversation
from messenger.serializers import ConversationSerializer
from neon.testing import client_for, make_account, make_organization
from user.models import Token

classify = importlib.import_module("messenger.migrations.0016_conversation_origin")


def trigger(conversation_id="conv-1"):
    return Trigger(
        source=TriggerSource.MESSAGE,
        reason=TriggerReason.MENTION,
        author_entity_id="entity-human",
        conversation_id=conversation_id,
        message_id="m1",
        text="@helper hello",
        query="hello",
        dedupe_key=f"msg:{uuid.uuid4().hex}",
    )


class OriginAtWriteTimeTests(TestCase):
    """Each writer records its own surface, rather than a reader guessing."""

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")

    def test_a_native_conversation_is_labelled_native(self):
        response = client_for(self.alice, self.org).post(
            reverse("api-messenger:messenger-conversation"), {"name": "My chat"}
        )
        self.assertEqual(response.status_code, 200)

        conversation = Conversation.objects.get(
            conversation_id=response.data["conversation_id"]
        )
        self.assertEqual(conversation.origin, Conversation.ORIGIN_NATIVE)
        self.assertFalse(conversation.is_external)

    def test_the_default_is_native_so_nothing_is_silently_unlabelled(self):
        conversation = Conversation.objects.create(
            organization=self.org, name="bare", created_by=self.alice
        )
        self.assertEqual(conversation.origin, Conversation.ORIGIN_NATIVE)

    def test_a_third_party_apps_conversation_is_labelled_external(self):
        """Through the real view, because the label is only worth anything if
        the writer that is meant to set it actually does."""
        token = Token.objects.create(account=self.alice, name="Widget")
        role = Role.objects.create(
            organization=self.org, name="Support", system_prompt="Be helpful."
        )
        agent = Agent.objects.create(
            organization=self.org, name="Helper", slug="helper", role=role
        )
        service = Service.objects.create(name="openai")
        model = Model.objects.create(service=service, model="gpt-4o-mini")
        self.org.llm_api_key = "sk-test"
        self.org.save(update_fields=["llm_api_key"])

        llm = SimpleNamespace(
            stream_chat_completion=lambda *args, **kwargs: iter(["Hi there."])
        )
        client = APIClient()
        client.credentials(HTTP_X_DEVELOPER_TOKEN=token.token)

        with patch("messenger.views.LLMFactory") as factory, patch(
            "messenger.views.get_rag"
        ) as rag, patch("messenger.views.index_chat_message_task"):
            factory.return_value.create.return_value = llm
            rag.return_value.retrieve.return_value = []
            response = client.post(
                reverse("api-messenger:messenger-external-chat"),
                {
                    "email": "carol@example.com",
                    "first_name": "Carol",
                    "last_name": "Tester",
                    "external_conversation_id": "widget-1",
                    "agent_uuid": str(agent.uuid),
                    "model_uuid": str(model.uuid),
                    "content": "hello",
                },
                format="json",
            )

        self.assertEqual(response.status_code, 200, response.data)
        conversation = Conversation.objects.get(
            conversation_id=response.data["conversation_id"]
        )
        self.assertEqual(conversation.origin, Conversation.ORIGIN_EXTERNAL)
        self.assertTrue(conversation.is_external)

    def test_a_mirrored_bot_conversation_is_labelled_chatterloop(self):
        bot = ChatterloopBot.objects.create(
            organization=self.org,
            created_by=self.alice,
            entity_id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            name="Helper",
            handle="helper",
            owner_entity_id=self.alice.entity_id,
            status=ChatterloopBot.STATUS_ACTIVE,
        )

        conversation = _mirror_conversation(bot, trigger())

        self.assertEqual(conversation.origin, Conversation.ORIGIN_CHATTERLOOP)
        self.assertTrue(conversation.is_external)


class OriginInTheListTests(TestCase):
    """What the conversation list hands the frontend."""

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")

    def make(self, name, origin, footprint=None):
        return Conversation.objects.create(
            organization=self.org,
            name=name,
            footprint=footprint,
            created_by=self.alice,
            origin=origin,
        )

    def test_every_row_carries_a_raw_value_a_label_and_the_grouping_bit(self):
        expected = {
            Conversation.ORIGIN_NATIVE: ("Neon platform", False),
            Conversation.ORIGIN_EXTERNAL: ("External app", True),
            Conversation.ORIGIN_CHATTERLOOP: ("Chatterloop bot", True),
        }
        for origin, (label, external) in expected.items():
            conversation = self.make(origin, origin, footprint=f"fp-{origin}")
            data = ConversationSerializer(
                conversation, context={"include_latest_message": False}
            ).data

            self.assertEqual(data["origin"], origin)
            self.assertEqual(data["origin_label"], label)
            self.assertIs(data["is_external"], external)

    def test_the_list_endpoint_returns_the_label(self):
        self.make("A bot thread", Conversation.ORIGIN_CHATTERLOOP, "chatterloop:x")

        response = client_for(self.alice, self.org).get(
            reverse("api-messenger:messenger-list")
        )

        self.assertEqual(response.status_code, 200)
        row = response.data["results"][0]
        self.assertEqual(row["origin"], Conversation.ORIGIN_CHATTERLOOP)
        self.assertEqual(row["origin_label"], "Chatterloop bot")
        self.assertTrue(row["is_external"])

    def test_a_bot_thread_and_an_own_chat_are_distinguishable_side_by_side(self):
        """The case that made this necessary - both are attributed to Alice and
        both land in her list."""
        self.make("My chat", Conversation.ORIGIN_NATIVE)
        self.make("@helper: conv-1", Conversation.ORIGIN_CHATTERLOOP, "chatterloop:x")

        response = client_for(self.alice, self.org).get(
            reverse("api-messenger:messenger-list")
        )

        origins = {row["name"]: row["origin"] for row in response.data["results"]}
        self.assertEqual(
            origins,
            {
                "My chat": Conversation.ORIGIN_NATIVE,
                "@helper: conv-1": Conversation.ORIGIN_CHATTERLOOP,
            },
        )


class OriginFilterTests(TestCase):
    """`?origin=` on the conversation list."""

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")
        self.make("My chat", Conversation.ORIGIN_NATIVE)
        self.make("Widget thread", Conversation.ORIGIN_EXTERNAL, "fp-external")
        self.make("@helper: conv-1", Conversation.ORIGIN_CHATTERLOOP, "fp-bot")

    def make(self, name, origin, footprint=None):
        return Conversation.objects.create(
            organization=self.org,
            name=name,
            footprint=footprint,
            created_by=self.alice,
            origin=origin,
        )

    def list(self, query=""):
        return client_for(self.alice, self.org).get(
            reverse("api-messenger:messenger-list") + query
        )

    def names(self, response):
        return {row["name"] for row in response.data["results"]}

    def test_without_the_parameter_everything_is_listed(self):
        response = self.list()
        self.assertEqual(len(response.data["results"]), 3)

    def test_one_origin_narrows_to_it(self):
        response = self.list("?origin=native")
        self.assertEqual(self.names(response), {"My chat"})

    def test_several_comma_separated(self):
        response = self.list("?origin=external,chatterloop")
        self.assertEqual(self.names(response), {"Widget thread", "@helper: conv-1"})

    def test_several_as_repeated_parameters(self):
        response = self.list("?origin=external&origin=chatterloop")
        self.assertEqual(self.names(response), {"Widget thread", "@helper: conv-1"})

    def test_whitespace_and_empty_values_are_tolerated(self):
        response = self.list("?origin=native,%20,")
        self.assertEqual(self.names(response), {"My chat"})

    def test_an_empty_parameter_is_not_a_filter(self):
        response = self.list("?origin=")
        self.assertEqual(len(response.data["results"]), 3)

    def test_an_unknown_origin_is_rejected_rather_than_ignored(self):
        """The important one: silently listing everything would show the caller
        the bot threads it explicitly asked not to see."""
        response = self.list("?origin=natvie")

        self.assertEqual(response.status_code, 400)
        self.assertIn("natvie", response.data["message"])
        # And it says what the valid values are, rather than only that this
        # one was not.
        self.assertIn("chatterloop", response.data["message"])

    def test_one_bad_value_among_good_ones_still_fails(self):
        response = self.list("?origin=native,nonsense")
        self.assertEqual(response.status_code, 400)

    def test_filtering_does_not_reach_another_users_conversations(self):
        """The filter narrows; it must not widen past `created_by`."""
        bob = make_account("bob", entity_id=str(uuid.uuid4()))
        Conversation.objects.create(
            organization=self.org,
            name="Bob's chat",
            created_by=bob,
            origin=Conversation.ORIGIN_NATIVE,
        )

        response = self.list("?origin=native")

        self.assertEqual(self.names(response), {"My chat"})

    def test_ordering_survives_the_filter(self):
        """Most recently active first - the filter is a WHERE, not a reorder."""
        self.make("Older bot thread", Conversation.ORIGIN_CHATTERLOOP, "fp-bot-2")

        response = self.list("?origin=chatterloop")

        listed = [row["name"] for row in response.data["results"]]
        self.assertEqual(listed, ["Older bot thread", "@helper: conv-1"])


class ClassifyExistingRowsTests(TestCase):
    """Migration 0016's one-time guess at rows written before the column."""

    databases = {"default"}

    def setUp(self):
        self.alice = make_account("alice", entity_id=str(uuid.uuid4()))
        self.org = make_organization(self.alice, "Acme", "acme")

    def unclassified(self, name, footprint):
        """A row as it stood before 0016 - the column defaults to native."""
        return Conversation.objects.create(
            organization=self.org,
            name=name,
            footprint=footprint,
            created_by=self.alice,
        )

    def test_a_chatterloop_footprint_is_recognised(self):
        row = self.unclassified(
            "@helper: conv-1", f"chatterloop:{self.org.id}:conv-1"
        )

        classify.classify_existing(live_apps, None)

        row.refresh_from_db()
        self.assertEqual(row.origin, Conversation.ORIGIN_CHATTERLOOP)

    def test_an_external_apps_footprint_is_recognised(self):
        row = self.unclassified(
            "widget-1", f"{self.org.id}:carol@example.com:widget-1"
        )

        classify.classify_existing(live_apps, None)

        row.refresh_from_db()
        self.assertEqual(row.origin, Conversation.ORIGIN_EXTERNAL)

    def test_a_footprint_naming_another_organization_is_not_claimed(self):
        """The organization prefix is what makes the external shape more than a
        coincidence, so a footprint carrying somebody else's id is not it."""
        other = make_organization(make_account("bob"), "Beta", "beta")
        row = self.unclassified("odd", f"{other.id}:carol@example.com:widget-1")

        classify.classify_existing(live_apps, None)

        row.refresh_from_db()
        self.assertEqual(row.origin, Conversation.ORIGIN_NATIVE)

    def test_a_client_supplied_footprint_stays_native(self):
        row = self.unclassified("My chat", "my-own-dedupe-key")

        classify.classify_existing(live_apps, None)

        row.refresh_from_db()
        self.assertEqual(row.origin, Conversation.ORIGIN_NATIVE)

    def test_a_row_with_no_footprint_is_left_alone(self):
        row = self.unclassified("My chat", None)

        classify.classify_existing(live_apps, None)

        row.refresh_from_db()
        self.assertEqual(row.origin, Conversation.ORIGIN_NATIVE)

    def test_nothing_to_do_is_not_an_error(self):
        classify.classify_existing(live_apps, None)
        self.assertEqual(Conversation.objects.count(), 0)
