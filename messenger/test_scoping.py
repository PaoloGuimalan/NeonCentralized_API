"""Conversations belong to an organization, and only to it.

`MessagingView` was scoped when the cross-tenant reads were first fixed;
`ConversationView.get` was missed, because it returns only a conversation's
name and footprint. That is still another tenant's data, and the id needed to
ask for it travels in URLs and logs.
"""

from django.test import TestCase
from django.urls import reverse

from messenger.models import Conversation
from neon.testing import add_member, client_for, make_account, make_organization


class ConversationScopingTests(TestCase):

    def setUp(self):
        self.alice = make_account("alice")
        self.bob = make_account("bob")
        self.acme = make_organization(self.alice, "Acme", "acme")
        self.beta = make_organization(self.bob, "Beta", "beta")

        self.theirs = Conversation.objects.create(
            organization=self.beta, name="Beta's private thread", created_by=self.bob
        )
        self.ours = Conversation.objects.create(
            organization=self.acme, name="Acme's thread", created_by=self.alice
        )

    def _detail(self, conversation):
        return reverse(
            "api-messenger:messenger-conversation",
            args=[str(conversation.conversation_id)],
        )

    def test_a_conversation_in_another_organization_is_not_readable(self):
        response = client_for(self.alice, self.acme).get(self._detail(self.theirs))
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Beta's private thread", str(response.content))

    def test_your_own_conversation_is_readable(self):
        response = client_for(self.alice, self.acme).get(self._detail(self.ours))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["name"], "Acme's thread")

    def test_messages_in_another_organization_are_not_readable(self):
        response = client_for(self.alice, self.acme).get(
            reverse(
                "api-messenger:messenger-conversation",
                args=[str(self.theirs.conversation_id)],
            )
        )
        self.assertEqual(response.status_code, 404)

    def test_belonging_to_two_organizations_does_not_break_reading(self):
        """Before the header was honoured here, a second membership made every
        message route answer 409 - so creating a second organization silently
        broke chat for that user."""
        add_member(self.beta, self.alice, self.bob)

        response = client_for(self.alice, self.acme).get(self._detail(self.ours))
        self.assertEqual(response.status_code, 200)

        # And naming the other one reaches the other one's conversation.
        response = client_for(self.alice, self.beta).get(self._detail(self.theirs))
        self.assertEqual(response.status_code, 200)

    def test_without_naming_an_organization_an_ambiguous_caller_is_told(self):
        add_member(self.beta, self.alice, self.bob)
        response = client_for(self.alice).get(self._detail(self.ours))
        self.assertEqual(response.status_code, 409)
        self.assertIn("X-Organization", response.data["message"])
