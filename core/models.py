"""Identities a Neon account can act as, beyond its own.

WHY THIS TABLE IS NOT CALLED `ChatterloopPage`
----------------------------------------------
Chatterloop is the identity provider today, and the only one. It will not
necessarily be the only one - the whole point of Neon holding agents rather
than chatterloop holding them is that an agent can eventually answer somewhere
else. Naming the abstraction after its first implementation is how a second
one ends up bolted on as a parallel table with its own half of every query.

So `provider` is a column, not a class name. Adding a platform means new rows
and a new adapter, not a new table.

WHAT LIVES HERE AND WHAT DOES NOT
---------------------------------
The signed-in user's OWN chatterloop identity is NOT here - it is on
`user.Account.entity_id`, because signing in already established it and a
second row asserting the same fact is a second thing that can disagree.

This table holds the identities sign-in cannot establish: chatterloop **page
realms** the user owns or administers, which they may publish bots under.
"""

import uuid

from django.db import models
from django.utils.timezone import now

from neon.utils.identifiers import new_id


class ConnectedAccount(models.Model):
    """An external identity a Neon account may act as."""

    PROVIDER_CHATTERLOOP = "chatterloop"
    PROVIDER_CHOICES = [
        (PROVIDER_CHATTERLOOP, "Chatterloop"),
    ]

    TYPE_USER = "user"
    TYPE_REALM = "realm"
    TYPE_CHOICES = [
        (TYPE_USER, "User"),
        (TYPE_REALM, "Realm"),
    ]

    id = models.CharField(
        max_length=150, default=new_id, unique=True, blank=True, primary_key=True
    )

    account = models.ForeignKey(
        "user.Account",
        on_delete=models.CASCADE,
        related_name="connected_accounts",
    )

    provider = models.CharField(
        max_length=50,
        choices=PROVIDER_CHOICES,
        default=PROVIDER_CHATTERLOOP,
    )

    # The provider's entity id. For chatterloop this is what
    # `bot_bot.owner_entity` is set to, which is why it is the one field that
    # must never be wrong - everything else on this row is presentation.
    external_id = models.CharField(max_length=40, db_index=True)

    external_type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        default=TYPE_REALM,
    )

    # Handle: a realm's slug, or a user's username on a provider where that
    # applies. Display only - never used to resolve anything.
    external_username = models.CharField(max_length=150, blank=True, default="")
    external_name = models.CharField(max_length=255, blank=True, default="")
    external_profile = models.CharField(max_length=500, blank=True, default="none")

    # Provider-specific extras that do not deserve a column: for chatterloop,
    # `realm_id` and the member `role` the connection was granted under.
    metadata = models.JSONField(default=dict, blank=True)

    connected_at = models.DateTimeField(default=now)
    last_verified_at = models.DateTimeField(null=True, blank=True, default=None)
    is_active = models.BooleanField(default=True)
    disconnected_at = models.DateTimeField(null=True, blank=True, default=None)

    class Meta:
        constraints = [
            # One live claim per external identity, across all of Neon.
            # Without it two Neon accounts could each connect the same page and
            # both publish bots owned by it, with no way to tell afterwards
            # which of them a given bot came from.
            #
            # PARTIAL, on is_active, so disconnecting and reconnecting the same
            # page is not permanently blocked by the dead row left behind.
            models.UniqueConstraint(
                fields=["provider", "external_id"],
                condition=models.Q(is_active=True),
                name="one_live_claim_per_external_identity",
            ),
        ]
        indexes = [
            models.Index(fields=["account", "provider"]),
        ]

    def __str__(self):
        label = self.external_username or self.external_id
        return f"{self.provider}:{self.external_type}:{label}"
