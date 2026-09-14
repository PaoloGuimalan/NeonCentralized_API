"""Who a bot's conversations are attributed to.

A chatterloop bot has no Neon `Account` of its own - it is an identity in
another service - but every `Conversation` it mirrors into Neon needs an
answerable person on `created_by`. This module is the one rule for resolving
that person, and it always resolves to one.

WHY THIS IS A CHAIN AND NOT JUST `bot.created_by`
-------------------------------------------------
`ChatterloopBot.created_by` is SET_NULL, so the account that minted a bot can
be deleted while the bot keeps answering - and a bot minted before that column
existed never had one at all. Either way `_mirror_conversation` wrote NULL and
the exchange died on `Conversation.created_by`'s NOT NULL, which is the
failure `messenger/migrations/0014` relaxed the column to get past.

Relaxing the column is not the answer on its own: an unattributed conversation
is one nobody can be asked about. So the fallback asks the question
`created_by` was really asking - who is answerable for what this bot says -
by walking to the identity the bot SPEAKS AS and finding the Neon account
behind it.

THE ORDER, AND WHY
------------------
1. `created_by`       - whoever minted it. First, so nothing that already
                        resolves changes its answer.
2. connected page     - the Neon account that connected the page this bot
                        publishes under. That account is the one that claimed
                        the authority to speak as the page.
3. `owner_entity_id`  - the same question asked of chatterloop's entity id,
                        for a bot whose `connected_account` was cleared when
                        the page was disconnected (SET_NULL - see the model).
                        A person's own mirrored identity first, then a live
                        page claim.
4. organization owner - `Organization.created_by` is NOT NULL, which makes
                        this the terminating case and the reason the chain
                        cannot come back empty.
"""

import logging

from core.models import ConnectedAccount
from user.models import Account

logger = logging.getLogger(__name__)


def _account_for_entity(entity_id):
    """The Neon account behind a chatterloop entity id, if Neon knows one."""
    if not entity_id:
        return None

    # The entity is a person, and signing in mirrored them onto
    # `Account.entity_id`. Cheapest and most direct answer available.
    account = Account.objects.filter(entity_id=entity_id).first()
    if account is not None:
        return account

    # The entity is a page. Only a LIVE claim counts: the dead row a
    # disconnect leaves behind is precisely the authority that was given up,
    # and attributing new conversations to it would outlive the grant.
    connection = (
        ConnectedAccount.objects.select_related("account")
        .filter(
            provider=ConnectedAccount.PROVIDER_CHATTERLOOP,
            external_id=entity_id,
            is_active=True,
        )
        .first()
    )
    return connection.account if connection is not None else None


def conversation_owner(bot):
    """The Neon account `bot`'s conversations are attributed to.

    Never returns None for a bot with an organization, which is every bot -
    `ChatterloopBot.organization` is a CASCADE FK and `Organization.created_by`
    is NOT NULL.
    """
    if bot.created_by_id:
        return bot.created_by

    if bot.connected_account_id and bot.connected_account.account_id:
        owner = bot.connected_account.account
        logger.info(
            "bot %s has no minting account; attributing to the page's connector",
            bot.handle,
        )
        return owner

    owner = _account_for_entity(bot.owner_entity_id)
    if owner is not None:
        logger.info(
            "bot %s has no minting account; attributing to its owner entity",
            bot.handle,
        )
        return owner

    # Worth saying loudly: the bot is answering under a chatterloop identity
    # Neon can no longer put a name to, and its conversations are landing on
    # the organization's owner by default rather than by anybody's decision.
    logger.warning(
        "bot %s has no resolvable owner; attributing to the organization owner",
        bot.handle,
    )
    return bot.organization.created_by
