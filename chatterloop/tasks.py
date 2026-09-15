"""Answering a trigger: retrieve, generate, send, mirror.

WHY THIS IS A CELERY TASK AND NOT PART OF THE SUPERVISOR
--------------------------------------------------------
Everything here is expensive and stateless - a history fetch, an embedding, a
model call, a send. Everything the supervisor does is cheap and stateful. Split
that way, an idle bot costs a coroutine on the supervisor while the expensive
half scales on workers that hold nothing and can be added or lost freely.

THE REPLY IS RECORDED ONLY AFTER A SUCCESSFUL SEND
--------------------------------------------------
`policy.record_reply` fires here, not where the decision was made. A reply that
failed to send must not consume the conversation's hourly budget, or a broken
outbound path silently rate-limits the bot into silence - the failure mode the
reference implementation calls out by name. That is the whole reason the policy
store is in Redis rather than in the deciding process's memory.

MIRRORING
---------
Both sides of the exchange are written into a Neon `Conversation` keyed on a
`chatterloop:` footprint, so a bot's conversations appear in the same place as
everything else and feed the same retrieval index. The footprint is namespaced
per ORGANIZATION, not just by conversation id: `Conversation.footprint` is
globally unique, and two organizations' bots can be in the same chatterloop
conversation - without the namespace the second would collide onto the first's
row and leak its history.
"""

import logging

from celery import shared_task
from django.utils.timezone import now

from llm.services.llm_factory import LLMFactory
from messenger.models import Conversation, Message
from organization.credentials import CredentialNotConfigured, chat_api_key, embedding_api_key

from .client import ChatterloopAPIError, TokenRejected, post_comment, send_message
from .models import ChatterloopBot
from .policy import AddressedOnlyPolicy, build_store
from .runtime import BotIdentity
from .triggers import Trigger, TriggerSource

logger = logging.getLogger(__name__)

# How much retrieved context is put in front of the model. Matches the external
# chat path so a bot and an embedded widget answer from the same amount.
RETRIEVAL_TOP_K = 8

MAX_REPLY_CHARS = 5000


def footprint_for(bot, trigger):
    """The Neon conversation this exchange mirrors into.

    Namespaced by organization for the reason in the module docstring. Comments
    have no conversation, so they mirror per POST instead - a post's comment
    thread is the closest thing it has to one.
    """
    if trigger.source is TriggerSource.COMMENT:
        return f"chatterloop:{bot.organization_id}:post:{trigger.post_id}"
    return f"chatterloop:{bot.organization_id}:{trigger.conversation_id}"


def _live_token(bot):
    return bot.tokens.filter(
        revoked_at__isnull=True, provisioned_at__isnull=False
    ).order_by("-created_at").first()


def _mirror_conversation(bot, trigger):
    """Get or create the Neon conversation for this exchange."""
    footprint = footprint_for(bot, trigger)
    conversation, _ = Conversation.objects.get_or_create(
        footprint=footprint,
        defaults={
            "organization": bot.organization,
            "name": f"@{bot.handle}: {trigger.conversation_id or trigger.post_id}",
            # The bot has no Neon Account of its own, so the conversation is
            # attributed to whoever minted it - the person answerable for what
            # it says.
            "created_by": bot.created_by,
        },
    )
    return conversation


def _mirror_message(conversation, content, message_type, agent=None):
    """Write one side of the exchange into Neon.

    `sender` is left NULL: the author is a chatterloop entity with no Neon
    Account, and inventing one per participant would fill the account table
    with rows nobody can sign in as. The chatterloop identity lives on the
    conversation's footprint instead.
    """
    return Message.objects.create(
        conversation=conversation,
        sender=None,
        agent=agent,
        message_type=message_type,
        content=content,
    )


@shared_task(name="chatterloop.answer_trigger")
def answer_trigger(bot_id, payload):
    """Generate and send one reply.

    Takes a plain dict rather than a Trigger: Celery serialises arguments as
    JSON, so the queue never carries executable state.
    """
    trigger = Trigger.from_payload(payload)

    bot = (
        ChatterloopBot.objects.select_related(
            "organization",
            "agent__role",
            "model__service",
            "created_by",
            "provider_credential",
        )
        .filter(pk=bot_id)
        .first()
    )
    if bot is None:
        logger.warning("answer_trigger for unknown bot %s", bot_id)
        return

    # Re-checked here, not just at dispatch. Between the supervisor deciding
    # and this task running, the bot may have been deactivated or its page
    # disconnected - and a queued task is exactly how a "stopped" bot gets one
    # more word in.
    if not bot.is_live:
        logger.info("bot %s is no longer live; not answering", bot.handle)
        return
    # Switched off between the supervisor deciding and this task running. A
    # queued task is exactly how a bot somebody just turned off gets one more
    # word in, which is the opposite of what the switch is for.
    if not bot.is_online:
        logger.info("bot %s was switched offline; not answering", bot.handle)
        return
    if bot.agent is None or bot.agent.role is None:
        logger.warning("bot %s has no agent or role; not answering", bot.handle)
        return
    if bot.model is None or bot.model.service is None:
        logger.warning("bot %s has no model; not answering", bot.handle)
        return

    credential = _live_token(bot)
    if credential is None:
        logger.warning("bot %s has no live token; not answering", bot.handle)
        return

    try:
        # The bot's own key when it has one, the organization's default
        # otherwise - see organization/credentials.py for why a mismatched
        # assignment falls back rather than failing.
        api_key = chat_api_key(
            bot.organization, bot.model.service, bot.provider_credential
        )
    except CredentialNotConfigured as ex:
        logger.warning("bot %s cannot answer: %s", bot.handle, ex)
        return

    conversation = _mirror_conversation(bot, trigger)
    _mirror_message(conversation, trigger.text or trigger.query, "text")

    context = _retrieve(bot, conversation, trigger)

    llm = LLMFactory().create(
        service=bot.model.service.name,
        api_key=api_key,
        model=bot.model.model,
    )
    if llm is None:
        logger.warning(
            "bot %s: unsupported provider %s", bot.handle, bot.model.service.name
        )
        return

    tools = list(bot.agent.role.tools.filter(is_enabled=True))

    try:
        reply = "".join(
            token
            for token in llm.stream_chat_completion(
                context, bot.agent.role.system_prompt, trigger.query, tools
            )
            if token is not None
        ).strip()
    except Exception:
        logger.exception("bot %s failed generating a reply", bot.handle)
        return

    if not reply:
        # Not counted against the conversation's budget: an empty generation is
        # a failure to answer, not an answer.
        logger.info("bot %s generated an empty reply; sending nothing", bot.handle)
        return

    if len(reply) > MAX_REPLY_CHARS:
        # The API rejects anything longer, and a rejected send is worse than a
        # trimmed one: the user gets nothing at all.
        reply = reply[: MAX_REPLY_CHARS - 1].rstrip() + "…"

    token_value = credential.token
    try:
        if trigger.source is TriggerSource.COMMENT:
            post_comment(token_value, trigger.post_id, reply, trigger.comment_id)
        else:
            send_message(
                token_value, trigger.conversation_id, reply, trigger.message_id
            )
    except TokenRejected as ex:
        # Not retried, and worth saying loudly: the credential is dead or is
        # missing a grant, and every future answer from this bot will fail the
        # same way until somebody acts.
        logger.error("bot %s: token rejected when sending: %s", bot.handle, ex.message)
        return
    except ChatterloopAPIError as ex:
        logger.error("bot %s could not deliver a reply: %s", bot.handle, ex)
        return

    # Only now. See the module docstring.
    identity = BotIdentity(bot.entity_id, bot.verified_handle or bot.handle)
    policy = AddressedOnlyPolicy(identity, store=build_store(bot.pk))
    policy.record_reply(trigger)

    _mirror_message(conversation, reply, "ai_reply", agent=bot.agent)

    logger.info(
        "bot %s replied in %s (%s)",
        bot.handle,
        trigger.conversation_id or trigger.post_id,
        trigger.reason,
    )


def _retrieve(bot, conversation, trigger):
    """Context for the model, in the shape the LLM services expect.

    Retrieval failing is not fatal. An unconfigured embedding key, or a
    Pinecone outage, should cost the bot its memory for that turn - not its
    ability to answer at all.
    """
    try:
        from llm.scripts.tasks import get_rag

        retrieved = get_rag().retrieve(
            trigger.query,
            conversation.conversation_id,
            conversation.organization_id,
            embedding_api_key(bot.organization),
            RETRIEVAL_TOP_K,
            # The bot answers people outside the organization, so which
            # documents are in scope is the whole point of passing this.
            agent=bot.agent,
        )
    except CredentialNotConfigured as ex:
        logger.info("bot %s answering without retrieval: %s", bot.handle, ex)
        return []
    except Exception:
        logger.exception("bot %s: retrieval failed; answering without it", bot.handle)
        return []

    return [
        {
            "role": "user" if row["msg_type"] == "text" else "assistant",
            "content": f'History: {row["text"]}',
        }
        for row in retrieved
    ]
