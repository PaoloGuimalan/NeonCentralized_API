"""Bots: minting them, binding an agent, and managing their credentials.

WHO A BOT MAY SPEAK AS IS DECIDED HERE, NOT BY THE CLIENT
---------------------------------------------------------
The request names a ConnectedAccount (or "personal"); this module resolves that
to an entity id and re-checks, against chatterloop's own membership table, that
the caller may still act as it. Re-checked at mint time rather than trusted
from the connection because a page admin role can be revoked after the page was
connected, and the stale row would otherwise be enough to publish a bot under
that page's name.
"""

import logging

from django.utils.timezone import now
from rest_framework import status

from core.models import ConnectedAccount
from core.services.chatterloop_identity import may_act_as
from llm.models import Agent, Model
from neon.api import OrganizationScopedView, fail, ok

from .client import ChatterloopAPIError, TokenRejected, whoami
from .control import announce
from .leases import running_bot_ids
from .models import ChatterloopBot, ChatterloopToken
from .provisioning import (
    ProvisioningError,
    deactivate_bot,
    grant_report,
    handle_conflict,
    reactivate_bot,
    rotate_token,
    revoke_token,
    mint_bot,
)
from .serializers import (
    ChatterloopBotSerializer,
    ChatterloopTokenSerializer,
    CreateBotSerializer,
    RotateTokenSerializer,
    UpdateBotSerializer,
)

logger = logging.getLogger(__name__)


class BotViewMixin:

    def _context(self, bots):
        """Serializer context carrying which bots are actually running.

        Resolved here rather than in the serializer so a listing costs one
        Redis round trip instead of one per bot.
        """
        return {"running": running_bot_ids([bot.pk for bot in bots])}

    def _bots(self):
        return (
            ChatterloopBot.objects.filter(organization=self.organization)
            # `model__service` as well as `model`: the serializer reads the
            # model's name, and without it a listing of N bots costs N extra
            # queries. `agent` and the rest were already joined; the model was
            # added later and missed.
            .select_related(
                "agent",
                "model",
                "model__service",
                "provider_credential",
                "connected_account",
                "created_by",
            )
            .prefetch_related("tokens")
        )

    def _resolve_owner(self, request, owner):
        """Turn the client's `owner` into `(entity_id, connected_account)`.

        Returns a `Response` instead when the caller may not act as it.
        """
        user = request.user

        if not user.entity_id:
            return fail(
                "This account is not linked to a Chatterloop identity, so it "
                "cannot publish bots.",
                status.HTTP_403_FORBIDDEN,
            )

        if owner in ("", "personal", None):
            return user.entity_id, None

        connection = ConnectedAccount.objects.filter(
            id=owner, account=user, is_active=True
        ).first()
        if connection is None:
            return fail(
                "That connected identity was not found.", status.HTTP_404_NOT_FOUND
            )

        # The re-check. A role revoked on chatterloop after connecting must
        # stop this here, not at the next page load.
        try:
            permitted = may_act_as(user.entity_id, connection.external_id)
        except Exception:
            logger.exception("could not re-check page authority")
            return fail(
                "Could not reach Chatterloop to confirm you still administer "
                "that page.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if not permitted:
            return fail(
                f"You are no longer an owner or admin of {connection.external_name}, "
                "so you cannot publish a bot as that page.",
                status.HTTP_403_FORBIDDEN,
            )

        return connection.external_id, connection

    def _agent(self, agent_uuid):
        """An agent in the caller's organization, or None."""
        if not agent_uuid:
            return None
        return Agent.objects.filter(
            uuid=agent_uuid, organization=self.organization
        ).first()

    def _credential(self, credential_id):
        """A provider credential in the caller's organization, or None.

        Scoped, like every other related id here: an unrestricted lookup would
        let a bot be pointed at another tenant's API key.
        """
        if not credential_id:
            return None
        from organization.models import ProviderCredential

        return ProviderCredential.objects.filter(
            id=credential_id, organization=self.organization
        ).first()

    @staticmethod
    def _model(model_uuid):
        """A model from the catalogue, or None.

        Not organization-scoped, because models are not: the catalogue is
        platform data, and "OpenAI / gpt-4o-mini" means the same thing to
        everyone. What IS scoped is the API key used to call it, which the
        answering task resolves from the organization's own credentials.
        """
        if not model_uuid:
            return None
        return Model.objects.select_related("service").filter(uuid=model_uuid).first()


class BotListView(BotViewMixin, OrganizationScopedView):

    def get(self, request):
        bots = list(self._bots())
        return ok(
            ChatterloopBotSerializer(
                bots, many=True, context=self._context(bots)
            ).data
        )

    def post(self, request):
        serializer = CreateBotSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        resolved = self._resolve_owner(request, data["owner"])
        if not isinstance(resolved, tuple):
            return resolved
        owner_entity_id, connection = resolved

        agent = self._agent(data.get("agent_uuid"))
        if data.get("agent_uuid") and agent is None:
            return fail("That agent was not found.", status.HTTP_404_NOT_FOUND)

        model = self._model(data.get("model_uuid"))
        if data.get("model_uuid") and model is None:
            return fail("That model was not found.", status.HTTP_404_NOT_FOUND)

        credential = self._credential(data.get("credential_id"))
        if data.get("credential_id") and credential is None:
            return fail("That API key was not found.", status.HTTP_404_NOT_FOUND)

        try:
            bot, credential = mint_bot(
                organization=self.organization,
                created_by=request.user,
                name=data["name"],
                handle=data["handle"],
                owner_entity_id=owner_entity_id,
                agent=agent,
                model=model,
                provider_credential=credential,
                connected_account=connection,
                description=data.get("description", ""),
                scopes=data.get("scopes"),
            )
        except ProvisioningError as ex:
            return fail(str(ex), status.HTTP_409_CONFLICT)

        payload = ChatterloopBotSerializer(bot, context=self._context([bot])).data
        # THE ONLY TIME THIS VALUE IS EVER RETURNED. Chatterloop stores only a
        # hash, and Neon will not hand it back through any other endpoint.
        payload["token"] = credential.plaintext
        return ok(
            payload,
            status.HTTP_201_CREATED,
            message="Copy the token now - it is not shown again.",
        )


class BotDetailView(BotViewMixin, OrganizationScopedView):

    def _bot(self, bot_id):
        return self._bots().filter(id=bot_id).first()

    def get(self, request, bot_id):
        bot = self._bot(bot_id)
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)
        return ok(ChatterloopBotSerializer(bot, context=self._context([bot])).data)

    def patch(self, request, bot_id):
        bot = self._bot(bot_id)
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        serializer = UpdateBotSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if "agent_uuid" in data:
            agent = self._agent(data["agent_uuid"])
            if data["agent_uuid"] and agent is None:
                return fail("That agent was not found.", status.HTTP_404_NOT_FOUND)
            bot.agent = agent

        if "model_uuid" in data:
            model = self._model(data["model_uuid"])
            if data["model_uuid"] and model is None:
                return fail("That model was not found.", status.HTTP_404_NOT_FOUND)
            bot.model = model

        if "credential_id" in data:
            credential = self._credential(data["credential_id"])
            if data["credential_id"] and credential is None:
                return fail("That API key was not found.", status.HTTP_404_NOT_FOUND)
            bot.provider_credential = credential

        # `name` and `description` are Neon-side presentation here. They are
        # NOT pushed to bot_bot: that row is what chatterloop renders, and a
        # rename is a write to somebody else's table for a cosmetic change.
        # Worth revisiting if the mismatch turns out to confuse people.
        if "name" in data:
            bot.name = data["name"]
        if "description" in data:
            bot.description = data["description"]

        # Applied to a RUNNING bot by the sweep, which re-reads the row and
        # reconfigures the live policy - see BotWorker.refresh. Without that,
        # ticking this on a bot that was already online did nothing at all, and
        # the most natural order to do things in was the broken one.
        bot_chat_changed = (
            "allow_bot_conversations" in data
            and bot.allow_bot_conversations != data["allow_bot_conversations"]
        )
        if "allow_bot_conversations" in data:
            bot.allow_bot_conversations = data["allow_bot_conversations"]

        bot.save(
            update_fields=[
                "agent",
                "model",
                "provider_credential",
                "allow_bot_conversations",
                "name",
                "description",
                "updated_at",
            ]
        )

        if bot_chat_changed:
            # Best effort, exactly like the online switch: the sweep converges
            # within its interval regardless, this only makes it immediate.
            announce(str(bot.pk), "settings")
        return ok(ChatterloopBotSerializer(bot, context=self._context([bot])).data)

    def delete(self, request, bot_id):
        """Deactivate and revoke. Never a hard delete.

        The chatterloop entity, and everything it has said, outlives the Neon
        row - so deleting here would orphan a live identity Neon could no
        longer name, let alone revoke.
        """
        bot = self._bot(bot_id)
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        try:
            deactivate_bot(bot, reason=f"Deactivated by @{request.user.username}.")
        except Exception as ex:
            logger.exception("failed to deactivate bot %s", bot.entity_id)
            return fail(
                f"Could not deactivate that bot on Chatterloop: {ex}",
                status.HTTP_502_BAD_GATEWAY,
            )

        return ok(
            ChatterloopBotSerializer(bot, context=self._context([bot])).data,
            message="Deactivated and its tokens revoked.",
        )


class BotOnlineView(BotViewMixin, OrganizationScopedView):
    """Switch a bot's event stream on or off.

    NOTHING ELSE CHANGES. The chatterloop identity stays, the token stays
    valid, the agent binding stays. Offline simply means no supervisor leases
    the bot, so no event stream is opened and no frames reach it - and turning
    it back on needs no new credential.

    Distinct from DELETE (deactivate), which revokes every token and is the
    heavy, hard-to-undo option.
    """

    def post(self, request, bot_id):
        return self._set(request, bot_id, True)

    def delete(self, request, bot_id):
        return self._set(request, bot_id, False)

    def _set(self, request, bot_id, online):
        bot = self._bots().filter(id=bot_id).first()
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        if bot.is_online == online:
            return ok(
                ChatterloopBotSerializer(bot, context=self._context([bot])).data,
                message=f"Already {'online' if online else 'offline'}.",
            )

        bot.is_online = online
        bot.online_changed_at = now()
        bot.save(update_fields=["is_online", "online_changed_at", "updated_at"])

        # Nudge any running supervisor so this takes effect now rather than at
        # its next sweep. Best effort: the sweep converges either way, which is
        # why a failure here is not reported as one.
        announce(str(bot.pk), "online" if online else "offline")

        if online and not bot.can_answer:
            # Switched on, but it still cannot answer. Said now rather than
            # leaving somebody to wonder why a bot they just enabled is silent.
            return ok(
                ChatterloopBotSerializer(bot, context=self._context([bot])).data,
                message="Online, but it has no agent or no model yet, so it "
                "will receive messages and not reply.",
            )

        return ok(
            ChatterloopBotSerializer(bot, context=self._context([bot])).data,
            message=(
                "Online. It will start listening within a few seconds."
                if online
                else "Offline. Its token is still valid - switch it back on any time."
            ),
        )


class BotReactivateView(BotViewMixin, OrganizationScopedView):

    def post(self, request, bot_id):
        bot = self._bots().filter(id=bot_id).first()
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        try:
            reactivate_bot(bot)
        except ProvisioningError as ex:
            return fail(str(ex), status.HTTP_409_CONFLICT)
        except Exception as ex:
            logger.exception("failed to reactivate bot %s", bot.entity_id)
            return fail(f"Could not reactivate that bot: {ex}", status.HTTP_502_BAD_GATEWAY)

        return ok(
            ChatterloopBotSerializer(bot, context=self._context([bot])).data,
            message="Reactivated. Issue a new token - the old ones stay revoked.",
        )


class BotVerifyView(BotViewMixin, OrganizationScopedView):
    """Ask developer_service what this credential actually is.

    Two things only this can tell you, and both fail silently otherwise:

      * the HANDLE developer_service resolves, which can differ from the one
        Neon set, because handles are resolved by UNION across accounts,
        realms and bots and the first match wins;
      * whether each scope has a matching entity grant - the other half of the
        authorization intersection, which a token cannot report about itself.
    """

    def post(self, request, bot_id):
        bot = self._bots().filter(id=bot_id).first()
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        credential = bot.tokens.filter(
            revoked_at__isnull=True, provisioned_at__isnull=False
        ).first()
        if credential is None:
            return fail(
                "This bot has no live token to verify.", status.HTTP_409_CONFLICT
            )

        try:
            answer = whoami(credential.token)
        except TokenRejected as ex:
            # Not a server error - it is the answer. A rejected token is
            # exactly what this endpoint exists to discover.
            return ok(
                {
                    "ok": False,
                    "reason": ex.message,
                    "status_code": ex.status_code,
                    "grants": grant_report(bot.entity_id, credential.scopes),
                }
            )
        except ChatterloopAPIError as ex:
            return fail(str(ex), status.HTTP_503_SERVICE_UNAVAILABLE)

        stamp = now()
        bot.verified_handle = (answer.get("handle") or "")[:50]
        bot.last_verified_at = stamp
        bot.save(update_fields=["verified_handle", "last_verified_at", "updated_at"])

        credential.last_verified_at = stamp
        credential.save(update_fields=["last_verified_at"])

        return ok(
            {
                "ok": True,
                "entity_id": answer.get("entity_id"),
                "handle": answer.get("handle"),
                "scopes": answer.get("scopes", []),
                "handle_mismatch": bot.handle_mismatch,
                # The half whoami cannot report on.
                "grants": grant_report(bot.entity_id, credential.scopes),
            }
        )


class BotTokenListView(BotViewMixin, OrganizationScopedView):

    def post(self, request, bot_id):
        """Issue a replacement credential.

        The existing ones stay live on purpose: rotation is "start using the
        new one, then cut off the old", and revoking first takes the bot
        offline for as long as a redeploy takes.
        """
        bot = self._bots().filter(id=bot_id).first()
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        serializer = RotateTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            credential = rotate_token(
                bot,
                name=serializer.validated_data.get("name") or None,
                scopes=serializer.validated_data.get("scopes"),
            )
        except ProvisioningError as ex:
            return fail(str(ex), status.HTTP_409_CONFLICT)

        payload = ChatterloopTokenSerializer(credential).data
        payload["token"] = credential.plaintext
        return ok(
            payload,
            status.HTTP_201_CREATED,
            message="Copy the token now - it is not shown again. The previous "
            "tokens are still live until you revoke them.",
        )


class BotTokenDetailView(BotViewMixin, OrganizationScopedView):

    def delete(self, request, bot_id, token_id):
        bot = self._bots().filter(id=bot_id).first()
        if bot is None:
            return fail("Bot not found.", status.HTTP_404_NOT_FOUND)

        credential = ChatterloopToken.objects.filter(id=token_id, bot=bot).first()
        if credential is None:
            return fail("Token not found.", status.HTTP_404_NOT_FOUND)
        if credential.revoked_at is not None:
            return ok(ChatterloopTokenSerializer(credential).data, message="Already revoked.")

        try:
            revoke_token(credential, reason=f"Revoked by @{request.user.username}.")
        except Exception as ex:
            logger.exception("failed to revoke token %s", credential.prefix)
            return fail(
                f"Could not revoke that token on Chatterloop: {ex}",
                status.HTTP_502_BAD_GATEWAY,
            )

        return ok(ChatterloopTokenSerializer(credential).data, message="Revoked.")


class HandleAvailabilityView(OrganizationScopedView):
    """Whether a handle is free, before the user fills in the rest of the form.

    Checks accounts and pages as well as bots, because `bot_bot.handle` is
    unique only among bots and the loser of a cross-table collision silently
    stops matching mentions.
    """

    def get(self, request):
        handle = (request.query_params.get("handle") or "").strip().lstrip("@")
        if not handle:
            return fail("Provide a handle to check.")

        try:
            conflict = handle_conflict(handle)
        except Exception:
            logger.exception("could not check handle availability")
            return fail(
                "Could not reach Chatterloop to check that handle.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return ok(
            {
                "handle": handle,
                "available": conflict is None,
                "taken_by": conflict or "",
            }
        )
