"""Minting bots and tokens.

These run against real tables (chatterloop/testing.py) rather than mocks. A
mock cannot catch a column Neon writes that does not exist, a NOT NULL it does
not fill, or a transaction that does not roll back - which are the failures
that matter when one service writes into another's database.
"""

import hashlib
import uuid
from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils.timezone import now

from chatterloop.external_models import (
    Account as ChatterloopAccount,
    Bot,
    Entity,
    EntityPermission,
    Realm,
    Token,
)
from chatterloop.models import ChatterloopBot, ChatterloopToken
from chatterloop.provisioning import (
    ALL_SCOPES,
    HandleUnavailable,
    ProvisioningError,
    deactivate_bot,
    generate_token,
    grant_report,
    handle_conflict,
    hash_token,
    mint_bot,
    reactivate_bot,
    revoke_token,
    rotate_token,
    validate_scopes,
)
from chatterloop.testing import create_chatterloop_schema
from neon.testing import make_account, make_organization
from neon.utils import crypto


def make_chatterloop_account(username, entity_id=None):
    entity_id = entity_id or str(uuid.uuid4())
    Entity.objects.create(id=entity_id, type=Entity.USER, created_at=now())
    # is_active / is_verified have no model default: these are chatterloop's
    # columns and the projection deliberately does not invent defaults for a
    # table it does not own.
    return ChatterloopAccount.objects.create(
        id=str(uuid.uuid4()),
        entity_id=entity_id,
        username=username,
        first_name=username.title(),
        last_name="Tester",
        email=f"{username}@example.com",
        profile="none",
        is_active=True,
        is_verified=True,
        join_type="system",
    )


def make_realm(slug, entity_id=None, realm_type=Realm.PAGE):
    entity_id = entity_id or str(uuid.uuid4())
    Entity.objects.create(id=entity_id, type=Entity.REALM, created_at=now())
    return Realm.objects.create(
        id=str(uuid.uuid4()),
        realm_id=str(uuid.uuid4()),
        entity_id=entity_id,
        name=slug.title(),
        slug=slug,
        profile="none",
        type=realm_type,
        is_active=True,
    )


class ProvisioningTestCase(TestCase):
    """Base fixture: both databases, and a schema to write into."""

    databases = {"default", "chatterloop"}

    def setUp(self):
        crypto._cipher = None
        create_chatterloop_schema()
        self.alice = make_account("alice", entity_id="entity-alice")
        self.org = make_organization(self.alice, "Acme", "acme")

    def tearDown(self):
        crypto._cipher = None

    def mint(self, handle="helper", **kwargs):
        kwargs.setdefault("name", "Helper")
        kwargs.setdefault("owner_entity_id", self.alice.entity_id)
        return mint_bot(
            organization=self.org,
            created_by=self.alice,
            handle=handle,
            **kwargs,
        )


class TokenFormatTests(TestCase):
    """The format three other implementations already agree on."""

    def test_shape_matches_the_go_parser(self):
        """developer_service splits on "_" and checks the two hex lengths.

        Anything else is rejected before a database round trip, so getting
        this wrong produces a token that can never authenticate.
        """
        prefix, secret, token, _ = generate_token()
        self.assertRegex(token, r"^clt_[0-9a-f]{12}_[0-9a-f]{64}$")
        self.assertEqual(token, f"clt_{prefix}_{secret}")
        self.assertEqual(len(prefix), 12)
        self.assertEqual(len(secret), 64)

    def test_hash_is_sha256_of_the_whole_string(self):
        """Not of the secret half. Verified against the Go implementation,
        which hashes `raw` - the entire `clt_..._...` string."""
        _, _, token, digest = generate_token()
        self.assertEqual(digest, hashlib.sha256(token.encode("utf-8")).hexdigest())
        self.assertEqual(len(digest), 64)

    def test_tokens_are_not_predictable(self):
        tokens = {generate_token()[2] for _ in range(50)}
        self.assertEqual(len(tokens), 50)

    def test_hash_is_stable_for_a_known_value(self):
        self.assertEqual(
            hash_token("clt_000000000000_" + "0" * 64),
            hashlib.sha256(("clt_000000000000_" + "0" * 64).encode()).hexdigest(),
        )


class ScopeValidationTests(TestCase):

    def test_unknown_scopes_are_refused(self):
        """developer_service refuses a codename outside its five rather than
        guessing, so a typo would mint a token that authenticates and is then
        refused by every route."""
        with self.assertRaises(ProvisioningError) as caught:
            validate_scopes(["messages.send", "messages.delete"])
        self.assertIn("messages.delete", str(caught.exception))

    def test_empty_means_the_default_set(self):
        self.assertEqual(validate_scopes([]), list(ALL_SCOPES))
        self.assertEqual(validate_scopes(None), list(ALL_SCOPES))

    def test_scopes_are_deduplicated_and_ordered(self):
        self.assertEqual(
            validate_scopes(["messages.send", "messages.send", "events.subscribe"]),
            ["events.subscribe", "messages.send"],
        )


class HandleConflictTests(ProvisioningTestCase):
    """`bot_bot.handle` is unique among bots ONLY."""

    def test_a_free_handle_has_no_conflict(self):
        self.assertIsNone(handle_conflict("brand-new"))

    def test_an_existing_account_username_conflicts(self):
        """The gap that matters. The database would accept this bot, and then
        developer_service - which resolves handles by UNION across accounts,
        realms and bots, first match winning - would silently stop matching
        mentions for one of them."""
        make_chatterloop_account("taken")
        self.assertEqual(handle_conflict("taken"), "a Chatterloop account")

    def test_an_existing_page_slug_conflicts(self):
        make_realm("acmepage")
        self.assertEqual(handle_conflict("acmepage"), "a Chatterloop page")

    def test_an_existing_bot_handle_conflicts(self):
        self.mint(handle="helper")
        self.assertEqual(handle_conflict("helper"), "another bot")

    def test_the_check_is_case_insensitive(self):
        make_chatterloop_account("Taken")
        self.assertIsNotNone(handle_conflict("taken"))


class MintingTests(ProvisioningTestCase):

    def test_minting_writes_all_four_chatterloop_rows(self):
        bot, credential = self.mint()

        entity = Entity.objects.get(id=bot.entity_id)
        self.assertEqual(entity.type, Entity.BOT)

        chatterloop_bot = Bot.objects.get(id=bot.bot_id)
        self.assertEqual(chatterloop_bot.handle, "helper")
        self.assertEqual(chatterloop_bot.owner_entity_id, self.alice.entity_id)
        self.assertTrue(chatterloop_bot.is_active)
        # Never minted as a platform bot.
        self.assertFalse(chatterloop_bot.is_system)

        token = Token.objects.get(id=credential.token_id)
        self.assertEqual(token.entity_id, bot.entity_id)
        self.assertTrue(token.is_active)
        self.assertIsNone(token.revoked_at)
        # Unlimited, and both halves NULL together.
        self.assertIsNone(token.rate_limit_int)
        self.assertIsNone(token.rate_limit_type)

    def test_scopes_and_grants_are_written_together(self):
        """Authorization is an intersection: a scope with no grant row is a
        silent 403 that looks exactly like the reverse mistake."""
        bot, credential = self.mint()

        token = Token.objects.get(id=credential.token_id)
        self.assertEqual(sorted(token.scopes), sorted(ALL_SCOPES))

        grants = EntityPermission.objects.filter(
            entity_id=bot.entity_id, effect=EntityPermission.GRANT, realm_id=None
        ).values_list("permission", flat=True)
        self.assertEqual(sorted(grants), sorted(ALL_SCOPES))

    def test_a_narrow_scope_set_grants_only_those(self):
        bot, credential = self.mint(scopes=["events.subscribe", "messages.read"])
        grants = EntityPermission.objects.filter(entity_id=bot.entity_id).values_list(
            "permission", flat=True
        )
        self.assertEqual(sorted(grants), ["events.subscribe", "messages.read"])

    def test_the_stored_hash_matches_the_returned_token(self):
        _, credential = self.mint()
        token = Token.objects.get(id=credential.token_id)
        self.assertEqual(token.token_hash, hash_token(credential.plaintext))

    def test_the_secret_is_recoverable_from_neon_and_nowhere_else(self):
        """Chatterloop keeps only a hash, so Neon holding the plaintext is the
        only thing that lets a bot reconnect without re-minting."""
        _, credential = self.mint()
        stored = ChatterloopToken.objects.get(pk=credential.pk)
        self.assertEqual(stored.token, credential.plaintext)
        # And never in the clear at rest.
        self.assertNotIn(credential.plaintext, stored.secret_encrypted)

    def test_a_taken_handle_is_refused_before_anything_is_written(self):
        make_chatterloop_account("taken")
        with self.assertRaises(HandleUnavailable):
            self.mint(handle="taken")

        self.assertFalse(ChatterloopBot.objects.exists())
        self.assertFalse(Bot.objects.exists())
        self.assertFalse(Entity.objects.filter(type=Entity.BOT).exists())

    def test_a_failed_chatterloop_write_leaves_a_recoverable_neon_row(self):
        """The ordering rule. Neon writes its record first, so a failure names
        exactly which chatterloop entity to clean up - rather than leaving a
        live credential nothing knows about and nothing can revoke.
        """
        with patch(
            "chatterloop.provisioning._write_chatterloop_bot",
            side_effect=RuntimeError("pg is down"),
        ):
            with self.assertRaises(ProvisioningError):
                self.mint()

        bot = ChatterloopBot.objects.get()
        self.assertEqual(bot.status, ChatterloopBot.STATUS_FAILED)
        # The reason reaches the user, not just a log file.
        self.assertIn("pg is down", bot.status_reason)
        # And the entity id is recorded, so the orphan is nameable.
        self.assertTrue(bot.entity_id)

        credential = ChatterloopToken.objects.get()
        # Never provisioned, so never usable.
        self.assertIsNone(credential.provisioned_at)
        self.assertFalse(credential.is_live)

    def test_the_chatterloop_write_is_all_or_nothing(self):
        """A half-written bot is worse than none: an Entity with no Bot row
        renders as a raw uuid everywhere it appears."""
        with patch(
            "chatterloop.provisioning._grant_scopes",
            side_effect=RuntimeError("grant failed"),
        ):
            with self.assertRaises(ProvisioningError):
                self.mint()

        self.assertFalse(Entity.objects.filter(type=Entity.BOT).exists())
        self.assertFalse(Bot.objects.exists())
        self.assertFalse(Token.objects.exists())

    def test_a_page_can_own_a_bot(self):
        """The payoff: owner_entity is an FK to Entity, so a realm works as
        happily as a person - and the bot outlives its creator leaving."""
        realm = make_realm("acmepage")
        bot, _ = self.mint(handle="pagebot", owner_entity_id=realm.entity_id)
        self.assertEqual(
            Bot.objects.get(id=bot.bot_id).owner_entity_id, realm.entity_id
        )


class RotationAndRevocationTests(ProvisioningTestCase):

    def test_rotating_leaves_the_previous_token_live(self):
        """Rotation is "start using the new one, then cut off the old".
        Revoking first takes the bot offline for as long as a redeploy takes.
        """
        bot, first = self.mint()
        second = rotate_token(bot)

        self.assertNotEqual(first.prefix, second.prefix)
        self.assertTrue(Token.objects.get(id=first.token_id).is_active)
        self.assertTrue(Token.objects.get(id=second.token_id).is_active)
        self.assertEqual(bot.tokens.count(), 2)

    def test_rotating_re_grants_widened_scopes(self):
        bot, _ = self.mint(scopes=["events.subscribe"])
        rotate_token(bot, scopes=list(ALL_SCOPES))

        grants = EntityPermission.objects.filter(entity_id=bot.entity_id).values_list(
            "permission", flat=True
        )
        self.assertEqual(sorted(grants), sorted(ALL_SCOPES))

    def test_re_granting_does_not_duplicate_rows(self):
        """Postgres treats NULLs as distinct in a unique index, so the
        (entity, permission, realm) constraint does NOT stop duplicate global
        rows on its own."""
        bot, _ = self.mint()
        rotate_token(bot)
        self.assertEqual(
            EntityPermission.objects.filter(entity_id=bot.entity_id).count(),
            len(ALL_SCOPES),
        )

    def test_revoking_marks_both_sides(self):
        _, credential = self.mint()
        revoke_token(credential, reason="test")

        token = Token.objects.get(id=credential.token_id)
        self.assertFalse(token.is_active)
        self.assertIsNotNone(token.revoked_at)
        self.assertIsNotNone(credential.revoked_at)
        self.assertFalse(credential.is_live)

    def test_deactivating_a_bot_revokes_every_token(self):
        """A deactivated bot with a live credential is a bot the user believes
        is off and that still authenticates."""
        bot, first = self.mint()
        second = rotate_token(bot)

        deactivate_bot(bot, reason="disconnected")

        self.assertEqual(bot.status, ChatterloopBot.STATUS_DEACTIVATED)
        self.assertFalse(Bot.objects.get(id=bot.bot_id).is_active)
        for credential in (first, second):
            self.assertFalse(Token.objects.get(id=credential.token_id).is_active)

    def test_reactivating_does_not_restore_tokens(self):
        """A revoked credential may have leaked; un-revoking would defeat the
        point."""
        bot, credential = self.mint()
        deactivate_bot(bot)
        reactivate_bot(bot)

        self.assertEqual(bot.status, ChatterloopBot.STATUS_ACTIVE)
        self.assertTrue(Bot.objects.get(id=bot.bot_id).is_active)
        self.assertFalse(Token.objects.get(id=credential.token_id).is_active)


class GrantReportTests(ProvisioningTestCase):
    """The half of the intersection a token cannot report about itself."""

    def test_a_scope_with_no_grant_is_reported_missing(self):
        bot, _ = self.mint(scopes=["events.subscribe"])
        report = grant_report(bot.entity_id, ["events.subscribe", "messages.send"])
        self.assertEqual(report["granted"], ["events.subscribe"])
        self.assertEqual(report["missing"], ["messages.send"])

    def test_an_explicit_deny_beats_a_grant(self):
        """Matching the resolver's first rule: a deny short-circuits."""
        bot, _ = self.mint()
        EntityPermission.objects.create(
            id=str(uuid.uuid4()),
            entity_id=bot.entity_id,
            permission="messages.send",
            effect=EntityPermission.DENY,
            realm_id=None,
            created_at=now(),
        )
        report = grant_report(bot.entity_id, ["messages.send"])
        self.assertEqual(report["granted"], [])
        self.assertEqual(report["missing"], ["messages.send"])

    def test_an_expired_grant_does_not_count(self):
        bot, _ = self.mint(scopes=["events.subscribe"])
        EntityPermission.objects.filter(entity_id=bot.entity_id).update(
            expires_at=now() - timedelta(days=1)
        )
        report = grant_report(bot.entity_id, ["events.subscribe"])
        self.assertEqual(report["missing"], ["events.subscribe"])


class RoutingTests(ProvisioningTestCase):
    """The projections must read from chatterloop's database, not Neon's.

    Worth an explicit test because the failure is SILENT: Neon's own
    `user.Account` is physically `user_account` too, so a projection that lost
    its routing would query a real, existing table with overlapping column
    names and return the wrong rows rather than erroring.
    """

    def test_external_models_are_pinned_to_the_chatterloop_alias(self):
        for model in (Entity, Bot, Token, EntityPermission, ChatterloopAccount, Realm):
            with self.subTest(model=model.__name__):
                self.assertEqual(model.objects.all().db, "chatterloop")

    def test_neon_owned_models_stay_in_neon(self):
        for model in (ChatterloopBot, ChatterloopToken):
            with self.subTest(model=model.__name__):
                self.assertEqual(model.objects.all().db, "default")

    def test_the_two_user_account_tables_are_not_the_same_table(self):
        from user.models import Account as NeonAccount

        make_chatterloop_account("only-on-chatterloop")
        self.assertFalse(
            NeonAccount.objects.filter(username="only-on-chatterloop").exists()
        )
        self.assertTrue(
            ChatterloopAccount.objects.filter(username="only-on-chatterloop").exists()
        )


class BotRecordTests(ProvisioningTestCase):

    def test_handle_mismatch_is_detected(self):
        """A bot configured with the wrong handle matches no mentions, and
        nothing in any log says why."""
        bot, _ = self.mint()
        self.assertFalse(bot.handle_mismatch)

        bot.verified_handle = "something-else"
        self.assertTrue(bot.handle_mismatch)

    def test_a_masked_token_never_contains_the_secret(self):
        from chatterloop.serializers import ChatterloopTokenSerializer

        _, credential = self.mint()
        data = ChatterloopTokenSerializer(credential).data
        self.assertNotIn("token", data)
        self.assertNotIn(credential.plaintext, str(data))
        self.assertTrue(data["masked"].startswith(f"clt_{credential.prefix}_"))

    def test_two_organizations_can_each_hold_a_handle_record(self):
        """Neon's per-org constraint is about its own bookkeeping; chatterloop's
        global uniqueness is what actually prevents a duplicate handle, and is
        checked separately."""
        other_org = make_organization(make_account("bob"), "Beta", "beta")
        self.mint(handle="helper")
        # A second org cannot reuse it either, because the chatterloop check
        # runs first and sees the existing bot.
        with self.assertRaises(HandleUnavailable):
            mint_bot(
                organization=other_org,
                created_by=self.alice,
                name="Helper",
                handle="helper",
                owner_entity_id=self.alice.entity_id,
            )
