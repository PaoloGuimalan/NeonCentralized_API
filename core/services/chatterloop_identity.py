"""Signing in through chatterloop, and keeping Neon's projection of it.

THE SHAPE, IN ONE LINE
----------------------
Chatterloop authenticates; Neon mirrors the result and issues its own session.

WHY FORWARD RATHER THAN CHECK THE HASH OURSELVES
------------------------------------------------
Neon holds chatterloop's database credentials, so it *could* read
`user_account.password` and run bcrypt against it. That would be materially
worse: it bypasses every rule chatterloop applies around a login - account
deactivation, verification state, whatever lockout exists now or later - and it
turns Neon's sign-in form into a password oracle for chatterloop, reachable by
anyone who can reach Neon. Forwarding keeps one implementation of "is this
person who they say they are", and it stays in the service that owns the
answer.

The password is never persisted, never logged, and is dropped as soon as the
forwarded request returns.

WHY THE BROWSER IS NOT INVOLVED
-------------------------------
This call is made by Neon's backend, so its result needs no verification: Neon
sent the request and read the reply. An earlier design had a popup on
chatterloop's domain hand the outcome to Neon's frontend, which then told
Neon's backend what had happened - and a backend cannot distinguish its own
frontend from `curl`, so that design needed a shared secret or a server-side
cross-check to be safe. Calling chatterloop directly removes the question.

The cost, stated plainly: the user types a chatterloop password into a Neon
form. An OAuth authorize flow on chatterloop is the only thing that removes
that, and this is structured so the swap touches only `authenticate()`.
"""

import logging
import secrets
from dataclasses import dataclass

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import Error as DatabaseFailure
from django.db import IntegrityError, transaction
from django.utils.timezone import now

# From `external_models`, not from `chatterloop.models`. Both re-export it,
# but only this module is unambiguous about WHICH `Account` is meant - see the
# note on the alias in chatterloop/models.py.
from chatterloop.external_models import Account as ChatterloopAccount
from chatterloop.models import Realm, RealmMember
from neon.utils.generators import generate_unique_username
from neon.utils.jwt_tools import JWTTools
from user.models import Account

logger = logging.getLogger(__name__)

PASSWORD_PATH = "/api/user/auth"
GOOGLE_PATH = "/api/user/tp_auth"


class ChatterloopAuthError(Exception):
    """Chatterloop refused the sign-in.

    `message` is chatterloop's own wording, passed through deliberately: it
    knows why (deactivated, unverified, underage, consent outstanding) and
    restating those rules in Neon would be a second copy that drifts.
    """

    def __init__(self, message, status_code=401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ChatterloopUnavailable(Exception):
    """Chatterloop could not be reached at all.

    Kept distinct from a refusal so the user is not told their password is
    wrong when the truth is that a service is down - the single most
    misleading error this flow could produce.
    """


# --------------------------------------------------------------- forwarding --


def _forward(path, payload):
    """POST to chatterloop's own auth endpoint and return `result`.

    Both endpoints are AllowAny with `get_authenticators()` returning [], so
    none of chatterloop's session stack applies here - no origin, no x-nonce,
    no x-access-token. A `device-token` header is required and is the only
    header that carries meaning.

    The WHOLE `result` comes back, not just the entity id it used to return.
    Chatterloop puts its own serialization of the account it just authenticated
    in `result["usertoken"]`, and that payload is what lets a sign-in be
    completed when the chatterloop DATABASE is not readable from here - see
    `_identity_from_login`.
    """
    url = f"{settings.CHATTERLOOP_API_BASE_URL}{path}"
    headers = {
        # Random per sign-in. Chatterloop records a device session against
        # whatever is sent here; a random value keeps Neon's sessions from
        # colliding with each other or with the user's real devices.
        "device-token": f"neon-{secrets.token_hex(16)}",
        # So the session is recognisable in the user's own device list rather
        # than appearing as an unexplained login from a datacentre.
        "User-Agent": settings.CHATTERLOOP_USER_AGENT,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=settings.CHATTERLOOP_HTTP_TIMEOUT,
        )
    except requests.RequestException as ex:
        logger.error("chatterloop auth unreachable", extra={"error": str(ex)})
        raise ChatterloopUnavailable(
            "Could not reach Chatterloop to sign you in. Please try again."
        ) from ex

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code >= 400 or not body.get("status"):
        message = body.get("message") or "Chatterloop rejected the sign-in."
        # 5xx is chatterloop failing, not the credential failing. Telling the
        # user their password is wrong in that case sends them to reset a
        # perfectly good one.
        if response.status_code >= 500:
            logger.error(
                "chatterloop auth server error",
                extra={"status": response.status_code},
            )
            raise ChatterloopUnavailable(
                "Chatterloop is having trouble signing you in. Please try again."
            )
        raise ChatterloopAuthError(message, status_code=response.status_code or 401)

    result = body.get("result") or {}
    if not result.get("personal_entity_id"):
        # A 200 with no entity id is a contract change, not a user error.
        logger.error("chatterloop auth returned no personal_entity_id")
        raise ChatterloopUnavailable(
            "Chatterloop returned an unexpected response. Please try again."
        )
    return result


def authenticate_with_password(email_username, password):
    return _forward(
        PASSWORD_PATH, {"email_username": email_username, "password": password}
    )


def authenticate_with_google(id_token):
    """Forward a Google ID token to chatterloop's third-party endpoint.

    Note that chatterloop checks the token's `azp` against its OWN
    `core_tpauthentication` table, so Neon's Google client id must have a row
    there or every Google sign-in fails with a lookup error that says nothing
    about the cause.
    """
    return _forward(GOOGLE_PATH, {"token": id_token})


# --------------------------------------------------------------- projection --


def _resolve_username(desired, entity_id):
    """Give the incoming chatterloop user their own username.

    `Account.username` is unique in Neon, and an auto-provisioned end-user row
    (ExternalChatView) may already be sitting on the name this person owns on
    chatterloop. Chatterloop's username wins - it is the one they recognise and
    the one shown everywhere else - so the local squatter is renamed rather
    than the real owner being suffixed into something they would not recognise.
    """
    clash = Account.objects.filter(username=desired).exclude(entity_id=entity_id).first()
    if clash is not None:
        clash.username = generate_unique_username(clash.first_name or "user")
        clash.save(update_fields=["username"])
        logger.info(
            "renamed a local account to free a chatterloop username",
            extra={"username": desired, "renamed_to": clash.username},
        )
    return desired


@dataclass(frozen=True)
class RemoteIdentity:
    """Chatterloop's account, in the only shape Neon's projection needs.

    A value object rather than either source's own type, so `sync_account` has
    one thing to project and does not branch on where the identity came from.
    """

    entity_id: str
    account_id: str
    username: str
    first_name: str
    middle_name: str
    last_name: str
    email: str
    profile: str
    is_active: bool
    is_verified: bool


def _identity_from_database(entity_id):
    """Identity as chatterloop's own tables hold it, or None.

    The authoritative source and still the first one tried: it is current at
    the moment it is read, where the login response is a snapshot of one
    instant.

    Returns None rather than raising for BOTH of its failure modes. A missing
    row and an unreachable database are equally not the signing-in user's
    problem - chatterloop has already said who they are - so each falls through
    to `_identity_from_login` instead of ending a sign-in that succeeded.
    """
    try:
        remote = ChatterloopAccount.objects.get(entity_id=entity_id)
    except ChatterloopAccount.DoesNotExist:
        logger.warning(
            "no chatterloop account row for entity; falling back to the login "
            "response",
            extra={"entity": entity_id},
        )
        return None
    except (DatabaseFailure, ImproperlyConfigured):
        # The chatterloop alias is optional configuration (CHATTERLOOP_DB_* in
        # settings) and a deployment can perfectly well have the API base URL
        # without it - or have it pointed at a different environment than the
        # API. Either way this raised straight out of the view and turned an
        # accepted sign-in into a 500, which is the bug the fallback exists for.
        logger.warning(
            "chatterloop database unreadable; falling back to the login response",
            exc_info=True,
            extra={"entity": entity_id},
        )
        return None

    return RemoteIdentity(
        entity_id=entity_id,
        account_id=str(remote.id),
        username=remote.username,
        first_name=remote.first_name,
        middle_name=remote.middle_name or "N/A",
        last_name=remote.last_name,
        email=remote.email,
        profile=remote.profile or "none",
        is_active=bool(remote.is_active),
        is_verified=bool(remote.is_verified),
    )


def _identity_from_login(entity_id, login_result):
    """Identity as the login response itself reported it, or None.

    `result["usertoken"]` is chatterloop's own serialization of the account it
    has just authenticated - `AccountSerializer` plus `entity_id` - which is
    everything the projection writes.

    NOT SIGNATURE-CHECKED, DELIBERATELY
    -----------------------------------
    Neon made this request, to a URL Neon configured, over TLS, and read the
    reply; the transport is the trust boundary and a signature adds nothing to
    it (the module docstring makes the same point about the response as a
    whole). Verifying would also make sign-in depend on Neon and chatterloop
    sharing a JWT secret, which is a coupling neither service needs and which
    fails closed and silently the day one of them rotates.
    """
    token = login_result.get("usertoken")
    if not token:
        return None

    try:
        payload = JWTTools.decoder(token, options={"verify_signature": False})
    except Exception:
        logger.error("could not read the chatterloop usertoken", exc_info=True)
        return None

    email = payload.get("email")
    if not email:
        # Without it there is nothing to adopt an existing row by and nothing
        # to satisfy Account.email, so this is not a usable identity.
        logger.error("chatterloop usertoken carried no email")
        return None

    return RemoteIdentity(
        entity_id=entity_id,
        account_id=str(payload.get("id") or ""),
        username=payload.get("username") or "",
        first_name=payload.get("first_name") or "",
        middle_name=payload.get("middle_name") or "N/A",
        last_name=payload.get("last_name") or "",
        email=email,
        profile=payload.get("profile") or "none",
        is_active=bool(payload.get("is_active", True)),
        is_verified=bool(payload.get("is_verified", False)),
    )


def _remote_identity(login_result):
    """Who chatterloop says just signed in - database first, response second."""
    entity_id = str(login_result["personal_entity_id"])

    identity = _identity_from_database(entity_id)
    if identity is None:
        identity = _identity_from_login(entity_id, login_result)

    if identity is None:
        # Both sources failed. Chatterloop accepted the credential but nothing
        # readable from here says who it belongs to, so there is no projection
        # to write - and that is an outage, not a bad password.
        logger.error(
            "chatterloop identity could not be resolved from either source",
            extra={"entity": entity_id},
        )
        raise ChatterloopUnavailable(
            "Your Chatterloop account could not be loaded. Please try again."
        )
    return identity


def sync_account(login_result):
    """Create or refresh the local projection of a chatterloop account.

    AUTO-IMPORT, NOT LOOK-UP
    ------------------------
    Anyone chatterloop authenticates gets a Neon `Account` here, whether or not
    Neon has seen them before. Chatterloop owns who exists; Neon's row is a
    projection of that answer, so "we have no record of you" is never a reason
    to refuse a sign-in chatterloop already accepted.
    """
    identity = _remote_identity(login_result)
    entity_id = identity.entity_id

    if not identity.is_active:
        raise ChatterloopAuthError("This Chatterloop account is deactivated.", 403)

    with transaction.atomic():
        account = Account.objects.filter(entity_id=entity_id).first()
        if account is None:
            # Fall back to email so an account that predates the cutover is
            # adopted rather than duplicated - Account.email is unique, so a
            # second row for the same person would fail anyway.
            account = Account.objects.filter(email=identity.email).first()

        if account is None:
            account = Account(
                email=identity.email,
                # No usable password: this row can only ever be signed into
                # through chatterloop.
                password=None,
                is_default_user=True,
            )

        # Left to Account.save()'s own generator when chatterloop sent none, so
        # a blank username produces a usable Neon handle rather than colliding
        # every such row onto "".
        if identity.username:
            account.username = _resolve_username(identity.username, entity_id)

        account.entity_id = entity_id
        account.chatterloop_account_id = identity.account_id or None
        account.first_name = identity.first_name
        account.middle_name = identity.middle_name
        account.last_name = identity.last_name
        account.email = identity.email
        account.profile = identity.profile
        account.is_active = identity.is_active
        account.is_verified = identity.is_verified
        account.join_type = "chatterloop"
        account.last_synced_at = now()

        try:
            account.save()
        except IntegrityError:
            logger.exception(
                "projection save conflicted", extra={"entity": entity_id}
            )
            raise ChatterloopUnavailable(
                "Your Chatterloop account could not be linked. Please try again."
            )

    return account


def sign_in_with_password(email_username, password):
    return sync_account(authenticate_with_password(email_username, password))


def sign_in_with_google(id_token):
    return sync_account(authenticate_with_google(id_token))


# ------------------------------------------------------------- page realms --


def available_pages(entity_id):
    """Chatterloop pages this identity may act as.

    The condition is NOT Neon's: `EntitySwitch`
    (user/entity_switch_views.py:73-87) permits acting as a realm only when it
    is a page and the caller's membership role is owner or admin. Applying the
    same test here means the two services cannot disagree about who may speak
    for a page - and it is checked server-side against chatterloop's own
    membership table, so it cannot be asserted by a client.

    `EntitySwitch` itself is not callable from Neon: it runs the full session
    stack, including an `x-nonce` whose AES key is derived from chatterloop's
    JWT_TOKEN. Reading the same rows is the equivalent that needs no shared
    secret.
    """
    memberships = (
        RealmMember.objects.filter(
            entity_id=entity_id,
            role__in=RealmMember.ACTING_ROLES,
            realm__type=Realm.PAGE,
            realm__is_active=True,
        )
        .select_related("realm")
        .order_by("realm__name")
    )

    return [
        {
            "realm_id": m.realm.realm_id or m.realm.id,
            "entity_id": m.realm.entity_id,
            "name": m.realm.name,
            "slug": m.realm.slug or "",
            "profile": m.realm.profile or "none",
            "role": m.role,
        }
        for m in memberships
    ]


def may_act_as(entity_id, realm_entity_id):
    """Whether this identity may still act as that page, right now.

    Re-checked at connect time AND at bot creation: a role can be revoked
    after a page was connected, and a stale ConnectedAccount row must not
    outlive the authority it was granted under.
    """
    return RealmMember.objects.filter(
        entity_id=entity_id,
        role__in=RealmMember.ACTING_ROLES,
        realm__entity_id=realm_entity_id,
        realm__type=Realm.PAGE,
        realm__is_active=True,
    ).exists()
