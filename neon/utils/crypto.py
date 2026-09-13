"""Encrypting secrets Neon has to be able to read back.

NOT HASHING, AND THE DIFFERENCE MATTERS
---------------------------------------
Passwords are hashed because nothing ever needs the original. These are the
opposite case: an organization's OpenAI key and a chatterloop token's secret
both have to be presented verbatim to somebody else, so they must be
recoverable. Encryption is therefore the only option, and the honest framing
is that this protects against a leaked DATABASE - a dump, a backup, a stray
read replica - not against an attacker who already has the application.

Fernet is AES-128-CBC with an HMAC and a timestamp, which is the boring
correct choice here: authenticated, versioned, and with no parameters to get
wrong.

THE KEY
-------
`TOKEN_ENCRYPTION_KEY` must be a urlsafe-base64 32-byte Fernet key, and must
NOT be `SECRET_KEY` reused - a value that appears in more places is a value
with more ways to leak, and rotating Django's secret should not silently make
every stored credential unreadable.

Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

ROTATION
--------
`TOKEN_ENCRYPTION_KEY_FALLBACKS` accepts previous keys, newest first. A
MultiFernet decrypts with any of them and always encrypts with the current
one, so rotating means prepending a new key, re-saving the affected rows at
leisure, and dropping the old key once nothing fails to decrypt. Without this,
rotation is a flag day.
"""

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

_cipher = None


class DecryptionFailed(Exception):
    """A stored value could not be decrypted.

    Almost always a key problem - rotated without a fallback, or a different
    deployment's key - rather than corruption. Raised rather than returning
    None so it cannot be mistaken for "no credential set", which would send a
    caller off to configure something that is already configured.
    """


def _build_cipher():
    key = getattr(settings, "TOKEN_ENCRYPTION_KEY", None)
    if not key:
        raise ImproperlyConfigured(
            "TOKEN_ENCRYPTION_KEY is required to store provider credentials. "
            "Generate one with: python -c \"from cryptography.fernet import "
            'Fernet; print(Fernet.generate_key().decode())"'
        )

    keys = [key] + list(getattr(settings, "TOKEN_ENCRYPTION_KEY_FALLBACKS", []))
    try:
        return MultiFernet([Fernet(k.encode() if isinstance(k, str) else k) for k in keys])
    except (ValueError, TypeError) as ex:
        raise ImproperlyConfigured(
            f"TOKEN_ENCRYPTION_KEY is not a valid Fernet key: {ex}"
        ) from ex


def get_cipher():
    """Built once per process, on first use.

    Lazily rather than at import so a deployment that has not set the key yet
    fails when it actually tries to store a credential - with the message
    above - instead of refusing to start Django at all.
    """
    global _cipher
    if _cipher is None:
        _cipher = _build_cipher()
    return _cipher


def encrypt(value):
    """Encrypt a string for storage. Empty input stores as empty."""
    if value is None or value == "":
        return ""
    return get_cipher().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(value):
    """Recover a stored string."""
    if not value:
        return ""
    try:
        return get_cipher().decrypt(value.encode("ascii")).decode("utf-8")
    except InvalidToken as ex:
        raise DecryptionFailed(
            "A stored credential could not be decrypted. This usually means "
            "TOKEN_ENCRYPTION_KEY changed without the previous key being "
            "listed in TOKEN_ENCRYPTION_KEY_FALLBACKS."
        ) from ex
