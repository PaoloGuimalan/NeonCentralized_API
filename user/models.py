import random
import uuid
import secrets
from django.core.exceptions import ValidationError
from django.db import models, IntegrityError
from django.core.validators import EmailValidator
from django.utils.timezone import now

from neon.utils.identifiers import new_id


def generate_random_digit(digit):
    if digit < 1:
        raise ValueError("digit must be at least 1")
    start = 10 ** (digit - 1)
    end = 10**digit - 1
    return str(random.randint(start, end))


def generate_developer_token():
    """Default for Token.token.

    A named module-level function rather than a lambda because Django has to
    SERIALIZE field defaults into migration files, and it cannot serialize a
    lambda. While this was `default=lambda: secrets.token_hex(16)`,
    `makemigrations` raised `ValueError: Cannot serialize function: lambda` for
    the whole project - which is why `Token.name` and `Message.integration`
    were added to their models but never got migrations.
    """
    return secrets.token_hex(16)


class Account(models.Model):

    GENDER_CHOICES = [
        ("male", "Male"),
        ("female", "Female"),
        ("other", "Other"),
    ]

    id = models.CharField(
        max_length=150, default=new_id, unique=True, blank=True, primary_key=True
    )
    username = models.CharField(max_length=150, unique=True, blank=True)
    first_name = models.CharField(max_length=150, null=False)
    middle_name = models.CharField(max_length=150, default="N/A")
    last_name = models.CharField(max_length=150, null=False)
    birthdate = models.DateTimeField(null=True, blank=True)
    profile = models.CharField(default="none")
    gender = models.CharField(
        max_length=150, null=True, blank=True, choices=GENDER_CHOICES
    )
    email = models.EmailField(unique=True, validators=[EmailValidator()])
    password = models.CharField(
        max_length=400,
        null=True,
        blank=True,
        default=new_id,
        help_text=(
            "No longer read for sign-in - chatterloop owns credentials. Kept "
            "because auto-provisioned end-user rows still get one written."
        ),
    )
    date_created = models.DateTimeField(default=now)
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    is_default_user = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    join_type = models.CharField(default="system", max_length=150, null=False)

    # ---------------------------------------------------------- projection --
    #
    # Chatterloop owns identity; this row is a PROJECTION of it, refreshed at
    # sign-in. It is not authoritative for anything except Neon's own
    # relations - eleven of them, including two M2M tables on Message - which
    # is precisely why the projection exists rather than those relations
    # pointing at a table in another database. Django cannot enforce or join a
    # foreign key across connections.

    entity_id = models.CharField(
        max_length=40,
        null=True,
        blank=True,
        unique=True,
        default=None,
        db_index=True,
        help_text="The chatterloop entity this account mirrors.",
    )

    chatterloop_account_id = models.CharField(
        max_length=150,
        null=True,
        blank=True,
        default=None,
        help_text="chatterloop user_account.id, for correlating back.",
    )

    last_synced_at = models.DateTimeField(
        null=True,
        blank=True,
        default=None,
        help_text="When the projection was last refreshed from chatterloop.",
    )

    @property
    def can_sign_in(self):
        """Whether this row represents somebody who can log in to Neon.

        A NULL `entity_id` is not an error. ExternalChatView provisions
        accounts by email for a third-party app's end users
        (user/utils/user_manipulation.py::get_or_create_account); those people
        are not chatterloop users and never sign in, but their messages still
        need an author. The cutover to chatterloop-only auth applies to
        LOGIN, not to this table.
        """
        return bool(self.entity_id)

    def is_authenticated(self):
        return True

    USERNAME_FIELD = "username"  # Use the username field for login
    REQUIRED_FIELDS = ["email"]  # Email is required but not for login

    def clean(self):
        super().clean()

        if self.join_type == "system" and not self.password:
            raise IntegrityError("Password cannot be empty on system creation.")

    def save(self, *args, **kwargs):
        if not self.username:
            prefix = self.first_name.split(" ")[0] + "_"
            prefix = prefix.lower()
            max_attempts = 5
            for _ in range(max_attempts):
                initial_un = prefix + generate_random_digit(3)
                listified_un = list(initial_un)
                random.shuffle(listified_un)
                self.username = "".join(listified_un)
                try:
                    super().save(*args, **kwargs)
                    break
                except IntegrityError:
                    # Collision happened, reset and retry
                    self.username = None
            else:
                raise IntegrityError(
                    "Could not generate a unique user_id after several attempts."
                )
        else:
            super().save(*args, **kwargs)

    def __str__(self):
        return self.username


# Verification (email code confirmation) was removed when chatterloop became
# the identity provider: an account is verified on chatterloop, and Neon reads
# `is_verified` off the projection rather than running a second, independent
# confirmation for the same person. Its only writer,
# neon/utils/external_requests.py::send_email_verification_code, already had
# no callers.


class Token(models.Model):
    id = models.CharField(
        max_length=150, default=new_id, unique=True, primary_key=True
    )
    token = models.CharField(default=generate_developer_token, null=False)
    account = models.ForeignKey(Account, null=False, on_delete=models.DO_NOTHING)
    date_generated = models.DateTimeField(default=now)
    name = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Human-readable label for the integration/app this token belongs to.",
    )
