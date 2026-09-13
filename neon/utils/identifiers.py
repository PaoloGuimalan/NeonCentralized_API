"""Identifiers for models that store a uuid in a CharField.

THE BUG THIS EXISTS TO FIX
--------------------------
Several models were declared like this:

    id = models.CharField(max_length=150, default=uuid.uuid4, ...)

`uuid.uuid4` is a FUNCTION returning a `UUID` object, and the field is a
CharField. Django writes it as text - `get_prep_value` calls `str()` - so the
database is fine. What is not fine is Python:

    created = Account.objects.create(...)      # .pk is a UUID object
    fetched = Account.objects.get(pk=created.pk)  # .pk is a str
    created == fetched                          # False

Django's `Model.__eq__` compares primary keys directly, so two references to
THE SAME ROW compare unequal depending on how you got hold of them. The same
applies to any `in` test, any dict or set keyed on an instance, and any
`assertEqual` between a created object and a fetched one.

It is a quiet bug. Nothing raises; a comparison simply answers False, and code
that looks obviously correct - `if message.conversation.organization ==
request_organization` - takes the wrong branch.

THE FIX
-------
Return a string, so the in-memory value is the same type the database gives
back. `str(uuid.uuid4())` produces exactly the format `CharField` was already
storing - 36 characters, lowercase, hyphenated - so existing rows need no
migration and new rows are indistinguishable from them.

WHY NOT SWITCH TO UUIDField
---------------------------
Because it would be a much larger change for a worse fit. `UUIDField` holds a
`UUID` object in Python, so every place that compares an id against a string
from a URL, a header or a JSON body would need converting, and a data
migration would have to rewrite the column type on a live table. The models
that legitimately use `UUIDField` (messenger's) are already consistent and are
left alone.

A NAMED FUNCTION, NOT A LAMBDA
------------------------------
Django serialises field defaults into migration files and cannot serialise a
lambda. `user.models.generate_developer_token` carries the same note: a lambda
default once broke `makemigrations` for the WHOLE project, which is why two
models had fields with no migration behind them.
"""

import uuid


def new_id():
    """A fresh identifier, as a string.

    The `str()` is the entire point - see the module docstring. Removing it
    reintroduces a bug that makes an object stop equalling itself.
    """
    return str(uuid.uuid4())
