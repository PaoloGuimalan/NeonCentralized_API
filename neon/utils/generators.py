from user.models import Account
import random


def generate_random_digit(digit):
    if digit < 1:
        raise ValueError("digit must be at least 1")
    start = 10 ** (digit - 1)
    end = 10**digit - 1
    return str(random.randint(start, end))


def make_id(length=10):
    digits = "0123456789"
    return "".join(random.choices(digits, k=length))


def generate_unique_username(base_name):
    prefix = base_name.lower().replace(" ", "") + "_"
    max_attempts = 5
    for _ in range(max_attempts):
        random_digits = "".join(random.choices("0123456789", k=3))
        shuffled = list(prefix + random_digits)
        random.shuffle(shuffled)
        candidate = "".join(shuffled)
        if not Account.objects.filter(username=candidate).exists():
            return candidate
    raise ValueError("Could not generate unique username after several attempts")
