import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

ALGORITHMS = ["HS256"]
JWT_TOKEN = os.getenv("JWT_TOKEN")


class JWTTools:

    def decoder(token, options=None):
        decode_value = jwt.decode(
            token, JWT_TOKEN, algorithms=ALGORITHMS, options=options
        )
        return decode_value

    def encoder(value):
        expire_time = datetime.now(timezone.utc) + timedelta(
            days=30
        )  # 30 days expiration
        payload = {
            **value,
            "exp": expire_time,
            "iat": datetime.now(timezone.utc),
        }
        encoded_value = jwt.encode(payload, JWT_TOKEN, algorithm=ALGORITHMS[0])
        return encoded_value
