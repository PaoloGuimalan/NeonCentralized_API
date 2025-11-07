from user.models import Account, Verification
from django.utils.timezone import now
from .generators import make_id
from neon.settings import MAILINGSERVICE
import requests


def send_email_verification_code(from_email, to_email, subject, user_id):
    generated_code = make_id(6)  # your random code generator for your logic

    content = f"""
Welcome to ChatterLoop!

Your registration was successful! Here is your verification code for the account activation: {generated_code}
"""

    # Prepare payload to your mailing API
    payload = {
        "from": from_email,
        "email": to_email,
        "subject": subject,
        "content": content,
    }

    try:
        # response = requests.post(f"{MAILINGSERVICE}/sendEmail", json=payload)
        # if response.status_code == 200 and response.json().get("status") == True:
        # Save your Verification record here only if email sent
        ver_record = Verification(
            user=Account.objects.get(username=user_id),
            ver_code=generated_code,
            date_generated=now(),
            is_used=False,
        )
        ver_record.save()
        return True
    # else:
    #     print("Failed to send email:", response.text)
    #     return False
    except Exception as e:
        print("Error sending verification email:", e)
        return False
