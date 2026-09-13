from rest_framework import serializers

from neon.utils.bcrypt_tools import hash_password

from .models import Account


class AccountSerializer(serializers.ModelSerializer):
    # Write-only so a password never appears in a response. It was previously
    # absent from `fields` altogether, which made the `validated_data.pop(
    # "password")` below raise KeyError on every create.
    password = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = Account
        fields = [
            "password",
            "id",
            "username",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
            "profile",
            "gender",
            "email",
            "date_created",
            "is_active",
            "is_verified",
        ]
        read_only_fields = ["id", "date_created"]

    # Account is a plain models.Model, NOT an AbstractBaseUser, so it has no
    # set_password() - both methods below used to call it and would have
    # raised AttributeError on the first use. Hashing goes through the same
    # bcrypt helper that user/utils/user_manipulation.py::create_user uses, so
    # there is one hashing implementation and login keeps working against it.

    def create(self, validated_data):
        password = validated_data.pop("password", None)
        account = Account(**validated_data)
        if password:
            account.password = hash_password(password)
        account.save()
        return account

    def update(self, instance, validated_data):
        password = validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.password = hash_password(password)
        instance.save()
        return instance
