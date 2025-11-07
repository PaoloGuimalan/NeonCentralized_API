from rest_framework import serializers
from .models import Account


class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = [
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

    def create(self, validated_data):
        password = validated_data.pop("password")
        account = Account(**validated_data)
        # Hash using Django's default password system or bcrypt before save
        account.set_password(password)
        account.save()
        return account

    def update(self, instance, validated_data):
        password = validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance
