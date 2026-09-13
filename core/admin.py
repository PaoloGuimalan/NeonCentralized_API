from django.contrib import admin
from .models import ConnectedAccount


@admin.register(ConnectedAccount)
class ConnectedAccountAdmin(admin.ModelAdmin):
    list_display = (
        "account",
        "provider",
        "external_type",
        "external_username",
        "external_id",
        "is_active",
    )
    list_filter = ("provider", "external_type", "is_active")
    search_fields = ("external_id", "external_username", "external_name")
