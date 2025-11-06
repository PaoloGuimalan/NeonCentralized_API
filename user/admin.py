from django.contrib import admin
from .models import Account, Organization

admin.site.register(Account)
admin.site.register(Organization)