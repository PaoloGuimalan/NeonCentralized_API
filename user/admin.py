from django.contrib import admin
from .models import Account, Organization, Member

admin.site.register(Account)
admin.site.register(Organization)
admin.site.register(Member)