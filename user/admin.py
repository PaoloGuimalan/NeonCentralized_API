from django.contrib import admin
from .models import Account, Verification, Token

admin.site.register(Account)
admin.site.register(Verification)
admin.site.register(Token)
