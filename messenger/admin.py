from django.contrib import admin
from messenger.models import Conversation, Message, Summary

admin.site.register(Conversation)
admin.site.register(Message)
admin.site.register(Summary)
