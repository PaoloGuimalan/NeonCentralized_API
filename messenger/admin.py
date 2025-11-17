from django.contrib import admin
from messenger.models import Conversation, Message

admin.site.register(Conversation)
admin.site.register(Message)
