from django.contrib import admin
from .models import Agent, Role, Tool

admin.site.register(Agent)
admin.site.register(Role)
admin.site.register(Tool)
