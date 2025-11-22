from django.contrib import admin
from .models import Agent, Role, Tool, Service, Model

admin.site.register(Agent)
admin.site.register(Role)
admin.site.register(Tool)
admin.site.register(Service)
admin.site.register(Model)
