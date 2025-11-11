from django.db import models
from user.models import Account
from organization.models import Organization

class Tool(models.Model):
    """
    Represents a callable function/tool agents can use.
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    parameters_schema = models.JSONField(blank=True, null=True)  # JSON schema for parameters
    api_endpoint = models.URLField(help_text="Tool callable API endpoint URL", blank=True, null=True)
    http_method = models.CharField(max_length=10, choices=[('GET', 'GET'), ('POST', 'POST')], default='POST')
    requires_auth = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class Role(models.Model):
    """
    Defines dynamic roles with system prompts and assigned tools.
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    system_prompt = models.TextField(help_text="System prompt or personality template for this role")
    tools = models.ManyToManyField(Tool, blank=True, related_name='roles')

    def __str__(self):
        return self.name


class Agent(models.Model):
    """
    Represents an agent belonging to an organization with a dynamic role.
    """
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='agents')
    role = models.ForeignKey(Role, on_delete=models.SET_NULL, null=True, blank=True, related_name='agents')
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(Account, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_agents')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        role_name = self.role.name if self.role else 'No Role'
        return f"{self.name} ({self.organization.name}, Role: {role_name})"
