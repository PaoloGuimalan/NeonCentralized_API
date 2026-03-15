from django.db import models


class TPAuthentication(models.Model):

    SERVICE_CHOICES = [
        ("google", "Google"),
    ]

    service_id = models.TextField(blank=False, null=False)
    service_type = models.CharField(max_length=150, null=False, choices=SERVICE_CHOICES)
