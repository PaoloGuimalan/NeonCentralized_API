from django.urls import path

from organization import views

app_name = "organization"

urlpatterns = [
    # No id in these paths. The organization a request acts in comes from the
    # X-Organization header (see organization/tenancy.py), so a caller cannot
    # reach another tenant's data by editing a URL.
    path("", views.OrganizationListView.as_view(), name="organizations"),
    path("current", views.OrganizationDetailView.as_view(), name="organization-current"),
    path("members", views.MemberListView.as_view(), name="members"),
    path("members/<str:member_id>", views.MemberDetailView.as_view(), name="member-detail"),
    path("credentials", views.ProviderCredentialListView.as_view(), name="credentials"),
    path(
        "credentials/embedding",
        views.EmbeddingCredentialStatusView.as_view(),
        name="credentials-embedding",
    ),
    path(
        "credentials/<str:credential_id>",
        views.ProviderCredentialDetailView.as_view(),
        name="credential-detail",
    ),
]
