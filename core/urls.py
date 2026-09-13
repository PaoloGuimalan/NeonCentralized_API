from django.urls import path

from core import views

app_name = "core"

urlpatterns = [
    path(
        "connections",
        views.ConnectionsView.as_view(),
        name="connections",
    ),
    path(
        "connections/pages/available",
        views.AvailablePagesView.as_view(),
        name="connections-pages-available",
    ),
    path(
        "connections/pages",
        views.ConnectPageView.as_view(),
        name="connections-pages",
    ),
    path(
        "connections/<str:connection_id>",
        views.ConnectionDetailView.as_view(),
        name="connections-detail",
    ),
]
