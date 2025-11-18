from django.conf.urls import include
from django.urls import re_path, path
from rest_framework import routers

from messenger import views

router = routers.DefaultRouter()

app_name = "messenger"

urlpatterns = [
    re_path("", include((router.urls, "messenger-routes"))),
    path(
        "<str:conversation_id>/",
        views.MessagingView.as_view(),
        name="messenger-conversation",
    ),
    path(
        "list",
        views.MessagingListView.as_view(),
        name="messenger-list",
    ),
]
