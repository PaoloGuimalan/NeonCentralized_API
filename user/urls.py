from django.conf.urls import include
from django.urls import re_path, path
from rest_framework import routers

from user import views

router = routers.DefaultRouter()

app_name = "user"

urlpatterns = [
    re_path("", include((router.urls, "user-routes"))),
    path(
        "auth/<str:username>/",
        views.UserAuthentication.as_view(),
        name="user-profile",
    ),
    path(
        "auth",
        views.UserAuthentication.as_view(),
        name="user-authentication",
    ),
]
