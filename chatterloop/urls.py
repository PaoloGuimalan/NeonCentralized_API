from django.urls import path

from chatterloop import views

app_name = "chatterloop"

urlpatterns = [
    path("bots", views.BotListView.as_view(), name="bots"),
    # Before the <bot_id> route, or "handle" would be read as a bot id.
    path(
        "bots/handle-available",
        views.HandleAvailabilityView.as_view(),
        name="handle-available",
    ),
    path("bots/<str:bot_id>", views.BotDetailView.as_view(), name="bot-detail"),
    path("bots/<str:bot_id>/online", views.BotOnlineView.as_view(), name="bot-online"),
    path(
        "bots/<str:bot_id>/reactivate",
        views.BotReactivateView.as_view(),
        name="bot-reactivate",
    ),
    path("bots/<str:bot_id>/verify", views.BotVerifyView.as_view(), name="bot-verify"),
    path("bots/<str:bot_id>/tokens", views.BotTokenListView.as_view(), name="bot-tokens"),
    path(
        "bots/<str:bot_id>/tokens/<str:token_id>",
        views.BotTokenDetailView.as_view(),
        name="bot-token-detail",
    ),
]
