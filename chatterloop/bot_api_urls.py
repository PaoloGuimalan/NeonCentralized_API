"""The public bot API: `/api/bots/...`.

WHY THESE ARE NOT UNDER /api/chatterloop/
-----------------------------------------
Everything in `urls.py` is Neon's dashboard talking about a bot's chatterloop
identity - minting it, binding an agent, listing its tokens. This file is a
different thing: a small, stable surface that anything can integrate with.

A bot is a general integration, not a chatterloop feature. The URL somebody
copies into a cron job, a deploy script or another product should not name the
platform that happens to carry the bot's messages today - and a path that says
`chatterloop` would have to keep saying it even after a second platform is
added, or break every copy that was already pasted somewhere.

The views live in the chatterloop app because that is where a bot's rows and
its supervisor live. Only the addressing is neutral.
"""

from django.urls import path

from chatterloop import views

app_name = "bots"

urlpatterns = [
    # The control plane: ?action=wake|sleep, authenticated by the bot's own
    # control key rather than a session - see BotControlView.
    path("<str:bot_id>/control", views.BotControlView.as_view(), name="control"),
    # Issues and rotates the key the route above accepts. A session route:
    # who may hold a bot's control key is a Neon question, even though the
    # endpoint it unlocks is not.
    path(
        "<str:bot_id>/control-key",
        views.BotControlKeyView.as_view(),
        name="control-key",
    ),
]
