from django.urls import path

from llm import views

app_name = "llm"

urlpatterns = [
    path("agents", views.AgentListView.as_view(), name="agents"),
    path("agents/<str:agent_uuid>", views.AgentDetailView.as_view(), name="agent-detail"),
    path("roles", views.RoleListView.as_view(), name="roles"),
    path("roles/<int:role_id>", views.RoleDetailView.as_view(), name="role-detail"),
    path("tools", views.ToolListView.as_view(), name="tools"),
    path("tools/<int:tool_id>", views.ToolDetailView.as_view(), name="tool-detail"),
    path("services", views.ServiceListView.as_view(), name="services"),
    path("models", views.ModelListView.as_view(), name="models"),
    path("knowledge", views.KnowledgeListView.as_view(), name="knowledge"),
    path(
        "knowledge/<str:document_id>",
        views.KnowledgeDetailView.as_view(),
        name="knowledge-detail",
    ),
]
