"""Shared helper to build an authenticated Vector Search client."""
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

def get_clients():
    w = WorkspaceClient()
    vsc = VectorSearchClient(
        workspace_url=w.config.host,
        personal_access_token=w.config.oauth_token().access_token,
        disable_notice=True,
    )
    return w, vsc