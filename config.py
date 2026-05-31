"""Project configuration for the RAG pipeline.

Most values come from environment variables so I can keep workspace-specific
settings out of the public repo. For local runs, create a .env file from
.env.example and fill in the values for your own Databricks workspace.
"""

import os

from dotenv import load_dotenv


# Load local environment variables when running the project outside Databricks.
load_dotenv()


def _require(name):
    """Read a required environment variable and fail early if it is missing."""

    value = os.environ.get(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and add the value there."
        )

    return value


# Databricks cluster used by Databricks Connect.
CLUSTER_ID = _require("DATABRICKS_CLUSTER_ID")


# Unity Catalog location used for all project tables.
# Defaults are fine for a small demo, but can be changed in .env.
CATALOG = os.environ.get("CATALOG", "main")
SCHEMA = os.environ.get("SCHEMA", "rag_sprint")
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"


# Tables and Vector Search index created by the pipeline.
DOCUMENTS_TABLE = f"{FQ_SCHEMA}.documents"
CHUNKS_TABLE = f"{FQ_SCHEMA}.chunks"
INDEX_NAME = f"{FQ_SCHEMA}.chunks_index"


# Vector Search settings.
VS_ENDPOINT = os.environ.get("VS_ENDPOINT", "rag_sprint_vs")

# Optional because some Databricks workspaces require a budget policy and others do not.
BUDGET_POLICY_ID = os.environ.get("BUDGET_POLICY_ID")


# Model serving endpoints used for answering questions and creating embeddings.
CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT", "databricks-claude-haiku-4-5")
EMBED_ENDPOINT = os.environ.get("EMBED_ENDPOINT", "databricks-qwen3-embedding-0-6b")
