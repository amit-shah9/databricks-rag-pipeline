import config
from vs_client import get_clients


_, vector_search_client = get_clients()


# Create the Vector Search endpoint only if it is not already available.
# Endpoint creation can take a while, so the script prints clear progress messages.
existing_endpoints = [
    endpoint["name"]
    for endpoint in vector_search_client.list_endpoints().get("endpoints", [])
]

if config.VS_ENDPOINT in existing_endpoints:
    print(f"Endpoint '{config.VS_ENDPOINT}' already exists.")
else:
    print(f"Creating endpoint '{config.VS_ENDPOINT}'...")
    vector_search_client.create_endpoint_and_wait(
        name=config.VS_ENDPOINT,
        endpoint_type="STANDARD",
    )
    print(f"Endpoint '{config.VS_ENDPOINT}' is ready.")


# The index is built from the chunks table created earlier in the pipeline.
# I use a triggered Delta Sync index here so the demo can be rebuilt manually.
existing_indexes = [
    index["name"]
    for index in vector_search_client.list_indexes(config.VS_ENDPOINT).get(
        "vector_indexes",
        []
    )
]

if config.INDEX_NAME in existing_indexes:
    print(f"Index '{config.INDEX_NAME}' already exists.")
else:
    print(f"Creating index '{config.INDEX_NAME}' and embedding the chunks...")

    index_config = {
        "endpoint_name": config.VS_ENDPOINT,
        "index_name": config.INDEX_NAME,
        "source_table_name": config.CHUNKS_TABLE,
        "pipeline_type": "TRIGGERED",
        "primary_key": "chunk_id",
        "embedding_source_column": "text",
        "embedding_model_endpoint_name": config.EMBED_ENDPOINT,
    }

    # Some Databricks workspaces require a budget policy for serving resources.
    if config.BUDGET_POLICY_ID:
        index_config["budget_policy_id"] = config.BUDGET_POLICY_ID

    vector_search_client.create_delta_sync_index_and_wait(**index_config)
    print(f"Index '{config.INDEX_NAME}' is ready.")