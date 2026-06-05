import config
from vs_client import get_clients

_, vsc = get_clients()

INDEX_NAME = f"{config.FQ_SCHEMA}.chunks_small_to_big_index"
SOURCE_TABLE = f"{config.FQ_SCHEMA}.chunks_small_to_big"

existing = [i["name"] for i in vsc.list_indexes(config.VS_ENDPOINT).get("vector_indexes", [])]
if INDEX_NAME in existing:
    print(f"Index '{INDEX_NAME}' already exists.")
else:
    print(f"Creating index '{INDEX_NAME}' and embedding 174 child chunks...")
    kwargs = dict(
        endpoint_name=config.VS_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",          # embed the SMALL child text
        embedding_model_endpoint_name=config.EMBED_ENDPOINT,
    )
    if config.BUDGET_POLICY_ID:
        kwargs["budget_policy_id"] = config.BUDGET_POLICY_ID
    vsc.create_delta_sync_index_and_wait(**kwargs)
    print(f"Index '{INDEX_NAME}' is ready.")