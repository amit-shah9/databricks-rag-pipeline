import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
import config

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/amit.shah@quantum.de/rag_eval")

# Resources the agent needs at serving time, declared so Databricks can
# provision auth to them automatically (the chat endpoint, judge, embeddings,
# and the vector index the document tool queries).
resources = [
    DatabricksServingEndpoint(endpoint_name=config.CHAT_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=config.EMBED_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=config.INDEX_NAME),
]

with mlflow.start_run(run_name="grounded_rag_agent"):
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent_responses.py",
        code_paths=["agent.py", "rag.py", "config.py", "vs_client.py"],
        resources=resources,
        pip_requirements=[
            "mlflow>=3.12.0",
            "openai",
            "ddgs",
            "databricks-sdk",
            "databricks-vectorsearch",
            "databricks-connect==16.4.*",
        ],
    )
    print("Logged model URI:", logged.model_uri)

# quick load-back test (Step 5) — prove the crate is self-contained
print("\n--- loading back from MLflow ---")
loaded = mlflow.pyfunc.load_model(logged.model_uri)
result = loaded.predict({"input": [{"role": "user", "content": "What is the Energiewende?"}]})
print("Loaded model answered:", str(result)[:300])