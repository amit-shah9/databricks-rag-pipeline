import mlflow
from databricks import agents
import config

mlflow.set_registry_uri("databricks-uc")   # register into Unity Catalog
mlflow.set_tracking_uri("databricks")

# The logged model URI from your last run:
LOGGED_URI = "models:/m-b2b3904c04a14b9eaab12f461c966169"

# UC three-level name: catalog.schema.model_name
UC_MODEL_NAME = f"{config.FQ_SCHEMA}.grounded_rag_agent"

# 1. Register the logged model as a UC model version
result = mlflow.register_model(model_uri=LOGGED_URI, name=UC_MODEL_NAME)
print(f"Registered {UC_MODEL_NAME} version {result.version}")

# 2. Deploy to a serving endpoint (the agents.deploy helper handles provisioning)
deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=result.version,
    budget_policy_id=config.BUDGET_POLICY_ID,
    scale_to_zero=True,
)
print("Deployment started. Endpoint:", deployment.endpoint_name)
print("It will take 10-30 min to come online. Check status in the Serving UI.")