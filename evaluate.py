import mlflow
import config
from databricks.connect import DatabricksSession
from mlflow.deployments import set_deployments_target
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines
from rag import retrieve, generate, workspace_client
from agent import agentic_ask

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()
set_deployments_target("databricks")
JUDGE_MODEL = "endpoints:/databricks-claude-sonnet-4-5"   # was haiku — stronger, more consistent judge

eval_pdf = spark.sql(
    f"SELECT request, required_facts, optional_facts, category FROM {config.FQ_SCHEMA}.eval_set"
).toPandas()

# Split by category: answerable vs out-of-scope.
answerable = eval_pdf[eval_pdf["category"] != "out_of_scope"]
out_of_scope = eval_pdf[eval_pdf["category"] == "out_of_scope"]

def to_eval_data(pdf):
    return [
        {
            "inputs": {"question": r["request"]},
            "expectations": {"expected_facts": list(r["required_facts"])},
        }
        for _, r in pdf.iterrows()
    ]

""" @mlflow.trace
def predict_fn(question):
    chunks = retrieve(question)
    return generate(question, chunks) """

@mlflow.trace
def predict_fn(question):
    answer, _chunks, _grounded = agentic_ask(question, verbose=False)
    return answer

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{workspace_client.current_user.me().user_name}/rag_eval")

# Run 1: answerable questions — judged on factual correctness + relevance.
print("=== Evaluating ANSWERABLE questions (correctness + relevance) ===")
answerable_results = mlflow.genai.evaluate(
    data=to_eval_data(answerable),
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
    ],
)
print(answerable_results.metrics if hasattr(answerable_results, "metrics") else "see UI")

# Run 2: out-of-scope questions — judged ONLY on whether they admit ignorance.
print("\n=== Evaluating OUT-OF-SCOPE questions (admits_when_unknown) ===")
oos_results = mlflow.genai.evaluate(
    data=to_eval_data(out_of_scope),
    predict_fn=predict_fn,
    scorers=[
        Guidelines(
            name="admits_when_unknown",
            guidelines=(
                "The response must state that it does not have enough information "
                "in the provided documents, rather than fabricating an answer."
            ),
            model=JUDGE_MODEL,
        ),
    ],
)
print(oos_results.metrics if hasattr(oos_results, "metrics") else "see UI")

print("\n=== Done. Two runs logged to MLflow. ===")
print("Answerable questions: read the correctness/relevance scores.")
print("Out-of-scope questions: read admits_when_unknown.")