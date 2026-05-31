import re

import requests
from databricks.connect import DatabricksSession

import config


# Start a Databricks Connect session using the cluster configured in config.py.
# I keep these values outside the script so the code is easier to move between environments.
spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Make sure the target schema exists before writing any tables into it.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.FQ_SCHEMA}")


# Small set of Wikipedia pages used as source documents for the RAG demo.
# I chose pages around Germany's energy transition so the dataset has one clear theme.
TITLES = [
    "Energiewende",
    "Electricity sector in Germany",
    "Federal Network Agency",
    "Renewable Energy Sources Act",
]


def fetch_wikipedia(title):
    """Fetch the plain-text extract for a Wikipedia page."""

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        },
        headers={"User-Agent": "rag-sprint-learning/1.0"},
        timeout=30,
    )

    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))

    return page.get("extract")


rows = []

for title in TITLES:
    text = fetch_wikipedia(title)

    # Skip empty or very short pages so the table only contains useful documents.
    if text and len(text) > 200:
        doc_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        source_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        rows.append((doc_id, title, source_url, text))
        print(f"OK   {title}  ({len(text):,} chars)")
    else:
        print(f"SKIP {title}  (not found or too short)")


documents_df = spark.createDataFrame(rows, ["doc_id", "title", "source", "text"])

documents_df.write.mode("overwrite").saveAsTable(config.DOCUMENTS_TABLE)

print(f"\nWrote {documents_df.count()} documents to {config.DOCUMENTS_TABLE}")

spark.sql(
    f"""
    SELECT
        doc_id,
        title,
        length(text) AS chars
    FROM {config.DOCUMENTS_TABLE}
    """
).show(truncate=False)
