import config
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Golden evaluation set.
#   request           = question
#   expected_response = concise reference answer, written ONLY from the four ingested
#                       source pages (Energiewende; Electricity sector in Germany;
#                       Federal Network Agency; German Renewable Energy Sources Act).
#   category          = "answerable" vs "out_of_scope" (score these groups separately).
eval_rows = [
    (
        "Which agency regulates Germany's electricity and gas networks, and which companies operate the German transmission network?",
        "Germany's electricity and gas networks are regulated by the Federal Network Agency (Bundesnetzagentur, or BNetzA), and the transmission network is operated by four transmission system operators: 50Hertz Transmission, Amprion, TenneT TSO, and TransnetBW.",
        "answerable",
    ),
    (
        "What is the Energiewende?",
        "The Energiewende is Germany's ongoing energy transition toward an energy system based mainly on renewable energy (especially wind, photovoltaics, and hydroelectricity), together with energy efficiency and energy demand management.",
        "answerable",
    ),
    (
        "What mechanisms has the German Renewable Energy Sources Act used to promote renewable electricity, and how did those mechanisms change over time?",
        "The EEG originally used guaranteed 20-year feed-in tariffs with grid connection and priority dispatch (funded by a consumer surcharge), and over time shifted toward the market by adding a market premium in 2012 and replacing fixed tariffs with competitive auctions and deployment corridors in the 2014 and 2017 revisions.",
        "answerable",
    ),
    (
        "Under the EEG, how does the Federal Network Agency help implement Germany's renewable-electricity expansion targets?",
        "Under the EEG, the Federal Network Agency runs the auctions/tenders for renewable installations and sets the auctioned capacity to the trajectory needed for Germany's renewables-share target, and it also sets payment levels, monitors the EEG equalisation scheme, and publishes newly installed renewable capacity each month.",
        "answerable",
    ),
    (
        "How does the Renewable Energy Sources Act relate to the Energiewende?",
        "The EEG is the main legislative instrument used to expand renewable electricity toward the goals of the Energiewende: the Energiewende sets the overall direction and targets, while the EEG (through feed-in tariffs and later auctions and deployment corridors) is the mechanism that grew renewable generation.",
        "answerable",
    ),
    (
        "According to the electricity-sector article, what factors explain Germany's high electricity prices, and what was the 2025 price-component breakdown?",
        "German household and small-business electricity prices are among the highest in Europe, and in 2025 the price consisted of roughly 40.5% generation, 32% taxes and duties, and 27.5% grid fees.",
        "answerable",
    ),
    (
        "What is the current spot price of electricity on the EPEX exchange today?",
        "This cannot be answered from the source documents, which explain how spot-market pricing works but contain no live or current spot price.",
        "out_of_scope",
    ),
    (
        "How does France's current nuclear-energy policy compare with Germany's Energiewende?",
        "This cannot be answered from the source documents, which do not describe France's current nuclear-energy policy and mention France only in passing.",
        "out_of_scope",
    ),
]

df = spark.createDataFrame(eval_rows, ["request", "expected_response", "category"])
df.write.mode("overwrite").saveAsTable(f"{config.FQ_SCHEMA}.eval_set")

print(f"Wrote {df.count()} eval questions to {config.FQ_SCHEMA}.eval_set")
df.select("request", "category").show(truncate=80)