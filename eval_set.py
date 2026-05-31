import config
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Golden evaluation set. request = question, expected_response = reference answer.
# category drives how each row is scored (answerable vs out_of_scope).
eval_rows = [
    (
        "Which agency regulates Germany's electricity and gas networks, and which companies operate the German transmission network?",
        "Germany's Federal Network Agency, the Bundesnetzagentur / BNetzA, is the regulatory office for electricity and gas markets. In electricity and gas, it is responsible for non-discriminatory third-party access to networks and regulation of fees; it does not operate the grid. The transmission network is operated by four TSOs: 50Hertz Transmission, Amprion, TenneT TSO, and TransnetBW.",
        "factual",
    ),
    (
        "What is the Energiewende?",
        "The Energiewende is Germany's ongoing energy transition. The intended energy system relies heavily on renewable energy, especially wind, photovoltaics, and hydropower, plus energy efficiency and demand management. Legislative support included greenhouse-gas reduction goals and renewable-energy targets, and Germany's nuclear phase-out is described as part of the Energiewende. It is also associated with decentralization, democratization of energy, and citizen participation, though the 'citizen-driven rather than utility-driven' framing is treated as disputed.",
        "factual",
    ),
    (
        "What mechanisms has the German Renewable Energy Sources Act used to promote renewable electricity, and how did those mechanisms change over time?",
        "The Renewable Energy Sources Act, or EEG, is a series of German laws that encouraged renewable electricity generation. The original EEG guaranteed renewable generators grid connection, preferential dispatch, and technology-specific feed-in tariffs for 20 years. It was funded through an EEG surcharge on electricity consumers. Later versions, especially the 2014 and 2017 reforms, shifted most technologies away from administratively set feed-in tariffs toward direct marketing, market premiums, and auctions/tenders. The EEG surcharge was removed in 2022, while guaranteed tariffs for renewable projects continued in some form.",
        "factual",
    ),
    (
        "Under the EEG, how does the Federal Network Agency help implement Germany's renewable-electricity expansion targets?",
        "The EEG sets renewable-electricity expansion goals and trajectories, such as the EEG 2014 targets of 40-45% renewable electricity by 2025, 55-60% by 2035, and more than 80% by 2050. The Federal Network Agency connects to these goals through its EEG roles: determining payment levels, monitoring the nationwide EEG equalisation process, publishing newly installed renewable capacity, and conducting renewable-energy auctions. Under the auction system, BNetzA calls tenders and sets auction capacity to match the renewable-expansion trajectory. It helps administer implementation; it is not the body that defines the whole Energiewende strategy.",
        "multi_hop",
    ),
    (
        "How does the Renewable Energy Sources Act relate to the Energiewende?",
        "The Energiewende is the broader German energy-transition project: more renewable energy, greater efficiency, demand management, greenhouse-gas reduction, and phase-out of nuclear power. The EEG is one of the key legal instruments used to implement the renewable-electricity part of that transition. It promoted renewable generation through guaranteed grid connection, preferential dispatch, feed-in tariffs, and later auctions/market premiums.",
        "relationship",
    ),
    (
        "According to the electricity-sector article, what factors explain Germany's high electricity prices, and what was the 2025 price-component breakdown?",
        "The article says German households and small businesses have paid some of Europe's highest electricity prices for many years. It links recent increases to the 2021-2023 global energy crisis, especially higher gas and electricity prices. For 2025, it gives this price-component breakdown: 32% taxes and duties, 27.5% grid fees, and 40.5% electricity generation. Historically, renewable support through the EEG surcharge also affected consumer bills, though the EEG surcharge was removed in 2022.",
        "numeric",
    ),
    (
        "What is the current spot price of electricity on the EPEX exchange today?",
        "The system should state it does not have enough information in the provided documents to give a current EPEX spot price, as that requires live market data.",
        "out_of_scope",
    ),
    (
        "How does France's current nuclear-energy policy compare with Germany's Energiewende?",
        "The system should state the documents do not provide enough information about France's nuclear-energy policy to make the comparison, though it may answer the German side.",
        "out_of_scope",
    ),
]

df = spark.createDataFrame(eval_rows, ["request", "expected_response", "category"])
df.write.mode("overwrite").saveAsTable(f"{config.FQ_SCHEMA}.eval_set")

print(f"Wrote {df.count()} eval questions to {config.FQ_SCHEMA}.eval_set")
df.select("request", "category").show(truncate=80)