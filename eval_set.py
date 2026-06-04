import config
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Golden evaluation set — 19 answerable + 2 out-of-scope.
#   request           = question
#   expected_response = reference answer, written ONLY from the four ingested
#                       source pages and auto-verified against them
#                       (see verify_eval_set.py — all 19 returned SUPPORTED).
#   category          = "answerable" vs "out_of_scope" (scored separately).
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
        "Under the EEG, the Federal Network Agency conducts the auction process for renewable energy installations, determines the level of financial payment for renewable installations, monitors the nationwide EEG equalisation scheme between distribution network operators, transmission system operators, and electricity suppliers, and publishes the capacity of newly installed renewable energy installations each month.",
        "answerable",
    ),
    (
        "How does the Renewable Energy Sources Act relate to the Energiewende?",
        "The EEG is the legislative mechanism that drives renewable-electricity expansion as part of the Energiewende. The 2014 revised EEG set deployment corridors defining how much renewable energy is to be expanded and shifted funding from government-fixed feed-in tariffs to auction-determined rates. This market redesign was seen as a key part of the Energiewende.",
        "answerable",
    ),
    (
        "According to the electricity-sector article, what factors explain Germany's high electricity prices, and what was the 2025 price-component breakdown?",
        "German household and small-business electricity prices are among the highest in Europe, and in 2025 the price consisted of roughly 40.5% generation, 32% taxes and duties, and 27.5% grid fees.",
        "answerable",
    ),
    (
        "What are the main targets of the Energiewende?",
        "Legislative support passed in late 2010 included greenhouse-gas reductions of 80-95% by 2050 (relative to 1990) and a renewable energy target of 60% by 2050. A later coalition introduced an intermediate target of 55-60% renewable share in gross electricity consumption by 2035, and the coalition formed after the 2021 elections set 65% of energy from renewables by 2030 and 80% by 2040, with 2% of land set aside for onshore wind and offshore wind capacity to increase to 75 GW.",
        "answerable",
    ),
    (
        "When and why did Germany phase out nuclear power, and was it controversial?",
        "Germany phased out nuclear power in 2023 as part of the Energiewende; the firm phase-out policy was established by the second Merkel cabinet in 2011 following the Fukushima accident. The last three plants (Emsland, Isar II, Neckarwestheim II) were shut down on 15 April 2023. It was controversial: by 2023 the early retirement was no longer supported by the general public, and energy experts feared it could negatively affect Germany's greenhouse-gas reduction goals.",
        "answerable",
    ),
    (
        "What are the main criticisms of the Energiewende?",
        "The Energiewende has been criticised for high costs, the early nuclear phase-out which increased carbon emissions, continued or increased fossil fuel use, risks to power-supply stability during lengthy periods of unsuitable weather, and the environmental damage of biomass. In 2019 Germany's Federal Court of Auditors found it had cost EUR 160 billion over five years and that expenses were in extreme disproportion to results; the program was perceived as expensive, chaotic, and unfair.",
        "answerable",
    ),
    (
        "What role does fossil gas play in the Energiewende?",
        "As nuclear and coal are phased out, the government promoted fossil gas as a bridging/transition fuel to cover the intermittency of renewables. New gas plants were deemed necessary to guarantee supply security, and in February 2024 the government agreed to subsidise 10 GW of hydrogen-ready gas plants expected to switch from gas to hydrogen between 2035 and 2040. Reliance on Russian gas was later criticised as an energy-security risk.",
        "answerable",
    ),
    (
        "What was Germany's electricity generation and renewable share in 2024?",
        "Germany produced 488.5 TWh of electricity in 2024, with 59.4% from renewable energy sources. By the end of 2024, renewables accounted for roughly 56% of generation.",
        "answerable",
    ),
    (
        "Who are Germany's four transmission system operators, and why don't the electricity producers own the grid?",
        "As of 2016 the four German TSOs are 50Hertz Transmission, Amprion, TenneT TSO, and TransnetBW. Producers do not own the grid because, per the European Commission, electricity producers should not own the grid to ensure open competition; E.ON was accused of market misuse in 2008 and consequently sold its share of the network.",
        "answerable",
    ),
    (
        "How does Germany trade electricity internationally?",
        "In 2021 Germany exported 57,000 GWh and imported 39,600 GWh of electricity; by 2024 exports were 57,400 GWh. Germany is the second-largest electricity exporter after France, representing about 10% of world electricity exports, and has grid interconnections with neighbouring countries representing 10% of domestic capacity.",
        "answerable",
    ),
    (
        "What does the Bundesnetzagentur regulate besides electricity and gas?",
        "The Federal Network Agency regulates five markets: electricity, gas, telecommunications, post, and railway. In telecommunications it manages the telephone numbering plan, the radio frequency spectrum, and licenses telephone companies; in post it licenses postal companies and ensures non-discriminatory access to facilities like PO boxes; in railway it ensures non-discriminatory access to railway infrastructure, including train schedules and track-slot allocation.",
        "answerable",
    ),
    (
        "How is the Bundesnetzagentur governed and where is it based?",
        "The agency is a federal agency of the Federal Ministry for Economic Affairs and Climate Action, headquartered in Bonn. Its Advisory Council consists of 16 members of the Bundestag and 16 representatives of the Bundesrat. Its presidents have been Klaus-Dieter Scheurle, Matthias Kurth, Jochen Homann, and Klaus Müller (2022-present).",
        "answerable",
    ),
    (
        "What was the Electricity Feed-in Act of 1991, and how did it relate to the EEG?",
        "The Electricity Feed-in Act (Stromeinspeisungsgesetz), in force from 1 January 1991, was the world's first green electricity feed-in tariff scheme. It obliged grid companies to connect renewable power plants, grant them priority dispatch, and pay a guaranteed feed-in tariff over 20 years. It preceded and was replaced by the EEG (2000), which built on its experience.",
        "answerable",
    ),
    (
        "What were the three core principles of the EEG (2000)?",
        "The EEG (2000), in force from 1 April 2000, had three principles: investment protection through guaranteed technology-specific feed-in tariffs for 20 years plus a grid-connection requirement and preferential dispatch; no charge to public finances (remuneration funded by an EEG surcharge on electricity consumers rather than taxation); and innovation through 'degression' - feed-in tariffs decreasing at regular intervals to pressure costs down.",
        "answerable",
    ),
    (
        "What was the EEG surcharge, and what happened to it?",
        "The EEG surcharge (EEG-Umlage) was a levy on electricity consumers that funded the feed-in tariff payments, based on the difference between the EEG tariffs paid and the sale price of renewable electricity on the EEX exchange. Electricity-intensive industries could be largely exempted under a special equalisation scheme. The surcharge was removed effective 1 July 2022, with payments since met from emissions-trading proceeds and the federal budget; the average household was expected to save around EUR 200 per year.",
        "answerable",
    ),
    (
        "Why did the EEG shift from feed-in tariffs to auctions, and what concerns were raised?",
        "The shift to auctions (begun with the EEG 2014, completed in EEG 2017) responded to criticism that fixed feed-in tariffs could be too expensive if set too high or stimulate too few installations if set too low, and to the European Commission's preference for market-based support. Concerns raised: economist Claudia Kemfert argued auctions would not reduce costs and would undermine planning security; NGOs and Greenpeace Energy warned auctions would disadvantage citizen cooperatives and small investors (tender preparation costing EUR 50,000-100,000, sunk if the bid fails), threatening the citizen participation behind public acceptance.",
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
df.groupBy("category").count().show()