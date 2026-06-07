import config
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Golden evaluation set — tiered facts.
#   request        = question
#   required_facts = minimum facts a correct answer MUST contain (scored on these)
#   optional_facts = enriching facts the question didn't demand (not scored; nice-to-have)
#   category       = answerable | out_of_scope
# Every fact is quote-checkable against the four source documents (no outside knowledge).
eval_rows = [
    (
        "What is the Bundesnetzagentur (Federal Network Agency), and which sectors does it regulate?",
        ["It is Germany's federal regulatory office for network-bound markets.",
         "It regulates electricity, gas, telecommunications, post, and railway markets."],
        ["It is a federal agency of the Federal Ministry for Economic Affairs and Climate Action.",
         "It is headquartered in Bonn.",
         "It is also referred to as BNetzA."],
        "answerable",
    ),
    (
        "When did the German Renewable Energy Sources Act (EEG) first come into force?",
        ["1 April 2000."],
        ["It was preceded by the Electricity Feed-in Act (1991), in force from 1 January 1991."],
        "answerable",
    ),
    (
        "When did Germany complete its nuclear phase-out?",
        ["In 2023."],
        ["The final three reactors — Emsland, Isar II, Neckarwestheim II — were disconnected on 15 April 2023.",
         "Their operation had been briefly extended due to the energy crisis following the 2022 Russian invasion of Ukraine."],
        "answerable",
    ),
    (
        "What were the three guiding principles of the original EEG (2000)?",
        ["Investment protection — guaranteed feed-in tariffs plus a grid-connection/priority-dispatch requirement.",
         "No charge to public finances — funded by a surcharge on electricity consumers rather than taxation.",
         "Innovation via decreasing feed-in tariffs (a 'degression' applied to new installations)."],
        ["Each kWh from a renewable facility received a technology-specific tariff guaranteed for 20 years.",
         "Renewable electricity received preferential dispatch over nuclear, coal, and gas.",
         "Degression was meant to apply downward cost pressure as technologies matured."],
        "answerable",
    ),
    (
        "What roles does the Federal Network Agency perform under the Renewable Energy Sources Act (EEG)?",
        ["Determines the level of financial payment for renewable-energy installations.",
         "Monitors the nationwide EEG equalisation scheme.",
         "Publishes the capacity of newly installed renewable installations.",
         "Conducts the auction/tender process for renewable-energy installations."],
        ["Publication of new-install capacity is monthly.",
         "The equalisation scheme runs between distribution network operators, transmission system operators, and electricity suppliers."],
        "answerable",
    ),
    (
        "Is the Federal Network Agency responsible for licensing energy companies and railway companies?",
        ["No. It is not responsible for licensing energy companies.",
         "It is not responsible for the technical supervision and licensing of railway companies."],
        ["Energy-company licensing remains with authorities determined by State law.",
         "Railway technical supervision/licensing rests with the Federal Railway Office (Eisenbahn-Bundesamt, EBA).",
         "Its actual energy/rail role is ensuring non-discriminatory network access and regulating fees."],
        "answerable",
    ),
    (
        "Where did the term 'Energiewende' originate?",
        ["It first appeared in the title of a 1980 publication by the Öko-Institut."],
        ["That publication called for the complete abandonment of nuclear and petroleum energy.",
         "Its most groundbreaking claim was that economic growth was possible without increased energy consumption.",
         "A 16 February 1980 Berlin symposium was titled 'Energiewende: Atomausstieg und Klimaschutz'.",
         "The term in its present form dates back to 2002."],
        "answerable",
    ),
    (
        "Who are Germany's four transmission system operators (TSOs)?",
        ["50Hertz Transmission, Amprion, TenneT (TSO), and TransnetBW."],
        ["This is the line-up as of July 2016.",
         "Ownership: 50Hertz (Elia, formerly Vattenfall); Amprion (RWE); TenneT (formerly E.ON); TransnetBW (subsidiary of EnBW)."],
        "answerable",
    ),
    (
        "What major change did the EEG 2014 introduce to how renewable-energy generators are paid?",
        ["It shifted away from fixed, government-set feed-in tariffs.",
         "Operators now market their electricity directly and receive a 'market premium'.",
         "Funding rates are increasingly set by auction/tender rather than fixed by government."],
        ["The market premium covers the gap between the fixed EEG payment and the average monthly wholesale (EEX) spot price.",
         "The act is nicknamed 'EEG 2.0'.",
         "The transition to auctions was completed with the EEG 2017.",
         "Small installations remained on conventional feed-in tariffs."],
        "answerable",
    ),
    (
        "Why did Germany's installed electricity-generation capacity grow much faster than its actual electricity generation between 2000 and 2019?",
        ["The added capacity was renewable, which has lower capacity factors than conventional plants, so installed capacity rose steeply while generation rose only slightly."],
        ["Installed capacity rose from 121 GW (2000) to 218 GW (2019), an 80% increase.",
         "Electricity generation rose only about 5% over the same period."],
        "answerable",
    ),
    (
        "What is the 'merit order effect' in the German electricity market?",
        ["It occurs when preferentially dispatched wind and solar generation displaces more expensive fossil-fuel generation from the margin, lowering the cleared/spot electricity price."],
        ["It is more pronounced for photovoltaics because their midday peak coincides with peak demand.",
         "It also reduces revenues for conventional plants, making them less viable.",
         "Studies estimated the combined wind+PV effect at ~0.5 ¢/kWh (2010) to over 1.1 ¢/kWh (2012)."],
        "answerable",
    ),
    (
        "What are the main criticisms of the Energiewende?",
        ["The principal criticisms: high costs; the early nuclear phase-out increasing carbon emissions; continued or increased use of fossil fuels; risks to power-supply stability; and the environmental damage of biomass."],
        ["The 2019 Federal Court of Auditors called the €160bn over five years 'in extreme disproportion to the results'.",
         "It has been described as 'expensive, chaotic, and unfair' and a 'massive failure' (as of 2019).",
         "The VKU warned of stability risks during prolonged low-wind/low-sun periods given near-absent storage.",
         "A study suggested phasing out coal before nuclear could have saved ~1,100 lives and €3–8bn/year.",
         "Taxes and fees were noted to make up roughly half of household bills."],
        "answerable",
    ),
    (
        "What happened to the EEG surcharge in 2022?",
        ["It was removed/abolished, effective 1 July 2022."],
        ["The average German household was expected to save around €200 per year.",
         "The costs would instead be met from emissions-trading proceeds and the federal budget.",
         "Guaranteed tariffs for renewables projects continue going forward."],
        "answerable",
    ),
    (
        "How did Germany's coal consumption and international ranking change over the period described?",
        ["Coal's share of electricity generation fell substantially over the period (from a large share around 2008 to roughly a quarter by 2020).",
         "Germany's global ranking as a coal consumer dropped over the decade (from among the top consumers around 2010 to lower by 2019)."],
        ["Coal supplied 291 TWh / 46% of generation in 2008, falling to 118 TWh / 24% in 2020.",
         "Germany was the 4th-largest coal consumer in 2010 and had fallen to 8th by 2019.",
         "It closed its last hard-coal mine in December 2018.",
         "Three large lignite open-pit mines remain (Garzweiler, Lausitzer, Oberlausitzer).",
         "A 2019 commission set a 2038 phase-out for the 84 remaining coal plants (Kohleausstieg)."],
        "answerable",
    ),
    (
        "What renewable-electricity-share targets did the 2009 EEG set, looking out to 2050?",
        ["At least 35% by 2020, 50% by 2030, 65% by 2040, and 80% by 2050 (share of total electricity production)."],
        ["This raised the previous 2020 target from 20% to 35%.",
         "The 80%-by-2050 figure aligns with the government's standing renewables target."],
        "answerable",
    ),
    (
        "What is the Energiewende, and what does it intend to rely on?",
        ["It is Germany's ongoing energy transition.",
         "It intends to rely heavily on renewable energy (particularly wind, photovoltaics, and hydroelectricity), energy efficiency, and energy demand management."],
        ["'Energiewende' translates as 'energy turnaround'.",
         "Legislative support passed in late 2010.",
         "It set GHG-reduction goals of 80–95% by 2050 (relative to 1990) and a 60% renewable target by 2050.",
         "It also entailed decentralisation and a 'democratization' of energy."],
        "answerable",
    ),
    (
        "Why has Germany increasingly relied on fossil gas as part of the Energiewende?",
        ["Because it is phasing out both nuclear and coal at the same time, it turned to fossil gas as a bridging/transition fuel to provide dispatchable capacity, cover the intermittency of wind and solar, and ensure supply security."],
        ["Officials argued Germany could not exit nuclear and coal simultaneously without more gas.",
         "Russian gas was initially viewed as 'safe, cheap, and temporary'.",
         "New or restarted gas plants were cited (Irsching 4 & 5; RWE near Biblis; Leipheim).",
         "A 2023 EWI estimate put the need at ~50 new gas plants (~€60bn).",
         "A February 2024 plan would subsidise 10 GW of hydrogen-ready gas plants.",
         "Critics note gas is largely methane and raises Russia-dependency concerns."],
        "answerable",
    ),
    (
        "What is the current (2026) wholesale price of electricity per MWh on the German day-ahead market?",
        ["The information is not contained in the documents; a correct response abstains/says it is not available."],
        [],
        "out_of_scope",
    ),
    (
        "Which manufacturer supplies the largest number of onshore wind turbines installed in Germany?",
        ["The information is not contained in the documents; a correct response abstains/says it is not available."],
        [],
        "out_of_scope",
    ),
]

df = spark.createDataFrame(
    eval_rows,
    ["request", "required_facts", "optional_facts", "category"],
)
df.write.mode("overwrite").saveAsTable(f"{config.FQ_SCHEMA}.eval_set")

print(f"Wrote {df.count()} eval questions to {config.FQ_SCHEMA}.eval_set")
df.groupBy("category").count().show()