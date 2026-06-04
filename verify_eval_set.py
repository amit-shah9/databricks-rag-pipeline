"""
verify_eval_set.py — automated grounding check for the evaluation set.

For each candidate reference answer, this sends the COMPLETE source documents
plus the answer to a strong LLM judge and asks it to flag any factual claim
that is NOT supported by the documents. This catches the failure mode that has
bitten this project repeatedly: reference answers that assert facts true in the
world but absent from the corpus.

You read a short PASS/FAIL report instead of reading the documents yourself.
Anything flagged UNSUPPORTED gets fixed or dropped before it enters the eval set.

Reusable pattern: "LLM-as-auditor" — grounding generated content against sources.
"""
import re
import config
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.connect import DatabricksSession

# A strong judge for accurate grounding checks. Change if you prefer another.
JUDGE_MODEL = "databricks-claude-opus-4-5"

w = WorkspaceClient()
spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# ---- 1. Load the COMPLETE corpus (all four documents) -------------------
docs = spark.sql(
    f"SELECT doc_id, text FROM {config.DOCUMENTS_TABLE} ORDER BY doc_id"
).collect()
CORPUS = "\n\n".join(f"===== DOCUMENT: {d['doc_id']} =====\n{d['text']}" for d in docs)
print(f"Loaded {len(docs)} documents, {len(CORPUS):,} chars total.\n")

# ---- 2. Candidate reference answers to verify ---------------------------
# (question, answer). Only the answer's factual claims are audited.
CANDIDATES = [
    ("Q01 regulator/TSOs",
     "Germany's electricity and gas networks are regulated by the Federal Network Agency (Bundesnetzagentur, or BNetzA), and the transmission network is operated by four transmission system operators: 50Hertz Transmission, Amprion, TenneT TSO, and TransnetBW."),
    ("Q02 Energiewende definition",
     "The Energiewende is Germany's ongoing energy transition toward an energy system based mainly on renewable energy (especially wind, photovoltaics, and hydroelectricity), together with energy efficiency and energy demand management."),
    ("Q03 EEG mechanisms over time",
     "The EEG originally used guaranteed 20-year feed-in tariffs with grid connection and priority dispatch (funded by a consumer surcharge), and over time shifted toward the market by adding a market premium in 2012 and replacing fixed tariffs with competitive auctions and deployment corridors in the 2014 and 2017 revisions."),
    ("Q04 multi-hop BNetzA/EEG roles",
     "Under the EEG, the Federal Network Agency conducts the auction process for renewable energy installations, determines the level of financial payment for renewable installations, monitors the nationwide EEG equalisation scheme between distribution network operators, transmission system operators, and electricity suppliers, and publishes the capacity of newly installed renewable energy installations each month."),
    ("Q05 EEG-Energiewende relationship",
     "The EEG is the legislative mechanism that drives renewable-electricity expansion as part of the Energiewende. The 2014 revised EEG set deployment corridors defining how much renewable energy is to be expanded and shifted funding from government-fixed feed-in tariffs to auction-determined rates. This market redesign was seen as a key part of the Energiewende."),
    ("Q06 price breakdown 2025",
     "German household and small-business electricity prices are among the highest in Europe, and in 2025 the price consisted of roughly 40.5% generation, 32% taxes and duties, and 27.5% grid fees."),
    ("Q07 Energiewende targets",
     "Legislative support passed in late 2010 included greenhouse-gas reductions of 80-95% by 2050 (relative to 1990) and a renewable energy target of 60% by 2050. A later coalition introduced an intermediate target of 55-60% renewable share in gross electricity consumption by 2035, and the coalition formed after the 2021 elections set 65% of energy from renewables by 2030 and 80% by 2040, with 2% of land set aside for onshore wind and offshore wind capacity to increase to 75 GW."),
    ("Q08 nuclear phase-out",
     "Germany phased out nuclear power in 2023 as part of the Energiewende; the firm phase-out policy was established by the second Merkel cabinet in 2011 following the Fukushima accident. The last three plants (Emsland, Isar II, Neckarwestheim II) were shut down on 15 April 2023. It was controversial: by 2023 the early retirement was no longer supported by the general public, and energy experts feared it could negatively affect Germany's greenhouse-gas reduction goals."),
    ("Q09 Energiewende criticisms",
     "The Energiewende has been criticised for high costs, the early nuclear phase-out which increased carbon emissions, continued or increased fossil fuel use, risks to power-supply stability during lengthy periods of unsuitable weather, and the environmental damage of biomass. In 2019 Germany's Federal Court of Auditors found it had cost EUR 160 billion over five years and that expenses were in extreme disproportion to results; the program was perceived as expensive, chaotic, and unfair."),
    ("Q10 fossil gas role",
     "As nuclear and coal are phased out, the government promoted fossil gas as a bridging/transition fuel to cover the intermittency of renewables. New gas plants were deemed necessary to guarantee supply security, and in February 2024 the government agreed to subsidise 10 GW of hydrogen-ready gas plants expected to switch from gas to hydrogen between 2035 and 2040. Reliance on Russian gas was later criticised as an energy-security risk."),
    ("Q11 2024 generation and renewable share",
     "Germany produced 488.5 TWh of electricity in 2024, with 59.4% from renewable energy sources. By the end of 2024, renewables accounted for roughly 56% of generation."),
    ("Q12 four TSOs and why producers don't own grid",
     "As of 2016 the four German TSOs are 50Hertz Transmission, Amprion, TenneT TSO, and TransnetBW. Producers do not own the grid because, per the European Commission, electricity producers should not own the grid to ensure open competition; E.ON was accused of market misuse in 2008 and consequently sold its share of the network."),
    ("Q13 international electricity trade",
     "In 2021 Germany exported 57,000 GWh and imported 39,600 GWh of electricity; by 2024 exports were 57,400 GWh. Germany is the second-largest electricity exporter after France, representing about 10% of world electricity exports, and has grid interconnections with neighbouring countries representing 10% of domestic capacity."),
    ("Q14 BNetzA other markets",
     "The Federal Network Agency regulates five markets: electricity, gas, telecommunications, post, and railway. In telecommunications it manages the telephone numbering plan, the radio frequency spectrum, and licenses telephone companies; in post it licenses postal companies and ensures non-discriminatory access to facilities like PO boxes; in railway it ensures non-discriminatory access to railway infrastructure, including train schedules and track-slot allocation."),
    ("Q15 BNetzA governance",
     "The agency is a federal agency of the Federal Ministry for Economic Affairs and Climate Action, headquartered in Bonn. Its Advisory Council consists of 16 members of the Bundestag and 16 representatives of the Bundesrat. Its presidents have been Klaus-Dieter Scheurle, Matthias Kurth, Jochen Homann, and Klaus Müller (2022-present)."),
    ("Q16 Electricity Feed-in Act 1991",
     "The Electricity Feed-in Act (Stromeinspeisungsgesetz), in force from 1 January 1991, was the world's first green electricity feed-in tariff scheme. It obliged grid companies to connect renewable power plants, grant them priority dispatch, and pay a guaranteed feed-in tariff over 20 years. It preceded and was replaced by the EEG (2000), which built on its experience."),
    ("Q17 EEG 2000 three principles",
     "The EEG (2000), in force from 1 April 2000, had three principles: investment protection through guaranteed technology-specific feed-in tariffs for 20 years plus a grid-connection requirement and preferential dispatch; no charge to public finances (remuneration funded by an EEG surcharge on electricity consumers rather than taxation); and innovation through 'degression' - feed-in tariffs decreasing at regular intervals to pressure costs down."),
    ("Q18 EEG surcharge",
     "The EEG surcharge (EEG-Umlage) was a levy on electricity consumers that funded the feed-in tariff payments, based on the difference between the EEG tariffs paid and the sale price of renewable electricity on the EEX exchange. Electricity-intensive industries could be largely exempted under a 'special equalisation scheme.' The surcharge was removed effective 1 July 2022, with payments since met from emissions-trading proceeds and the federal budget; the average household was expected to save around EUR 200 per year."),
    ("Q19 auction shift and concerns",
     "The shift to auctions (begun with the EEG 2014, completed in EEG 2017) responded to criticism that fixed feed-in tariffs could be too expensive if set too high or stimulate too few installations if set too low, and to the European Commission's preference for market-based support. Concerns raised: economist Claudia Kemfert argued auctions would not reduce costs and would undermine planning security; NGOs and Greenpeace Energy warned auctions would disadvantage citizen cooperatives and small investors (tender preparation costing EUR 50,000-100,000, sunk if the bid fails), threatening the citizen participation behind public acceptance."),
]

# ---- 3. The audit prompt ------------------------------------------------
PROMPT_TEMPLATE = (
    "You are auditing a reference answer for a Q&A evaluation set. Below are the "
    "COMPLETE source documents, then a QUESTION and a proposed ANSWER.\n\n"
    "Check EVERY factual claim in the ANSWER against the source documents. A claim "
    "is SUPPORTED only if the documents state it. Numbers must match in value "
    "(treat '40,5' and '40.5' as the same; ignore pure formatting/spelling/comma "
    "differences). If a claim is not in the documents or contradicts them, it is "
    "unsupported.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "VERDICT: SUPPORTED\n"
    "ISSUES: none\n"
    "-- or --\n"
    "VERDICT: UNSUPPORTED\n"
    "ISSUES: <each unsupported or contradicted claim on its own line>\n\n"
    "===== SOURCE DOCUMENTS =====\n{corpus}\n\n"
    "===== QUESTION =====\n{question}\n\n"
    "===== ANSWER =====\n{answer}\n"
)

def audit(question, answer):
    prompt = PROMPT_TEMPLATE.format(corpus=CORPUS, question=question, answer=answer)
    resp = w.serving_endpoints.query(
        name=JUDGE_MODEL,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=500,
    )
    text = resp.choices[0].message.content
    verdict = "UNSUPPORTED" if re.search(r"VERDICT:\s*UNSUPPORTED", text, re.I) else (
        "SUPPORTED" if re.search(r"VERDICT:\s*SUPPORTED", text, re.I) else "UNCLEAR")
    return verdict, text.strip()

# ---- 4. Run the audit and report ----------------------------------------
print("Auditing each reference answer against the full corpus...\n")
results = []
for label, answer in CANDIDATES:
    verdict, detail = audit(label, answer)
    results.append((label, verdict, detail))
    print(f"[{verdict:>11}]  {label}")

print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)
problems = [r for r in results if r[1] != "SUPPORTED"]
if not problems:
    print(" All reference answers are fully supported by the corpus. ")
else:
    print(f" {len(problems)} answer(s) have unsupported claims — review below:\n")
    for label, verdict, detail in problems:
        print(f"--- {label} [{verdict}] ---")
        # print just the ISSUES portion
        m = re.search(r"ISSUES:\s*(.+)", detail, re.I | re.S)
        print((m.group(1).strip() if m else detail)[:600])
        print()
print("=" * 70)