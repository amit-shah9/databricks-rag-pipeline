"""
verify_eval_set_tiered.py — automated grounding check for the TIERED eval set.

Each question now has required_facts and optional_facts (lists). This sends the
COMPLETE source documents plus each individual fact to a strong LLM judge and asks
whether that specific fact is supported by the documents. Catches the failure mode
where a fact reads plausibly but isn't actually in the corpus (inflating scores).

You read a short report instead of reading the documents yourself.
Out-of-scope rows are checked differently: their 'required_fact' is an abstention
instruction, so we instead verify the corpus genuinely LACKS the answer.
"""
import re
import config
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.connect import DatabricksSession

JUDGE_MODEL = "databricks-claude-opus-4-5"

w = WorkspaceClient()
spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# ---- Load full corpus ----
docs = spark.sql(
    f"SELECT doc_id, text FROM {config.DOCUMENTS_TABLE} ORDER BY doc_id"
).collect()
CORPUS = "\n\n".join(f"===== DOCUMENT: {d['doc_id']} =====\n{d['text']}" for d in docs)
print(f"Loaded {len(docs)} documents, {len(CORPUS):,} chars total.\n")

# ---- Load the tiered eval set ----
rows = spark.sql(
    f"SELECT request, required_facts, optional_facts, category FROM {config.FQ_SCHEMA}.eval_set"
).collect()

FACT_PROMPT = (
    "Below are COMPLETE source documents, then a QUESTION and a single FACT that "
    "is part of a reference answer to that question. Determine whether the FACT is "
    "explicitly supported by the documents. Read the fact in the context of the "
    "question. Numbers must match in value (ignore formatting). If the fact is "
    "stated or directly entailed by the documents, it is SUPPORTED.\n\n"
    "Respond in EXACTLY this format:\n"
    "VERDICT: SUPPORTED|UNSUPPORTED\n"
    "NOTE: <brief reason if unsupported, else 'ok'>\n\n"
    "===== SOURCE DOCUMENTS =====\n{corpus}\n\n"
    "===== QUESTION =====\n{question}\n\n"
    "===== FACT =====\n{fact}\n"
)

ABSTAIN_PROMPT = (
    "Below are COMPLETE source documents, then a QUESTION. Determine whether the "
    "documents contain enough information to answer the question with a specific "
    "factual answer. (We EXPECT the answer to be NO — this is an out-of-scope "
    "question that should not be answerable from the corpus.)\n\n"
    "Respond in EXACTLY this format:\n"
    "ANSWERABLE: yes|no\n"
    "NOTE: <brief reason>\n\n"
    "===== SOURCE DOCUMENTS =====\n{corpus}\n\n"
    "===== QUESTION =====\n{question}\n"
)

def judge(prompt):
    resp = w.serving_endpoints.query(
        name=JUDGE_MODEL,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

problems = []
print("Auditing each fact against the full corpus...\n")
for r in rows:
    q = r["request"]
    cat = r["category"]
    label = q[:55]

    if cat == "out_of_scope":
        out = judge(ABSTAIN_PROMPT.format(corpus=CORPUS, question=q))
        answerable = bool(re.search(r"ANSWERABLE:\s*yes", out, re.I))
        if answerable:
            problems.append((label, "OUT_OF_SCOPE but corpus MAY answer it", out[:200]))
            print(f"[  WARN ]  {label}  (corpus may answer an out-of-scope Q)")
        else:
            print(f"[   OK  ]  {label}  (correctly unanswerable)")
        continue

    # answerable: check every required fact (and flag optional facts too)
    bad_required = []
    for fact in list(r["required_facts"]):
        out = judge(FACT_PROMPT.format(corpus=CORPUS, question=q, fact=fact))
        if not re.search(r"VERDICT:\s*SUPPORTED", out, re.I):
            bad_required.append((fact, out))
    bad_optional = []
    for fact in list(r["optional_facts"]):
        out = judge(FACT_PROMPT.format(corpus=CORPUS, question=q, fact=fact))
        if not re.search(r"VERDICT:\s*SUPPORTED", out, re.I):
            bad_optional.append((fact, out))

    if bad_required:
        problems.append((label, "UNSUPPORTED required fact(s)", bad_required))
        print(f"[ FAIL  ]  {label}  ({len(bad_required)} required fact(s) unsupported)")
    elif bad_optional:
        print(f"[ ok*   ]  {label}  (required ok; {len(bad_optional)} optional unsupported)")
    else:
        print(f"[   OK  ]  {label}")

print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)
if not problems:
    print(" All REQUIRED facts supported; out-of-scope questions correctly unanswerable.")
else:
    print(f" {len(problems)} issue(s) needing review:\n")
    for label, kind, detail in problems:
        print(f"--- {label} :: {kind} ---")
        if isinstance(detail, list):
            for fact, out in detail:
                print(f"   FACT: {fact}")
                print(f"   JUDGE: {out[:160]}")
        else:
            print(f"   {detail}")
        print()
print("=" * 70)