import re
from databricks.connect import DatabricksSession
import config

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

SKIP_SECTIONS = {"references", "see also", "external links", "further reading",
                 "notes", "bibliography", "sources"}
CHILD_WORDS = 150      # small = precise retrieval
CHILD_OVERLAP = 30

def split_into_sections(text):
    parts = re.split(r'\n(=+\s*[^=\n]+\s*=+)\n', text)
    sections = []
    intro = parts[0].strip()
    if intro:
        sections.append(("Introduction", intro))
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip("= ").strip()
        body = parts[i + 1].strip()
        sections.append((heading, body))
    return sections

def child_split(text, size=CHILD_WORDS, overlap=CHILD_OVERLAP):
    words = text.split()
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out

docs = spark.sql(
    f"SELECT doc_id, title, source, text FROM {config.DOCUMENTS_TABLE}"
).collect()

rows = []
for doc in docs:
    parent_n = 0
    for heading, body in split_into_sections(doc["text"]):
        if heading.lower().strip() in SKIP_SECTIONS:
            continue
        if len(body.split()) < 20:
            continue
        # the PARENT is the full section (heading + body)
        parent_id = f"{doc['doc_id']}__p{parent_n:03d}"
        parent_text = f"{heading}\n{body}"
        # split the section into small CHILD chunks
        children = child_split(body)
        for ci, child in enumerate(children):
            child_id = f"{parent_id}__c{ci:03d}"
            # child text we embed; parent_text carried along for lookup
            rows.append((
                child_id, parent_id, doc["doc_id"], doc["title"],
                doc["source"], heading,
                f"{heading}\n{child}",   # child text (what we search)
                parent_text,             # parent text (what we return to LLM)
            ))
        parent_n += 1
    print(f"{doc['doc_id']}: {parent_n} parents")

df = spark.createDataFrame(rows, [
    "chunk_id", "parent_id", "doc_id", "title", "source",
    "section", "text", "parent_text",
])
TABLE = f"{config.FQ_SCHEMA}.chunks_small_to_big"
df.write.mode("overwrite").saveAsTable(TABLE)
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"\nWrote {df.count()} child chunks to {TABLE}")
spark.sql(f"""
    SELECT COUNT(*) AS children, COUNT(DISTINCT parent_id) AS parents,
           AVG(length(text)) AS avg_child_chars, AVG(length(parent_text)) AS avg_parent_chars
    FROM {TABLE}
""").show(truncate=False)