import re
from databricks.connect import DatabricksSession
import config

spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()

# Sections we skip entirely — pure noise, no answerable content.
SKIP_SECTIONS = {"references", "see also", "external links", "further reading",
                 "notes", "bibliography", "sources"}

MAX_WORDS = 350      # if a section is longer than this, sub-split it
MIN_WORDS = 20       # drop tiny fragments

def split_into_sections(text):
    """Split Wikipedia plain text into (heading, body) sections on == headings ==."""
    # Split on lines that are == Heading == or === Subheading === etc.
    parts = re.split(r'\n(=+\s*[^=\n]+\s*=+)\n', text)
    # parts[0] is the intro (before any heading); then alternating heading, body.
    sections = []
    intro = parts[0].strip()
    if intro:
        sections.append(("Introduction", intro))
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip("= ").strip()
        body = parts[i + 1].strip()
        sections.append((heading, body))
    return sections

def word_split(text, size=MAX_WORDS, overlap=40):
    """Sub-split an over-long section into word windows with overlap."""
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

documents = spark.sql(
    f"SELECT doc_id, title, source, text FROM {config.DOCUMENTS_TABLE}"
).collect()

rows = []
for doc in documents:
    chunk_n = 0
    for heading, body in split_into_sections(doc["text"]):
        if heading.lower().strip() in SKIP_SECTIONS:
            continue
        # Long sections get sub-split; short ones stay whole.
        pieces = word_split(body) if len(body.split()) > MAX_WORDS else [body]
        for piece in pieces:
            if len(piece.split()) < MIN_WORDS:
                continue
            # Prepend the heading so the chunk carries its own context.
            chunk_text = f"{heading}\n{piece}"
            rows.append((
                f"{doc['doc_id']}__{chunk_n:03d}",
                doc["doc_id"], doc["title"], doc["source"],
                chunk_n, heading, chunk_text,
            ))
            chunk_n += 1
    print(f"{doc['doc_id']}: {chunk_n} chunks")

df = spark.createDataFrame(
    rows, ["chunk_id", "doc_id", "title", "source", "chunk_index", "section", "text"]
)
TABLE = f"{config.FQ_SCHEMA}.chunks_structured"
df.write.mode("overwrite").saveAsTable(TABLE)
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"\nWrote {df.count()} chunks to {TABLE}")
spark.sql(f"""
    SELECT doc_id, COUNT(*) AS n_chunks, AVG(length(text)) AS avg_chars
    FROM {TABLE} GROUP BY doc_id
""").show(truncate=False)