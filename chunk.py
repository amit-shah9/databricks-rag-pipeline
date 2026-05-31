from databricks.connect import DatabricksSession

import config


spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()


# I use word-based chunks here because the source documents are plain text.
# The overlap helps keep some context between neighboring chunks for retrieval.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a document into overlapping word chunks."""

    words = text.split()
    step = size - overlap
    chunks = []

    for start in range(0, len(words), step):
        piece = words[start : start + size]

        if piece:
            chunks.append(" ".join(piece))

        if start + size >= len(words):
            break

    return chunks


documents = spark.sql(
    f"""
    SELECT
        doc_id,
        title,
        source,
        text
    FROM {config.DOCUMENTS_TABLE}
    """
).collect()


rows = []

for document in documents:
    pieces = chunk_text(document["text"])

    for index, piece in enumerate(pieces):
        rows.append(
            (
                f"{document['doc_id']}__{index:03d}",
                document["doc_id"],
                document["title"],
                document["source"],
                index,
                piece,
            )
        )

    print(f"{document['doc_id']}: {len(pieces)} chunks")


chunks_df = spark.createDataFrame(
    rows,
    ["chunk_id", "doc_id", "title", "source", "chunk_index", "text"],
)

chunks_df.write.mode("overwrite").saveAsTable(config.CHUNKS_TABLE)


# Delta Sync in Vector Search relies on Change Data Feed, so I enable it here
# right after creating the chunks table.
spark.sql(
    f"""
    ALTER TABLE {config.CHUNKS_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)


print(f"\nWrote {chunks_df.count()} chunks to {config.CHUNKS_TABLE}")

spark.sql(
    f"""
    SELECT
        doc_id,
        COUNT(*) AS n_chunks,
        AVG(length(text)) AS avg_chars
    FROM {config.CHUNKS_TABLE}
    GROUP BY doc_id
    """
).show(truncate=False)

