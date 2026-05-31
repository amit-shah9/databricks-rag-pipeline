# RAG Pipeline on Databricks — German Energy Market

This project is a small retrieval-augmented generation pipeline built on Databricks Mosaic AI. It answers questions about the German electricity market by retrieving relevant text chunks from public documents and passing them to an LLM to generate a grounded answer with source references.

I built this as the first baseline version of a larger RAG project. The goal of this phase was to keep the pipeline simple and understandable: load documents, split them into chunks, create embeddings, retrieve the most relevant chunks, and generate an answer from that context.

## Architecture

```text
Documents Delta table
      │
      │  split into overlapping text chunks
      ▼
Chunks Delta table
      │
      │  Delta Sync creates managed embeddings
      ▼
Mosaic AI Vector Search index
      │
      │  similarity search returns top-k chunks
      ▼
Chat model
      │
      │  answers using only the retrieved context
      ▼
Grounded answer with chunk citations
```

The pipeline has three main model/service parts:

* an embedding model that turns text chunks into vectors
* a Vector Search index that stores and searches those vectors
* a chat model that writes the final answer using the retrieved context

## Stack

* **Databricks Mosaic AI Vector Search** for the managed vector index
* **Databricks Foundation Model APIs** for embeddings and answer generation
* **Unity Catalog** for storing the documents and chunks tables
* **Databricks Connect** for running the project locally against a Databricks cluster
* **Python** and **python-dotenv** for the pipeline code and configuration

## Files

| File             | What it does                                                           |
| ---------------- | ---------------------------------------------------------------------- |
| `config.py`      | Loads workspace-specific settings from `.env`                          |
| `ingest.py`      | Fetches the source documents and writes the `documents` Delta table    |
| `chunk.py`       | Splits documents into overlapping chunks and writes the `chunks` table |
| `build_index.py` | Creates the Vector Search endpoint and Delta Sync index                |
| `rag.py`         | Runs the retrieve-and-generate flow for a question                     |
| `vs_client.py`   | Creates shared Databricks and Vector Search clients                    |

## Running the project

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Log in to Databricks:

```bash
databricks auth login
```

Create your local environment file:

```bash
cp .env.example .env
```

Then fill in your Databricks cluster ID, catalog/schema, and endpoint names in `.env`.

Run the pipeline in order:

```bash
python ingest.py        # load source documents
python chunk.py         # split documents into chunks
python build_index.py   # create the Vector Search index
python rag.py           # run a sample question
```

## Corpus

For this first version, I used a small set of public English-language Wikipedia pages about the German energy market:

* Energiewende
* Electricity sector in Germany
* Federal Network Agency
* Renewable Energy Sources Act

I chose this topic because it has enough factual detail to test simple retrieval, but is still small enough to keep the project easy to inspect.

## Example

```text
Q: Who regulates the electricity grid in Germany?

A: The Federal Network Agency regulates the electricity grid in Germany [federal_network_agency__000]. The retrieved context also mentions the four transmission system operators in Germany: 50Hertz, Amprion, TenneT, and TransnetBW [electricity_sector_in_germany__003].
```

## Roadmap

* [x] Phase 1 — Naive RAG: chunk, embed, retrieve, generate
* [ ] Phase 2 — Add an evaluation harness
* [ ] Phase 3 — Try hybrid search, reranking, and query rewriting
* [ ] Phase 4 — Add agentic retrieval loops and self-correction
* [ ] Phase 5 — Explore GraphRAG for entity and relationship retrieval
* [ ] Phase 6 — Add adaptive routing for different question types
* [ ] Phase 7 — Prepare the project for production-style deployment
