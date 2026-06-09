import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from vs_client import get_clients
from databricks.vector_search.reranker import DatabricksReranker
import mlflow

workspace_client, vector_search_client = get_clients()
index = vector_search_client.get_index(config.VS_ENDPOINT, config.INDEX_NAME)


# =====================================================================
# QUERY-TRANSFORMATION HELPERS (used by some retrieve_* variants below)
# =====================================================================

def decompose(question):
    """Split a question into focused sub-questions. Helps multi-hop questions
    (facts in different places); simple questions return unchanged."""
    prompt = (
        "Break the following question into 1-3 focused sub-questions that together "
        "would retrieve everything needed to answer it. If it is already simple, "
        "return it unchanged as a single line. Return ONLY the sub-questions, one "
        "per line, no numbering.\n\n"
        f"Question: {question}"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=200,
    )
    subs = [l.strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    return subs or [question]


def hyde_answer(question):
    """Generate a hypothetical answer to use as the search probe. Need not be
    factually correct — only answer-shaped, so its embedding lands near real
    answer chunks. (Hurts precise-fact retrieval: invented specifics mislead.)"""
    prompt = (
        "Write a short, plausible paragraph that directly answers the question "
        "as if you were a reference document on the German energy market. "
        "Be specific and declarative. Do not hedge or say you are unsure.\n\n"
        f"Question: {question}"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=200,
    )
    return resp.choices[0].message.content


def rewrite_query(question):
    """Rephrase the question into a retrieval-optimized form. (Helps messy/terse
    queries; can distort already-clean ones.)"""
    prompt = (
        "Rewrite the following question into a clear, keyword-rich search query "
        "that would retrieve relevant passages. Keep it concise and factual. "
        "Return only the rewritten query.\n\n"
        f"Question: {question}"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def route_to_doc(question):
    """Classify which document a question targets (metadata routing). (Dangerous
    as a hard filter: wrong routing kills recall; can't serve multi-doc questions.)"""
    prompt = (
        "Which ONE document best answers this question? Reply with exactly one id:\n"
        "energiewende | electricity_sector_in_germany | federal_network_agency | renewable_energy_sources_act\n\n"
        f"Question: {question}\nDocument id:"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=30,
    )
    return resp.choices[0].message.content.strip().split()[0]


# =====================================================================
# RETRIEVAL STRATEGIES — measured results on the 19-question eval set
# (point INDEX_NAME via env var at the index each one expects)
# =====================================================================

def retrieve_plain(question, k=4):
    """Plain top-k ANN. No rerank, no transform.   RESULT: 5/19 (baseline floor).
    Index: any (chunks_index or chunks_structured_index)."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=k,
        query_type="ANN",
        disable_notice=True,
    )
    return results["result"]["data_array"]


def retrieve_reranking(question, k=4, candidates=20):
    """Over-retrieve wide, rerank, keep best k.   RESULT: 10/19 (BEST, co-leader).
    Index: chunks_structured_index. This is the default winner."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="ANN",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


def retrieve_hybrid(question, k=4, candidates=20):
    """Semantic + keyword fusion + rerank.   RESULT: hurt (lexical half = noise
    on a vocabulary-homogeneous corpus). Index: chunks_structured_index."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="HYBRID",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


def retrieve_decomposition(question, k=4, per_sub=3):
    """Sub-questions, plain ANN each, pool + dedupe.   RESULT: 7/19 (helps multi-hop).
    Index: chunks_structured_index."""
    seen = {}
    for sub in decompose(question):
        results = index.similarity_search(
            query_text=sub,
            columns=["chunk_id", "title", "text"],
            num_results=per_sub,
            query_type="ANN",
            disable_notice=True,
        )
        for row in results["result"]["data_array"]:
            if row[0] not in seen:
                seen[row[0]] = row
    return list(seen.values())[:k]


def retrieve_decomp_rerank(question, k=4, candidates=20):
    """Decompose, wide-retrieve+rerank each sub, pool by best score.
    RESULT: ~9-10/19 (decomposition adds little on top of reranking).
    Index: chunks_structured_index."""
    seen = {}
    for sub in decompose(question):
        results = index.similarity_search(
            query_text=sub,
            columns=["chunk_id", "title", "text"],
            num_results=candidates,
            query_type="ANN",
            reranker=DatabricksReranker(columns_to_rerank=["text"]),
            disable_notice=True,
        )
        for row in results["result"]["data_array"]:
            cid = row[0]
            if cid not in seen or row[-1] > seen[cid][-1]:
                seen[cid] = row
    pooled = sorted(seen.values(), key=lambda r: r[-1], reverse=True)
    return pooled[:k]


def retrieve_hyde(question, k=4, candidates=20):
    """Search with a hypothetical answer, then rerank.   RESULT: 8/19 (hurt:
    hallucinated specifics mislead fact-retrieval). Index: chunks_structured_index."""
    probe = hyde_answer(question)
    results = index.similarity_search(
        query_text=probe,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="ANN",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


def retrieve_rewrite(question, k=4, candidates=20):
    """LLM rewrites the query, then rerank.   RESULT: ~5/19 (distorts already-clean
    questions). Index: chunks_structured_index."""
    q = rewrite_query(question)
    results = index.similarity_search(
        query_text=q,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="ANN",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


def retrieve_metadata(question, k=4, candidates=20):
    """Route to one doc, hard-filter, then rerank.   RESULT: 5/19 + breakage
    (wrong routing kills recall; can't serve multi-doc Qs). Index: chunks_structured_index."""
    doc = route_to_doc(question)
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="ANN",
        filters={"doc_id": doc},
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


def retrieve_small_to_big(question, k=4, candidates=20):
    """Retrieve+rerank on small child chunks, return deduped PARENT sections.
    RESULT: 10/19 (co-leader; rich context, fixes different Qs than reranking).
    Index: chunks_small_to_big_index (NOTE: different index!)."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "parent_id", "title", "text", "parent_text"],
        num_results=candidates,
        query_type="ANN",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    rows = results["result"]["data_array"]
    seen = {}
    for row in rows:
        parent_id = row[1]
        if parent_id not in seen:
            seen[parent_id] = [parent_id, row[2], row[4], row[-1]]  # [id, title, parent_text, score]
    return list(seen.values())[:k]


# =====================================================================
# ACTIVE STRATEGY — swap this one line to test a different retriever.
# Make sure INDEX_NAME (env var) matches the index the strategy expects.
# =====================================================================
@mlflow.trace(name="retrieve", span_type="RETRIEVER")
def retrieve(question, **kwargs):
    return retrieve_reranking(question, **kwargs)
    # return retrieve_plain(question, **kwargs)
    # return retrieve_hybrid(question, **kwargs)
    # return retrieve_decomposition(question, **kwargs)
    # return retrieve_decomp_rerank(question, **kwargs)
    # return retrieve_hyde(question, **kwargs)
    # return retrieve_rewrite(question, **kwargs)
    # return retrieve_metadata(question, **kwargs)
    #    # needs chunks_small_to_big_index

@mlflow.trace(name="generate", span_type="LLM")
def generate(question, chunks):
    """Generate an answer using only the retrieved context."""
    context = "\n\n".join(
        f"[{chunk_id}] (from: {title})\n{text}"
        for chunk_id, title, text, _ in chunks
    )
    prompt = (
        "You are an expert on the German energy market. Answer the question using "
        "only the context below. Cite the chunk ids in square brackets when you use them. "
        "If the answer is not in the context, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=500,
    )
    return response.choices[0].message.content


def ask(question):
    """Run the full retrieve-and-generate flow for one question."""
    chunks = retrieve(question)
    answer = generate(question, chunks)
    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    print("Sources retrieved:", ", ".join(str(c[0]) for c in chunks))


if __name__ == "__main__":
    ask("Who regulates the electricity grid in Germany?")