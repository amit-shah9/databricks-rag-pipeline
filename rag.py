import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from vs_client import get_clients
from databricks.vector_search.reranker import DatabricksReranker


workspace_client, vector_search_client = get_clients()
index = vector_search_client.get_index(config.VS_ENDPOINT, config.INDEX_NAME)


def decompose(question):
    """Split a question into focused sub-questions for retrieval.

    Multi-hop questions need facts from different places; one search misses a
    side. Sub-questions let each piece be retrieved on its own. Simple questions
    return unchanged.
    """
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


def retrieve(question, k=4, candidates=20):
    """Reranking only: over-retrieve wide, rerank, keep best k."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=candidates,
        query_type="ANN",
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        disable_notice=True,
    )
    return results["result"]["data_array"][:k]


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
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt,
            )
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content


def ask(question):
    """Run the full retrieve-and-generate flow for one question."""

    chunks = retrieve(question)
    answer = generate(question, chunks)

    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    print("Sources retrieved:", ", ".join(chunk_id for chunk_id, _, _, _ in chunks))


if __name__ == "__main__":
    ask("Who regulates the electricity grid in Germany?")