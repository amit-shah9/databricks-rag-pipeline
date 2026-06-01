import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from vs_client import get_clients
from databricks.vector_search.reranker import DatabricksReranker


workspace_client, vector_search_client = get_clients()
index = vector_search_client.get_index(config.VS_ENDPOINT, config.INDEX_NAME)


def retrieve(question, k=3):
    """Retrieve the top-k most similar chunks for a question."""
    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=k,
        query_type="ANN",
        # Reranker disabled for now — it underperformed on oversized chunks
        # (only reads first ~2000 chars; our chunks were ~3000). Re-enable
        # after re-chunking smaller. See chunk.py.
        # reranker=DatabricksReranker(columns_to_rerank=["text"]),
    )
    return results["result"]["data_array"]

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