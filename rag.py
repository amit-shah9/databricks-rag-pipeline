import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from vs_client import get_clients


workspace_client, vector_search_client = get_clients()
index = vector_search_client.get_index(config.VS_ENDPOINT, config.INDEX_NAME)


def retrieve(question, k=3):
    """Find the most relevant chunks for a user question."""

    results = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "title", "text"],
        num_results=k,
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