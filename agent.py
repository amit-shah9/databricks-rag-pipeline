import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from rag import retrieve, generate, decompose, rewrite_query, workspace_client


def grade_chunks(question, chunks):
    """Grade whether the context contains the SPECIFIC facts for a complete answer.
    Returns (sufficient: bool, missing: str). 'missing' is phrased as a search
    query for the next retrieval pass, so the agent can target the gap."""
    context = "\n\n".join(f"[{c[0]}] {c[2]}" for c in chunks)
    prompt = (
        "You are checking whether the retrieved context contains the specific "
        "facts needed to answer the question COMPLETELY (names, dates, figures, "
        "and every item the question implies). Apply a high bar.\n\n"
        "Respond in EXACTLY this format:\n"
        "SUFFICIENT: yes|no\n"
        "MISSING: <if no, write a short search query that would retrieve the "
        "missing specific facts; if yes, write 'none'>\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=200,
    )
    text = resp.choices[0].message.content
    sufficient = "sufficient: yes" in text.lower()
    # extract the MISSING line as the next-pass query
    missing = ""
    for line in text.splitlines():
        if line.lower().startswith("missing:"):
            missing = line.split(":", 1)[1].strip()
    return sufficient, missing


def citation_gate(question, answer, chunks):
    """Check that every claim in the answer is supported by the chunks.
    Returns (grounded: bool, problems: str). This is what keeps the agent honest:
    it must not assert beyond what the retrieved context supports."""
    context = "\n\n".join(f"[{c[0]}] {c[2]}" for c in chunks)
    prompt = (
        "Check whether EVERY factual claim in the ANSWER is supported by the "
        "CONTEXT below. If any claim is not supported by the context, the answer "
        "fails.\n\n"
        "Respond in EXACTLY this format:\n"
        "GROUNDED: yes|no\n"
        "UNSUPPORTED: <unsupported claims, or 'none'>\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}"
    )
    resp = workspace_client.serving_endpoints.query(
        name=config.CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=300,
    )
    text = resp.choices[0].message.content
    grounded = "grounded: yes" in text.lower()
    return grounded, text.strip()


def agentic_ask(question, max_passes=3, k=4, verbose=True):
    """Agentic RAG: retrieve, grade for completeness, re-retrieve to fill gaps
    (accumulating chunks), then generate and citation-gate. Bounded by max_passes."""
    accumulated = {}        # chunk_id -> row, deduped across passes
    query = question
    for pass_num in range(1, max_passes + 1):
        new_chunks = retrieve(query, k=k)
        for c in new_chunks:
            accumulated[c[0]] = c
        chunks = list(accumulated.values())
        if verbose:
            print(f"  [pass {pass_num}] query={query[:60]!r} -> pool now {len(chunks)} chunks")
        sufficient, missing = grade_chunks(question, chunks)
        if sufficient:
            if verbose:
                print(f"  [pass {pass_num}] grader: SUFFICIENT")
            break
        if verbose:
            print(f"  [pass {pass_num}] grader: insufficient, missing -> {missing[:60]!r}")
        if not missing:
            break
        query = missing      # next pass targets the gap

    answer = generate(question, list(accumulated.values()))
    grounded, gate_detail = citation_gate(question, answer, list(accumulated.values()))
    if verbose:
        print(f"  [gate] grounded={grounded}")
    if not grounded:
        # honesty: refuse rather than emit unsupported content
        answer = ("I don't have enough supported information in the retrieved "
                  "documents to answer this fully and accurately.")
    return answer, list(accumulated.values()), grounded