import config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from rag import retrieve, generate, decompose, rewrite_query, workspace_client
from databricks.connect import DatabricksSession
_spark = DatabricksSession.builder.clusterId(config.CLUSTER_ID).getOrCreate()
from openai import OpenAI

def web_search(query, max_results=3):
    """Tool: search the live web for current information not in the corpus."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS   # older package name
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"{r.get('title','')}: {r.get('body','')}" for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"

WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for CURRENT information not in the document corpus "
            "and not in the energy-metrics table — e.g. recent news, events after the "
            "documents were written, or general facts outside German energy policy. "
            "Use only when neither the documents nor the energy data can answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the web search query"},
            },
            "required": ["query"],
        },
    },
}

def search_documents(query):
    """Tool: retrieve relevant passages from the German energy market document corpus."""
    chunks = retrieve(query)   # your active reranking retrieval
    return "\n\n".join(f"[{c[0]}] {c[2]}" for c in chunks)

DOC_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the German energy market document corpus (Energiewende, "
            "electricity sector, Federal Network Agency, Renewable Energy Sources "
            "Act). Use for any question about energy policy, history, regulation, "
            "the EEG, the Energiewende, or how the German energy system works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the search query"},
            },
            "required": ["query"],
        },
    },
}

def query_energy_data(metric, date=None):
    """Tool: query the live_energy_metrics Delta table for a metric (optionally by date)."""
    table = f"{config.FQ_SCHEMA}.live_energy_metrics"
    q = f"SELECT date, metric, value FROM {table} WHERE metric = '{metric}'"
    if date:
        q += f" AND date = '{date}'"
    rows = _spark.sql(q).collect()
    if not rows:
        return f"No data found for metric '{metric}'."
    return "; ".join(f"{r['date']}: {r['metric']}={r['value']}" for r in rows)

# Tool schema the LLM sees (function-calling description)
SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "query_energy_data",
        "description": (
            "Query current/live German energy metrics NOT in the document corpus: "
            "day_ahead_price_eur_mwh, wind_generation_gw, solar_generation_gw, "
            "co2_intensity_g_kwh. Use this for questions about current prices or "
            "live generation figures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string",
                           "description": "one of: day_ahead_price_eur_mwh, wind_generation_gw, solar_generation_gw, co2_intensity_g_kwh"},
                "date": {"type": "string", "description": "optional date YYYY-MM-DD"},
            },
            "required": ["metric"],
        },
    },
}

TOOL_FUNCS = {
    "query_energy_data": query_energy_data,
    "search_documents": search_documents,
    "web_search": web_search,
}

ALL_TOOLS = [DOC_TOOL, SQL_TOOL, WEB_TOOL]

GROUNDING_SYSTEM = (
    "You are an assistant on the German energy market. You must answer ONLY using "
    "information returned by your tools. Use search_documents for questions about "
    "energy policy/history/regulation, query_energy_data for current prices or live "
    "generation figures, and web_search for current information outside the corpus "
    "(recent news, facts the other tools can't provide). Do NOT answer from your own "
    "knowledge. If the tools do not return enough information, say so."
)

def grounded_agent_ask(question, max_tool_calls=4, max_gate_retries=2, verbose=False):
    """Unified grounded agent with citation-gate loop-back: if the gate fails,
    retrieve more and re-answer (up to max_gate_retries) before refusing."""
    import json

    def run_tools_and_answer(user_msg, extra_context_hint=None):
        """One pass: let the LLM call tools and produce an answer. Returns
        (answer, tool_outputs)."""
        sys = GROUNDING_SYSTEM
        if extra_context_hint:
            sys += ("\n\nThe previous answer had unsupported claims: "
                    f"{extra_context_hint}. Use your tools to find support for "
                    "those specific claims, or omit them.")
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_msg},
        ]
        outs = []
        for _ in range(max_tool_calls + 1):
            resp = _openai_client.chat.completions.create(
                model=config.CHAT_ENDPOINT, messages=messages,
                tools=ALL_TOOLS, max_tokens=600,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content, outs
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = TOOL_FUNCS[tc.function.name](**args)
                outs.append(str(result))
                if verbose:
                    print(f"  [tool] {tc.function.name}({args}) -> {str(result)[:70]}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)})
        return "Stopped after max tool calls.", outs

    # initial pass
    answer, tool_outputs = run_tools_and_answer(question)

    # citation-gate loop-back
    for attempt in range(max_gate_retries + 1):
        if not tool_outputs:
            break
        grounded, detail = citation_gate(
            question, answer, [("tool", "", "\n\n".join(tool_outputs), 0)])
        if verbose:
            print(f"  [gate attempt {attempt}] grounded={grounded}")
        if grounded:
            return answer
        if attempt == max_gate_retries:
            break   # out of retries -> refuse below
        # loop back: extract what was unsupported, retrieve to fill it, re-answer
        unsupported = detail.split("UNSUPPORTED:", 1)[-1].strip()
        if verbose:
            print(f"  [gate loop-back] retrying to ground: {unsupported[:70]!r}")
        new_answer, new_outs = run_tools_and_answer(question, extra_context_hint=unsupported)
        answer = new_answer
        tool_outputs += new_outs   # accumulate context across attempts

    return ("I don't have enough supported information from my tools to answer "
            "that fully and accurately.")

# OpenAI-compatible client pointed at Databricks serving endpoints.
# Databricks exposes an OpenAI-compatible API at /serving-endpoints.
_openai_client = OpenAI(
    api_key=workspace_client.config.oauth_token().access_token,
    base_url=f"{workspace_client.config.host}/serving-endpoints",
)

def tool_calling_ask(question, max_tool_calls=3):
    """Agent with function calling via the OpenAI-compatible Databricks API.
    The LLM decides whether to call the SQL tool."""
    import json
    messages = [{"role": "user", "content": question}]
    for _ in range(max_tool_calls + 1):
        resp = _openai_client.chat.completions.create(
            model=config.CHAT_ENDPOINT,
            messages=messages,
            tools=[SQL_TOOL],
            max_tokens=500,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content                       # answered, no tool needed
        # record the assistant's tool-call request
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        # execute each tool call, feed results back
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_FUNCS[tc.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
    return "Stopped after max tool calls."

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
    """Check that every claim in the answer is supported by the context.
    Tolerant of formatting/representation differences (units, rounding, field
    names) — judge whether the FACTS match, not the exact wording."""
    context = "\n\n".join(f"[{c[0]}] {c[2]}" for c in chunks)
    prompt = (
        "Check whether the factual claims in the ANSWER are supported by the "
        "CONTEXT. Treat differently-formatted representations of the same fact as "
        "supported: e.g. 'day_ahead_price_eur_mwh=79.1' supports '€79.10 per MWh'; "
        "ignore units, rounding, field-name vs prose differences. Only fail if the "
        "answer asserts something the context genuinely does not contain.\n\n"
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