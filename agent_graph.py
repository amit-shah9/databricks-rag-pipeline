from typing import TypedDict, List, Any
import config
from rag import retrieve, generate
from agent import grade_chunks, citation_gate
from langgraph.graph import StateGraph, END


# ---- 1. STATE: the object that flows through every node ----
class AgentState(TypedDict):
    question: str
    query: str                 # current search query (starts = question, updated on loop)
    chunks: List[Any]          # accumulated chunks across passes
    passes: int                # how many retrieval passes so far
    max_passes: int
    answer: str
    grounded: bool
    sufficient: bool


# ---- 2. NODES: each takes state, returns state updates ----
def retrieve_node(state: AgentState) -> dict:
    """Retrieve for the current query, accumulate (dedupe) into chunks."""
    new = retrieve(state["query"])
    seen = {c[0]: c for c in state["chunks"]}     # existing, keyed by chunk_id
    for c in new:
        seen[c[0]] = c
    return {"chunks": list(seen.values()), "passes": state["passes"] + 1}


def grade_node(state: AgentState) -> dict:
    """Grade sufficiency; if insufficient, set query to the 'missing' gap."""
    sufficient, missing = grade_chunks(state["question"], state["chunks"])
    return {"sufficient": sufficient,
            "query": missing if not sufficient else state["query"]}


def generate_node(state: AgentState) -> dict:
    """Generate the answer from accumulated chunks."""
    return {"answer": generate(state["question"], state["chunks"])}


def gate_node(state: AgentState) -> dict:
    """Citation-gate the answer."""
    grounded, _ = citation_gate(state["question"], state["answer"], state["chunks"])
    return {"grounded": grounded}


# ---- 3. ROUTERS: read state, return the name of the next node ----
def route_after_grade(state: AgentState) -> str:
    """If chunks insufficient and we haven't hit the cap, loop back to retrieve.
    Otherwise proceed to generate."""
    if not state["sufficient"] and state["passes"] < state["max_passes"]:
        return "retrieve"
    return "generate"


def route_after_gate(state: AgentState) -> str:
    """After gating, end. (grounded vs not is handled in the final output.)"""
    return "END"


# ---- 4. ASSEMBLE THE GRAPH ----
workflow = StateGraph(AgentState)

# add each node: (name, function)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_node("gate", gate_node)

# entry point: where execution starts
workflow.set_entry_point("retrieve")

# normal edge: retrieve always -> grade
workflow.add_edge("retrieve", "grade")

# conditional edge: after grade, route via route_after_grade.
# The mapping translates the router's returned string -> actual node name.
workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {"retrieve": "retrieve", "generate": "generate"},
)

# normal edge: generate -> gate
workflow.add_edge("generate", "gate")

# conditional edge after gate (ends either way for now)
workflow.add_conditional_edges(
    "gate",
    route_after_gate,
    {"END": END},
)

# compile into a runnable
app = workflow.compile()


# ---- 5. RUNNER ----
def graph_ask(question, max_passes=3):
    """Run the compiled graph for one question."""
    initial = {
        "question": question,
        "query": question,        # first search uses the question itself
        "chunks": [],
        "passes": 0,
        "max_passes": max_passes,
        "answer": "",
        "grounded": False,
        "sufficient": False,
    }
    final = app.invoke(initial)
    return final


if __name__ == "__main__":
    result = graph_ask("When and why did Germany phase out nuclear power, and was it controversial?")
    print("passes:", result["passes"])
    print("chunks:", len(result["chunks"]))
    print("grounded:", result["grounded"])
    print("answer:", result["answer"][:300])