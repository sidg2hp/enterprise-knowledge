from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

# State schema
class GraphState(TypedDict):
    query: str
    domain: str
    context: str
    response: str
    citations: List[str]

# Nodes
def query_analyzer(state: GraphState):
    # Mock analyzer that sets domain based on keyword
    query = state.get("query", "").lower()
    domain = "GENERAL"
    if "resume" in query or "hire" in query or "hr" in query:
        domain = "HR"
    elif "policy" in query or "audit" in query:
        domain = "POLICY"
    elif "precedent" in query or "history" in query:
        domain = "PRECEDENT"
    return {"domain": domain}

def hr_agent(state: GraphState):
    return {"context": "Mock HR Context: Candidate has Python and LangGraph experience [DocID: HR-101]."}

def policy_agent(state: GraphState):
    return {"context": "Mock Policy Context: Remote work policy allows 3 days at home [DocID: POL-202]."}

def precedent_agent(state: GraphState):
    return {"context": "Mock Precedent Context: Project X succeeded using Neo4j in 2024 [DocID: PREC-303]."}

def general_agent(state: GraphState):
    return {"context": "Mock General Context: Enterprise knowledge platform [DocID: GEN-000]."}

def hybrid_retriever(state: GraphState):
    # Mock retrieval logic (combining Neo4j, FAISS, BM25)
    return {"context": state.get("context", "") + "\nAdditional Retrieval Data."}

def synthesizer(state: GraphState):
    context = state.get("context", "")
    response = f"Based on the knowledge base: {context}"
    # Mock citation extraction
    citations = []
    if "HR-101" in context: citations.append("HR-101")
    if "POL-202" in context: citations.append("POL-202")
    if "PREC-303" in context: citations.append("PREC-303")
    if "GEN-000" in context: citations.append("GEN-000")
    
    return {"response": response, "citations": citations}

def evaluator(state: GraphState):
    # Mock evaluation logic checking if citations exist
    citations = state.get("citations", [])
    if not citations:
        return {"response": state.get("response", "") + "\nWarning: Low groundedness score, missing citations."}
    return {"response": state.get("response", "") + f"\n[Grounded with citations: {citations}]"}

# Router
def route_domain(state: GraphState):
    domain = state.get("domain", "GENERAL")
    if domain == "HR": return "hr_agent"
    if domain == "POLICY": return "policy_agent"
    if domain == "PRECEDENT": return "precedent_agent"
    return "general_agent"

# Graph Construction
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("query_analyzer", query_analyzer)
workflow.add_node("hr_agent", hr_agent)
workflow.add_node("policy_agent", policy_agent)
workflow.add_node("precedent_agent", precedent_agent)
workflow.add_node("general_agent", general_agent)
workflow.add_node("hybrid_retriever", hybrid_retriever)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("evaluator", evaluator)

# Add edges
workflow.set_entry_point("query_analyzer")

workflow.add_conditional_edges(
    "query_analyzer",
    route_domain,
    {
        "hr_agent": "hr_agent",
        "policy_agent": "policy_agent",
        "precedent_agent": "precedent_agent",
        "general_agent": "general_agent"
    }
)

workflow.add_edge("hr_agent", "hybrid_retriever")
workflow.add_edge("policy_agent", "hybrid_retriever")
workflow.add_edge("precedent_agent", "hybrid_retriever")
workflow.add_edge("general_agent", "hybrid_retriever")

workflow.add_edge("hybrid_retriever", "synthesizer")
workflow.add_edge("synthesizer", "evaluator")
workflow.add_edge("evaluator", END)

# Compile graph
app_graph = workflow.compile()
