from typing import Dict, Any, List
import networkx as nx

from .state import AgentState
from ..graph.ingestion import load_resilience_graph
from ..graph.retriever import retrieve_subgraph_context
from ..llm.engine import LLMEngine
from ..llm import prompts
from ..schemas.models import ExtractionOutput, AssessmentOutput

# Initialize LLM Engine singleton and load graph into memory once
llm_engine: LLMEngine = LLMEngine()
resilience_graph: nx.DiGraph = load_resilience_graph()


def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's input message to extract active resilience nodes 
    using the Gemma LLM with structured output enforcement.

    Args:
        state (AgentState): Current state containing 'user_message'.

    Returns:
        Dict[str, Any]: Updated state dict with 'active_nodes'.
    """
    print("[Node] Extractor Agent is analyzing input...")
    user_msg: str = state.get("user_message", "")
    
    extractor_chain = llm_engine.get_extractor_runner(prompts.EXTRACTOR_SYSTEM_PROMPT)
    result: ExtractionOutput = extractor_chain.invoke({"user_message": user_msg})
    
    # Extract unique node IDs from active signals
    active_node_ids: List[str] = list({signal.node_id for signal in result.active_signals})
    
    return {"active_nodes": active_node_ids}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves formatted sub-graph context for active nodes from NetworkX.

    Args:
        state (AgentState): Current state containing 'active_nodes'.

    Returns:
        Dict[str, Any]: Updated state dict with 'subgraph_context'.
    """
    print("[Node] Graph Retriever is fetching node knowledge...")
    active_nodes: List[str] = state.get("active_nodes", [])
    
    # Fetch structured string representation from NetworkX graph
    context: str = retrieve_subgraph_context(resilience_graph, active_nodes)
    
    return {"subgraph_context": context}


def assessor_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates resilience levels and calculates confidence to determine 
    if disambiguation is needed.

    Args:
        state (AgentState): Current state with 'user_message' and 'subgraph_context'.

    Returns:
        Dict[str, Any]: Updated state dict with 'assessments' and 'requires_disambiguation'.
    """
    print("[Node] Assessor Agent is evaluating resilience status...")
    user_msg: str = state.get("user_message", "")
    context: str = state.get("subgraph_context", "")
    
    assessor_chain = llm_engine.get_assessor_runner(prompts.ASSESSOR_SYSTEM_PROMPT)
    result: AssessmentOutput = assessor_chain.invoke({
        "user_message": user_msg,
        "subgraph_context": context
    })
    
    # Convert Pydantic models to dicts for LangGraph state compatibility
    assessments_list: List[Dict[str, Any]] = [
        assessment.model_dump() for assessment in result.assessments
    ]
    
    return {
        "assessments": assessments_list,
        "requires_disambiguation": result.requires_disambiguation
    }


def questioner_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a targeted, empathetic clarification question when input is ambiguous.

    Args:
        state (AgentState): Current state with 'user_message' and 'subgraph_context'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response'.
    """
    print("[Node] Questioner Agent is formulating clarification...")
    user_msg: str = state.get("user_message", "")
    context: str = state.get("subgraph_context", "")
    
    conversational_llm = llm_engine.get_conversational_llm()
    prompt_template = prompts.get_questioner_prompt()
    
    chain = prompt_template | conversational_llm
    response = chain.invoke({
        "user_message": user_msg,
        "subgraph_context": context
    })
    
    return {"final_response": response.content}


def advisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates tailored psychological advice and interventions using active graph guidance.

    Args:
        state (AgentState): Current state with 'user_message', 'subgraph_context', and 'assessments'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response'.
    """
    print("[Node] Advisor Agent is generating psychological interventions...")
    user_msg: str = state.get("user_message", "")
    context: str = state.get("subgraph_context", "")
    assessments: List[Dict[str, Any]] = state.get("assessments", [])
    
    conversational_llm = llm_engine.get_conversational_llm()
    prompt_template = prompts.get_advisor_prompt()
    
    chain = prompt_template | conversational_llm
    response = chain.invoke({
        "user_message": user_msg,
        "subgraph_context": context,
        "assessments": str(assessments)
    })
    
    return {"final_response": response.content}
