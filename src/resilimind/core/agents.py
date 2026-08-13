from typing import Dict, Any, List
import networkx as nx
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import AgentState
from .database import get_user_latest_node_statuses, get_user_node_timeline
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
    signals_list: List[Dict[str, Any]] = [
        signal.model_dump() for signal in result.active_signals
    ]
    
    # Extract unique node IDs from active signals
    active_node_ids: List[str] = list({signal.node_id for signal in result.active_signals})
    
    return {"active_nodes": active_node_ids, "active_signals": signals_list}


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
    Evaluates resilience levels using extracted evidence and polarity signals,
    grounded in the retrieved subgraph context.

    Args:
        state (AgentState): Current state containing 'user_message', 
                            'subgraph_context', and 'active_signals'.

    Returns:
        Dict[str, Any]: Updated state with 'assessments' and 'requires_disambiguation'.
    """
    print("[Node] Assessor Agent is evaluating resilience status with evidence...")
    user_msg: str = state.get("user_message", "")
    context: str = state.get("subgraph_context", "")
    active_signals: List[Dict[str, Any]] = state.get("active_signals", [])

    # 1. Format extracted signals & evidence substrings for prompt ingestion
    if active_signals:
        evidence_blocks = []
        for sig in active_signals:
            evidence_blocks.append(
                f"• Target Node: {sig.get('node_id')}\n"
                f"  - Extracted Polarity: {sig.get('detected_signal', 'mixed').upper()}\n"
                f"  - Exact User Substring (Evidence): \"{sig.get('evidence', '')}\""
            )
        formatted_evidence = "\n".join(evidence_blocks)
    else:
        formatted_evidence = "No explicit extracted signals provided."

    # 2. Construct clear evidence-aware payload conforming to assessor.txt prompt
    enriched_input = (
        f"=== EXTRACTED SIGNALS & EVIDENCE ===\n"
        f"{formatted_evidence}\n\n"
        f"=== FULL USER MESSAGE ===\n"
        f"{user_msg}"
    )

    # 3. Invoke LLM chain with evidence payload
    assessor_chain = llm_engine.get_assessor_runner(prompts.ASSESSOR_SYSTEM_PROMPT)
    result: AssessmentOutput = assessor_chain.invoke({
        "user_message": enriched_input,
        "subgraph_context": context
    })
    
    # 4. Convert Pydantic models to dicts for LangGraph state compatibility
    assessments_list: List[Dict[str, Any]] = [
        assessment.model_dump() for assessment in result.assessments
    ]
    
    return {
        "assessments": assessments_list,
        "requires_disambiguation": result.requires_disambiguation
    }


def questioner_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a targeted, empathetic clarification question when input is ambiguous,
    using the full conversational memory.

    Args:
        state (AgentState): Current state with 'messages' and 'subgraph_context'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    print("[Node] Questioner Agent is formulating clarification...")
    context: str = state.get("subgraph_context", "")
    
    # Fetch the full chat history from the graph state
    messages_history = state.get("messages", [])
    
    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_questioner_prompt()
    chain = prompt | conversational_llm    
    response = chain.invoke({
        "user_message": state.get("user_message", ""),
        "subgraph_context": context,
        "messages": messages_history
    })

    question_text: str = response.content
    return {
        "final_response": question_text,
        "messages": [AIMessage(content=question_text)]
    }


def advisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates tailored psychological advice and interventions using active graph guidance
    combined with the user's historical resilience profile and full conversational memory.

    Args:
        state (AgentState): Current state with 'user_id', 'messages', 'subgraph_context', and 'assessments'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    print("[Node] Advisor Agent is generating psychological interventions with full memory...")
    user_id: int = state.get("user_id", 0)
    subgraph_context: str = state.get("subgraph_context", "")
    assessments: List[Dict[str, Any]] = state.get("assessments", [])
    
    # Fetch the full chat history from the graph state
    messages_history = state.get("messages", [])
    
    # Fetch user's historical resilience timeline from SQLite database
    history_logs: List[Dict[str, Any]] = get_user_node_timeline(user_id) if user_id else []
    
    # Format historical profile into a chronological timeline block
    history_context: str = "No prior historical timeline recorded."
    if history_logs:
        timeline_by_node = {}
        for log in history_logs:
            nid = log.get('node_id')
            if nid not in timeline_by_node:
                timeline_by_node[nid] = []
            
            date_str = str(log.get('created_at', ''))[:10]
            status = log.get('status', 'UNKNOWN')
            score = log.get('score', 'N/A')
            timeline_by_node[nid].append(f"[{date_str}] {status}({score})")
        
        formatted_logs = []
        for nid, timeline in timeline_by_node.items():
            path_str = " ➔ ".join(timeline)
            formatted_logs.append(f"• Node {nid} Timeline: {path_str}")
            
        history_context = "\n".join(formatted_logs)
    
    # Combine real-time graph context with the user's historical profile
    full_context: str = (
        f"=== CURRENT GRAPH KNOWLEDGE ===\n{subgraph_context}\n\n"
        f"=== USER HISTORICAL RESILIENCE PROFILE ===\n{history_context}"
    )

    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_advisor_prompt()
    chain = prompt | conversational_llm    
    response = chain.invoke({
        "user_message": state.get("user_message", ""),
        "subgraph_context": full_context,
        "assessments": str(assessments),
        "messages": messages_history
    })
    
    advice_text: str = response.content
    
    return {
        "final_response": advice_text,
        "messages": [AIMessage(content=advice_text)]
    }
