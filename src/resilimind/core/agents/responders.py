import logging
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..state import AgentState
from ..database import get_user_node_timeline
from ...llm import prompts
from .common import llm_engine

logger = logging.getLogger(__name__)


def questioner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a targeted, empathetic clarification question when input is ambiguous,
    using the full conversational memory.

    Args:
        state (AgentState): Current state with 'messages' and 'subgraph_context'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.info("[Questioner] Questioner Agent is formulating clarification...")
    context: str = state.get("subgraph_context", "")
    
    # Fetch the full chat history from the graph state
    messages_history = state.get("messages", [])

    stream_handler = config.get("configurable", {}).get("stream_handler")
    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_questioner_prompt()
    chain = prompt | conversational_llm    
    question_text = ""
    for chunk in chain.stream({
        "user_message": state.get("user_message", ""),
        "subgraph_context": context,
        "messages": messages_history
    }):
        question_text += chunk.content
        if stream_handler:
            stream_handler.on_llm_new_token(chunk.content)

    logger.debug("[Questioner] Clarification question successfully streamed.")
    return {
        "route": "questioner",
        "final_response": question_text,
        "messages": [AIMessage(content=question_text)]
    }


def advisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates tailored psychological advice and interventions using active graph guidance
    combined with the user's historical resilience profile and full conversational memory.

    Args:
        state (AgentState): Current state with 'user_id', 'messages', 'subgraph_context', and 'assessments'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.info("[Advisor] Advisor Agent is generating psychological interventions with full memory...")
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
        logger.debug("[Advisor] Loaded timeline history for user %s.", user_id)
    
    # Combine real-time graph context with the user's historical profile
    full_context: str = (
        f"=== CURRENT GRAPH KNOWLEDGE ===\n{subgraph_context}\n\n"
        f"=== USER HISTORICAL RESILIENCE PROFILE ===\n{history_context}"
    )

    stream_handler = config.get("configurable", {}).get("stream_handler")
    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_advisor_prompt()
    chain = prompt | conversational_llm    
    advice_text = ""
    for chunk in chain.stream({
        "user_message": state.get("user_message", ""),
        "subgraph_context": full_context,
        "assessments": str(assessments),
        "messages": messages_history
    }):
        advice_text += chunk.content
        if stream_handler:
            stream_handler.on_llm_new_token(chunk.content)

    if not advice_text or not advice_text.strip():
        logger.error("[Advisor] LLM returned an empty response.")
        advice_text = "متأسفانه در پردازش پاسخ خطایی رخ داد. لطفاً صفحه را رفرش کرده یا نشست جدیدی آغاز کنید."
    else:
        logger.debug("[Advisor] Advice successfully streamed and generated.")
    
    return {
        "route": "advisor",
        "final_response": advice_text,
        "messages": [AIMessage(content=advice_text)]
    }
