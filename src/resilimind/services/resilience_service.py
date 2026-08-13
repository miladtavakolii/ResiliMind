import streamlit as st
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from resilimind.core.database import save_resilience_log

def process_user_message(app: Any, config: Dict[str, Any], user_input: str, user_id: int) -> str:
    """Invokes the LangGraph pipeline and persists assessments to the DB."""
    initial_state: Dict[str, Any] = {
        "user_id": user_id,
        "user_message": user_input,
        "active_nodes": [],
        "subgraph_context": "",
        "assessments": [],
        "requires_disambiguation": False,
        "final_response": "",
        "messages": [HumanMessage(content=user_input)]
    }

    final_state: Dict[str, Any] = app.invoke(initial_state, config=config)

    new_assessments: List[Dict[str, Any]] = final_state.get("assessments", [])
    for assessment in new_assessments:
        save_resilience_log(
            user_id=user_id,
            node_id=assessment.get("node_id", ""),
            category=assessment.get("category", "Personal_Resilience"),
            status=assessment.get("status", "YELLOW"),
            score=int(assessment.get("score", 50)),
            confidence=float(assessment.get("confidence", 0.0)),
            reasoning=assessment.get("reasoning", "")
        )

    st.session_state.last_state = final_state
    return final_state.get("final_response", "پاسخی از سمت سیستم دریافت نشد.")
