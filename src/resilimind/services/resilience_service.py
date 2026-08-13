from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from resilimind.core.database import save_resilience_log
from resilimind.schemas.models import ProcessResult

def process_user_message(app: Any, config: Dict[str, Any], user_input: str, user_id: int) -> ProcessResult:
    """
    Invokes the LangGraph workflow, persists new rubric assessments to the database,
    and returns a pure, framework-agnostic ProcessResult object without touching UI states.

    Args:
        app (Any): The compiled LangGraph application instance.
        config (Dict[str, Any]): The thread configuration dictionary containing thread_id.
        user_input (str): The raw text message submitted by the user.
        user_id (int): The unique identifier of the authenticated user.

    Returns:
        ProcessResult: A decoupled container holding the final response, new assessments, and state.
    """
    print("Processing user message through resilience workflow service...")
    
    initial_state: Dict[str, Any] = {
        "user_id": user_id,
        "user_message": user_input,
        "safety_flag": False,
        "active_nodes": [],
        "active_signals": [],
        "subgraph_context": "",
        "assessments": [],
        "requires_disambiguation": False,
        "final_response": "",
        "messages": [HumanMessage(content=user_input)]
    }

    try:
        final_state: Dict[str, Any] = app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        raise

    # 1. Persist new assessments into SQLite database using deterministic score/status extraction
    new_assessments: List[Dict[str, Any]] = final_state.get("assessments", [])
    for assessment in new_assessments:
        scores: Dict[str, int] = assessment.get("scores", {})
        
        # Fallback calculation if computed properties aren't pre-serialized in state dict
        total_score: int = assessment.get("score")
        if total_score is None:
            total_score = (
                scores.get("severity", 0) + 
                scores.get("frequency", 0) + 
                scores.get("functional", 0) + 
                scores.get("coping", 0)
            )

        status: str = assessment.get("status")
        if not status:
            status = "GREEN" if total_score >= 70 else ("YELLOW" if total_score >= 40 else "RED")

        save_resilience_log(
            user_id=user_id,
            node_id=assessment.get("node_id", ""),
            category=assessment.get("category", "Personal_Resilience"),
            status=status,
            score=int(total_score),
            confidence=float(assessment.get("confidence", 0.0)),
            reasoning=assessment.get("reasoning", "")
        )

    response_text: str = final_state.get("final_response", "پاسخی از سمت سیستم دریافت نشد.")

    return ProcessResult(
        final_response=response_text,
        new_assessments=new_assessments,
        state=final_state
    )
