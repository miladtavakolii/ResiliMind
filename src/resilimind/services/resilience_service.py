import logging
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from resilimind.core.database import save_resilience_log
from resilimind.schemas.models import ProcessResult

# Initialize module logger
logger = logging.getLogger(__name__)


class ResilienceService:
    """
    Encapsulates the core business logic of the resilience assessment pipeline,
    decoupling workflow execution, database persistence, and response formatting.
    """

    @staticmethod
    def process_message(app: Any, config: Dict[str, Any], user_input: str, user_id: int) -> ProcessResult:
        """
        Executes the LangGraph workflow pipeline, persists assessments, and builds the result.

        Args:
            app (Any): The compiled LangGraph application instance.
            config (Dict[str, Any]): The execution thread configuration.
            user_input (str): The raw text message submitted by the user.
            user_id (int): The unique identifier of the authenticated user.

        Returns:
            ProcessResult: The decoupled container holding the response, assessments, and final state.
        """
        logger.info("Executing resilience workflow for user...")
        
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
            logger.exception(f"Workflow execution failed: {e}")
            raise

        new_assessments: List[Dict[str, Any]] = final_state.get("assessments", [])
        
        # 1. Delegate DB persistence
        ResilienceService.persist_assessments(user_id, new_assessments)

        # 2. Build final decoupled response package
        response_text: str = ResilienceService.build_response(final_state)

        return ProcessResult(
            final_response=response_text,
            new_assessments=new_assessments,
            state=final_state
        )

    @staticmethod
    def persist_assessments(user_id: int, assessments: List[Dict[str, Any]]) -> None:
        """
        Persists newly generated node assessments into the SQLite database.

        Args:
            user_id (int): The unique identifier of the user.
            assessments (List[Dict[str, Any]]): List of assessment dictionaries from graph state.
        """
        logger.info(f"Persisting {len(assessments)} node assessments to database...")
        
        for assessment in assessments:
            scores: Dict[str, int] = assessment.get("scores", {})
            
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

    @staticmethod
    def build_response(final_state: Dict[str, Any]) -> str:
        """
        Extracts and formats the final textual response from the completed graph state.

        Args:
            final_state (Dict[str, Any]): The final execution state dictionary.

        Returns:
            str: The formatted response string.
        """
        return final_state.get("final_response", "پاسخی از سمت سیستم دریافت نشد.")


# Backward-compatible functional wrapper for UI invocation
def process_user_message(app: Any, config: Dict[str, Any], user_input: str, user_id: int) -> ProcessResult:
    """Legacy functional wrapper delegating to ResilienceService."""
    return ResilienceService.process_message(app, config, user_input, user_id)
