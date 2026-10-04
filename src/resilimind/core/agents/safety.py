import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage

from ..state import AgentState
from ...llm import prompts
from ...schemas.models import SafetyOutput
from .common import llm_engine
from .text_utils import normalize_persian_text

logger = logging.getLogger(__name__)

CRISIS_PHRASES: list[str] = [
    "خودکشی",
    "خودمو بکشم",
    "خودم رو بکشم",
    "خودکشی کنم",
    "نمیخوام زنده باشم",
    "نمی خواهم زنده باشم",
    "میخوام بمیرم",
    "می خواهم بمیرم",
    "پایان بدم به زندگی",
    "پایان دادن به زندگی",
    "رگمو بزنم",
    "رگم رو بزنم",
    "رگم بزنم",
    "قصدم خودکشیه",
    "خسته شدم از زندگی میخوام بمیرم",
]


def safety_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Primary zero-tolerance safety gate.
    Pipeline: Raw Input -> Persian Normalization -> High-Recall Heuristic -> LLM Classifier -> Policy Router
    """
    logger.info("[Safety] Safety Gate is checking for high-risk signals...")
    user_msg: str = state.get("user_message", "")
    
    # 1. Text Normalization
    normalized_msg = normalize_persian_text(user_msg)
    
    # 2. Context-Aware Crisis Intent Phrases (High-Recall, Low-False-Positive)
    if any(phrase in normalized_msg for phrase in CRISIS_PHRASES):
        logger.warning("[Safety] Fast heuristic triggered high-risk flag on normalized input.")
        return {
            "safety_status": "HIGH_RISK",
            "safety_flag": True,
            "safety_risk_category": "SELF_HARM",
        }
        
    # 3. LLM-based Safety Classification (Context-aware fallback)
    try:
        safety_chain: Any = llm_engine.get_safety_runner(prompts.SAFETY_CLASSIFIER_PROMPT)
        raw_result: SafetyOutput = safety_chain.invoke({"user_message": user_msg})
        
        if raw_result.get("parsed") is None:
            raw = raw_result.get("raw")
            logger.error("[Safety] Structured output parsing failed. Raw model output: %r", raw)
            raise ValueError(f"Safety returned invalid structured output: {raw!r}")

        result: SafetyOutput = raw_result["parsed"]

        effective_high_risk = (
            result.is_high_risk or result.risk_category != "SAFE"
        )

        if result.is_high_risk != (result.risk_category != "SAFE"):
            logger.warning("[Safety] Inconsistent classifier output: is_high_risk=%s, risk_category=%s. Applying fail-safe high-risk interpretation.", result.is_high_risk, result.risk_category)

        if effective_high_risk:
            logger.warning("[Safety] LLM Safety Classifier flagged high-risk signal. Category: %s", result.risk_category)
            return {
                "safety_status": "HIGH_RISK",
                "safety_flag": True,
                "safety_risk_category": result.risk_category,
            }

        logger.debug("[Safety] Input evaluated as SAFE.")
        return {
            "safety_status": "SAFE",
            "safety_flag": False,
            "safety_risk_category": "SAFE",
        }
        
    except Exception as e:
        logger.error("[Safety] Safety LLM execution failed (%s). Defaulting to SAFETY_UNAVAILABLE status.", e)
        return {
            "safety_status": "UNAVAILABLE",
            "safety_flag": False,
            "safety_risk_category": "SAFE",
        }


def service_unavailable_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles cases where the safety subsystem or LLM backend is unavailable, 
    preventing un-vetted processing while gracefully informing the user.

    Args:
        state (AgentState): The current state dictionary of the workflow.

    Returns:
        Dict[str, Any]: A state update dictionary containing the formatted service 
                        unavailable message and appended AI message history.
    """
    logger.error("[ServiceUnavailable] Pipeline halted: Safety subsystem is unavailable.")
    unavailable_text: str = (
        "⚠️ **The system is temporarily experiencing some issues with the safety assessment section.**\n\n"
        "For security reasons, it is not possible to continue the psychological analysis at this time. "
        "Please try again in a few minutes or contact the help desk if you need immediate support."
    )
    return {
        "route": "service_unavailable",
        "final_response": unavailable_text,
        "messages": [AIMessage(content=unavailable_text)]
    }


def emergency_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a deterministic emergency intervention response with official crisis hotlines,
    bypassing LLM generation completely to avoid hallucinations or clinical risks.

    Args:
        state (AgentState): Current graph state.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.critical("[Emergency] 🚨 HIGH RISK DETECTED: Routing to Emergency Protocols!")
    
    # Load static, clinically approved crisis text directly from prompt configuration
    emergency_text: str = prompts.EMERGENCY_RESPONSE_TEMPLATE
    
    return {
        "route": "emergency_response",
        "final_response": emergency_text,
        "messages": [AIMessage(content=emergency_text)]
    }
