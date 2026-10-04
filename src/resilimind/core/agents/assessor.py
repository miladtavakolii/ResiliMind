from collections import Counter
import logging
from typing import Dict, Any, List

from ..state import AgentState
from ..config import settings
from ...graph.retriever import retrieve_subgraph_context
from ...llm import prompts
from ...schemas.models import AssessmentOutput
from .common import llm_engine, resilience_graph
from .text_utils import normalize_persian_text, find_polarity_cues as _find_polarity_cues

logger = logging.getLogger(__name__)


def calculate_evidence_quality(
    evidence: str,
    node_id: str,
) -> float:
    """Estimate the quality of extracted evidence for routing confidence."""
    normalized_evidence = normalize_persian_text(evidence)

    if not normalized_evidence:
        return 0.0

    token_count = len(normalized_evidence.split())

    if token_count <= 2:
        return 0.85
    if token_count <= 16:
        return 1.0
    if token_count <= 24:
        return 0.9

    return 0.75


def calculate_composite_confidence(
    raw_confidence: float,
    signal_polarity: str,
    evidence_text: str,
    node_id: str,
    user_message: str,
    assessment_score: int,
) -> float:
    """Compute a deterministic heuristic confidence score for routing.

    Combines:
        - 30% LLM self-reported confidence
        - 30% extracted evidence quality
        - 40% knowledge/assessment consistency

    This is a composite heuristic confidence score, not a calibrated probability.

    Args:
        raw_confidence: Model self-reported assessment confidence (0.0 to 1.0).
        signal_polarity: Detected signal polarity ('positive', 'negative', 'mixed').
        evidence_text: Extracted supporting evidence phrase.
        node_id: Target knowledge graph node identifier.
        user_message: Raw user message.
        assessment_score: Total quantitative assessment score (0 to 100).

    Returns:
        float: Heuristic confidence score bounded between 0.0 and 1.0.
    """
    llm_confidence = min(max(raw_confidence, 0.0), 1.0)

    node_data = resilience_graph.nodes.get(node_id, {})
    cues = node_data.get("cues", {})

    positive_hits = _find_polarity_cues(
        evidence_text,
        cues.get("positive_keywords", []),
    )
    negative_hits = _find_polarity_cues(
        evidence_text,
        cues.get("negative_keywords", []),
    )

    polarity_lower = signal_polarity.lower()

    if positive_hits and negative_hits:
        polarity_consistency = (
            1.0 if polarity_lower == "mixed" else 0.0
        )
    elif positive_hits:
        polarity_consistency = (
            1.0 if polarity_lower == "positive" else 0.0
        )
    elif negative_hits:
        polarity_consistency = (
            1.0 if polarity_lower == "negative" else 0.0
        )
    else:
        polarity_consistency = 0.5

    evidence_quality = calculate_evidence_quality(
        evidence=evidence_text,
        node_id=node_id,
    )

    composite_score = (
        0.50 * llm_confidence
        + 0.30 * evidence_quality
        + 0.20 * polarity_consistency
    )

    final_confidence = round(
        min(max(composite_score, 0.0), 1.0),
        2,
    )

    logger.debug(
        "[Confidence] node=%s confidence=%s raw=%s "
        "evidence_quality=%s polarity_consistency=%s score=%s",
        node_id,
        final_confidence,
        raw_confidence,
        evidence_quality,
        polarity_consistency,
        assessment_score,
    )

    return final_confidence


def validate_assessment_result(result: AssessmentOutput, active_signals: list[dict[str, Any]]) -> AssessmentOutput:
    """Validate assessor output against active signals and graph topology.

    Verifies 1-to-1 parity between assessed nodes and active extraction signals,
    confirms node existence in the knowledge graph, checks for duplicate
    assessments, and ensures each assessment's category matches the node's domain.

    Args:
        result: AssessmentOutput instance containing scored node assessments.
        active_signals: List of extracted active signal dictionaries for the turn.

    Returns:
        AssessmentOutput: The validated assessment output instance.

    Raises:
        ValueError: If duplicate assessments are found, unknown or inactive nodes
            are assessed, active nodes are omitted, or an assessment category
            mismatches the node's domain in the resilience graph.
    """
    valid_node_ids = set(resilience_graph.nodes)
    signal_set = {signal["node_id"] for signal in active_signals}

    assessment_ids = [assessment.node_id for assessment in result.assessments]
    assessment_set = set(assessment_ids)

    if len(assessment_ids) != len(assessment_set):
        counts = Counter(assessment_ids)
        duplicate_ids = sorted(
            node_id for node_id, count in counts.items() if count > 1
        )
        raise ValueError(
            f"Assessor returned duplicate assessments: {duplicate_ids}"
        )

    if unknown_ids := sorted(assessment_set - valid_node_ids):
        raise ValueError(f"Assessor returned unknown nodes: {unknown_ids}")

    if extra_ids := sorted(assessment_set - signal_set):
        raise ValueError(
            f"Assessor returned assessments for inactive nodes: {extra_ids}"
        )

    if missing_ids := sorted(signal_set - assessment_set):
        raise ValueError(
            f"Assessor missing assessments for active nodes: {missing_ids}"
        )

    for assessment in result.assessments:
        expected_domain = resilience_graph.nodes[assessment.node_id].get("domain")
        if assessment.category != expected_domain:
            raise ValueError(
                f"Assessor returned wrong category for {assessment.node_id}: "
                f"expected={expected_domain}, got={assessment.category}"
            )

    return result


def assessor_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates resilience levels independently for each extracted node using
    node-specific evidence and retrieved graph context.

    Args:
        state: Current state containing 'user_message', 'subgraph_context',
            and 'active_signals'.

    Returns:
        Dict[str, Any]: Updated state with 'assessments' and
            'requires_disambiguation'.
    """
    logger.info(
        "[Assessor] Assessor Agent is evaluating resilience status "
        "with evidence..."
    )

    user_msg: str = state.get("user_message", "")
    active_signals: List[Dict[str, Any]] = state.get("active_signals", [])

    if not active_signals:
        logger.info("[Assessor] No active signals. Skipping assessment.")
        return {
            "assessments": [],
            "requires_disambiguation": False,
        }

    assessor_chain = llm_engine.get_assessor_runner(
        prompts.ASSESSOR_SYSTEM_PROMPT
    )

    max_assessment_attempts = 3
    assessments_list: List[Dict[str, Any]] = []
    requires_disambiguation_override = False

    for signal in active_signals:
        node_id = signal["node_id"]
        evidence = signal.get("evidence", "")
        polarity = signal.get("detected_signal", "mixed")

        target_context = retrieve_subgraph_context(
            resilience_graph,
            [node_id],
        )

        enriched_input = (
            "=== TARGET NODE ===\n"
            f"{node_id}\n\n"
            "=== TARGET SIGNAL ===\n"
            f"Polarity: {polarity.upper()}\n"
            f"Exact Evidence: \"{evidence}\"\n\n"
            "=== FULL USER MESSAGE ===\n"
            f"{user_msg}\n\n"
            "RULES:\n"
            "Assess ONLY the target node.\n"
            "Return exactly one assessment for this target node.\n"
            "Use the exact evidence assigned to this node as the primary evidence.\n"
            "Use the full user message only to clarify the local meaning of this evidence.\n"
            "Do not borrow severity, frequency, functional impact, or coping "
            "evidence from another resilience concept.\n"
        )

        last_validation_error: Exception | None = None

        for attempt in range(max_assessment_attempts):
            attempt_input = enriched_input

            if last_validation_error is not None:
                attempt_input += (
                    "\n\n=== PREVIOUS ASSESSMENT VALIDATION FAILURE ===\n"
                    f"{last_validation_error}\n\n"
                    "Regenerate the assessment completely.\n"
                    "Return exactly one assessment.\n"
                    f"The only allowed node_id is {node_id}.\n"
                    "Do not assess any other node.\n"
                    "Ensure all four dimensions contain valid rubric values.\n"
                )

            raw_result = assessor_chain.invoke({
                "user_message": attempt_input,
                "subgraph_context": target_context,
            })

            if raw_result.get("parsed") is None:
                raw = raw_result.get("raw")

                last_validation_error = ValueError(
                    f"Assessor returned invalid structured output: {raw!r}"
                )

                logger.warning(
                    "[Assessor] Structured output validation failed for %s "
                    "(attempt %d/%d). Retrying...",
                    node_id,
                    attempt + 1,
                    max_assessment_attempts,
                )
                continue

            result = raw_result["parsed"]

            try:
                if len(result.assessments) != 1:
                    raise ValueError(
                        f"Expected exactly one assessment for {node_id}, "
                        f"got {len(result.assessments)}"
                    )

                result = validate_assessment_result(
                    result=result,
                    active_signals=[signal],
                )

                assessment = result.assessments[0]

                if assessment.node_id != node_id:
                    raise ValueError(
                        f"Assessor returned wrong node: "
                        f"expected={node_id}, got={assessment.node_id}"
                    )

                assessment_dict = assessment.model_dump()
                assessment_dict["score"] = assessment.score
                assessment_dict["status"] = assessment.status

                routing_confidence = calculate_composite_confidence(
                    raw_confidence=assessment_dict["confidence"],
                    signal_polarity=polarity,
                    evidence_text=evidence,
                    node_id=node_id,
                    user_message=user_msg,
                    assessment_score=assessment_dict["score"],
                )

                assessment_dict["confidence"] = routing_confidence

                assessments_list.append(assessment_dict)

                if (
                    result.requires_disambiguation
                    or routing_confidence
                    < settings.RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD
                ):
                    logger.warning(
                        "[Assessor] Low confidence/disambiguation for %s "
                        "(confidence=%s, model_disambiguation=%s)",
                        node_id,
                        routing_confidence,
                        result.requires_disambiguation,
                    )
                    requires_disambiguation_override = True

                logger.debug(
                    "[Assessor] Successfully assessed %s "
                    "(score=%s, status=%s, confidence=%s)",
                    node_id,
                    assessment_dict["score"],
                    assessment_dict["status"],
                    routing_confidence,
                )

                break

            except ValueError as exc:
                last_validation_error = exc

                logger.warning(
                    "[Assessor] Validation failed for %s "
                    "(attempt %d/%d): %s",
                    node_id,
                    attempt + 1,
                    max_assessment_attempts,
                    exc,
                )

        else:
            raise RuntimeError(
                f"[Assessor] Failed to produce assessment for node "
                f"{node_id} after {max_assessment_attempts} attempts"
            ) from last_validation_error

    return {
        "assessments": assessments_list,
        "requires_disambiguation": requires_disambiguation_override,
    }
