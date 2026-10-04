from __future__ import annotations

import logging
from typing import Any

from evaluation.schemas import EvaluationCase, CasePrediction, TurnPrediction
from .common import compute_case_fingerprint

logger = logging.getLogger(__name__)


def validate_alignment(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> None:
    """Validate one-to-one alignment between cases and predictions.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.

    Raises:
        ValueError: If IDs are missing, duplicated, or unexpected.
    """
    case_map = {case.case_id: case for case in cases}
    prediction_ids = [prediction.case_id for prediction in predictions]
    prediction_id_set = set(prediction_ids)

    if len(prediction_ids) != len(prediction_id_set):
        duplicates = sorted(
            case_id
            for case_id in prediction_id_set
            if prediction_ids.count(case_id) > 1
        )
        raise ValueError(f"Duplicate case IDs found in predictions: {duplicates}")

    missing_predictions = set(case_map) - prediction_id_set
    unexpected_predictions = prediction_id_set - set(case_map)

    if missing_predictions:
        raise ValueError(
            f"Missing predictions for cases: {sorted(missing_predictions)}"
        )

    if unexpected_predictions:
        raise ValueError(
            f"Predictions contain unknown case IDs: {sorted(unexpected_predictions)}"
        )

    for prediction in predictions:
        case = case_map[prediction.case_id]
        expected_messages = case.input.messages
        predicted_messages = [
            turn.user_message
            for turn in prediction.turns
        ]

        if expected_messages != predicted_messages:
            raise ValueError(
                f"Input mismatch for {case.case_id}: "
                "predictions were generated from a different case snapshot."
            )

        if prediction.case_fingerprint:
            expected_fingerprint = compute_case_fingerprint(case)

            if prediction.case_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"Case fingerprint mismatch for {case.case_id}: "
                    "predictions and evaluation cases are not from the same snapshot."
                )


def validate_dataset_versions(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> None:
    """Ensure cases and predictions belong to the same dataset version.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.

    Raises:
        ValueError: If versions do not match.
    """
    case_versions = {case.dataset_version for case in cases}
    prediction_versions = {prediction.dataset_version for prediction in predictions}

    if len(case_versions) != 1:
        raise ValueError(
            f"Multiple dataset versions found in cases: {sorted(case_versions)}"
        )
    if len(prediction_versions) != 1:
        raise ValueError(
            f"Multiple dataset versions found in predictions: {sorted(prediction_versions)}"
        )

    case_version = next(iter(case_versions))
    prediction_version = next(iter(prediction_versions))

    if case_version != prediction_version:
        raise ValueError(
            f"Dataset version mismatch: cases={case_version}, predictions={prediction_version}"
        )


def normalize_assessments(
    assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize application assessment output for evaluation.

    Maps the four-dimensional scoring output stored under ``scores`` to the
    ``rubric`` key expected by evaluation modules.

    Args:
        assessments: List of raw assessment dictionaries produced by the application.

    Returns:
        list[dict[str, Any]]: List of normalized assessment dictionaries containing
            the ``rubric`` key.
    """
    normalized = []
    for assessment in assessments:
        item = dict(assessment)
        if "rubric" not in item:
            item["rubric"] = dict(item.get("scores", {}))
        normalized.append(item)
    return normalized


def build_evaluator_prediction(prediction: CasePrediction) -> dict[str, Any]:
    """Adapt the raw CasePrediction structure to the input contract expected by evaluators.

    Args:
        prediction: CasePrediction object containing recorded conversation turns.

    Returns:
        dict[str, Any]: Structured dictionary formatted for evaluation modules.
    """
    if not prediction.successful:
        return {
            "execution_error": prediction.error or "unknown execution error",
            "safety": {},
            "extraction": {"signals": [], "active_nodes": []},
            "assessment": {"assessments": []},
            "routing": {"route": "unknown"},
            "final_response": prediction.final_response,
            "user_context": "\n".join(
                turn.user_message for turn in prediction.turns
            ),
            "raw": prediction.model_dump(),
        }

    if not prediction.turns:
        return {
            "execution_error": "no turns were recorded",
            "safety": {},
            "extraction": {"signals": [], "active_nodes": []},
            "assessment": {"assessments": []},
            "routing": {"route": "unknown"},
            "final_response": "",
            "user_context": "",
            "raw": prediction.model_dump(),
        }

    all_signals = [
        signal
        for turn in prediction.turns
        for signal in turn.active_signals
    ]

    all_nodes = list(dict.fromkeys(
        node_id
        for turn in prediction.turns
        for node_id in turn.active_nodes
    ))

    latest_assessments: dict[str, dict[str, Any]] = {}

    for turn in prediction.turns:
        for assessment in normalize_assessments(turn.assessments):
            node_id = assessment.get("node_id")
            if node_id:
                latest_assessments[node_id] = assessment

    final_turn = prediction.turns[-1]

    is_high_risk = any(
        turn.safety_status == "HIGH_RISK" or turn.safety_flag
        for turn in prediction.turns
    )

    return {
        "safety": {
            "is_high_risk": is_high_risk,
            "status": final_turn.safety_status,
            "risk_category": final_turn.safety_risk_category,
        },
        "extraction": {
            "signals": all_signals,
            "active_nodes": all_nodes,
        },
        "assessment": {
            "assessments": list(latest_assessments.values()),
        },
        "routing": {
            "route": final_turn.route,
        },
        "final_response": prediction.final_response,
        "user_context": "\n".join(
            turn.user_message for turn in prediction.turns
        ),
        "raw": prediction.model_dump(),
    }


def build_prediction_mapping(
    predictions: list[CasePrediction],
) -> dict[str, dict[str, Any]]:
    """Build a case-ID keyed mapping for EvaluationRunner.

    Args:
        predictions: Raw benchmark predictions.

    Returns:
        Mapping from case IDs to normalized evaluator predictions.
    """
    return {
        prediction.case_id: build_evaluator_prediction(prediction)
        for prediction in predictions
    }


def get_turn_gold(case: EvaluationCase, turn_index: int) -> EvaluationCase:
    """Return a copy of the gold annotations containing only the signals that belong to one turn."""
    turn_case = case.model_copy(deep=True)
    turn_case.gold.extraction.active_signals = [
        signal
        for signal in turn_case.gold.extraction.active_signals
        if signal.evidence_message_index == turn_index
    ]
    return turn_case


def build_turn_prediction(turn: TurnPrediction) -> dict[str, Any]:
    """Build evaluator input for one conversation turn."""
    is_high_risk = turn.safety_status == "HIGH_RISK" or turn.safety_flag

    return {
        "safety": {
            "is_high_risk": is_high_risk,
            "status": turn.safety_status,
            "risk_category": turn.safety_risk_category,
        },
        "extraction": {
            "signals": turn.active_signals,
            "active_nodes": turn.active_nodes,
        },
        "assessment": {
            "assessments": normalize_assessments(turn.assessments),
        },
        "routing": {
            "route": turn.route,
        },
        "advisor_response": turn.final_response,
        "final_response": turn.final_response,
        "user_context": turn.user_message,
    }
