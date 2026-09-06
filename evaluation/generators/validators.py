from __future__ import annotations

from collections import Counter
from typing import Iterable

from evaluation.schemas import EvaluationCase, GoldAssessment


def validate_assessment(assessment: GoldAssessment) -> list[str]:
    """Validate the deterministic properties of an assessment.

    Args:
        assessment: The gold assessment object containing rubric scores and status.

    Returns:
        List of validation error messages, or an empty list if valid.
    """
    errors: list[str] = []
    rubric = assessment.rubric

    dimensions = {
        "severity": rubric.severity,
        "frequency": rubric.frequency,
        "functional": rubric.functional,
        "coping": rubric.coping,
    }

    for name, value in dimensions.items():
        if not 0 <= value <= 25:
            errors.append(f"{assessment.node_id}: {name}={value} is outside [0, 25]")

    total = rubric.total_score
    if not 0 <= total <= 100:
        errors.append(f"{assessment.node_id}: total score {total} is outside [0, 100]")

    expected_status = "GREEN" if total >= 70 else "YELLOW" if total >= 40 else "RED"
    if assessment.status != expected_status:
        errors.append(
            f"{assessment.node_id}: invalid status {assessment.status}; expected {expected_status}"
        )

    return errors


def validate_case_structure(case: EvaluationCase, valid_node_ids: set[str]) -> list[str]:
    """Validate the structural consistency of an evaluation case.

    Args:
        case: EvaluationCase instance to validate.
        valid_node_ids: Set of valid resilience graph node identifiers.

    Returns:
        List of structural error messages, or an empty list if valid.
    """
    errors: list[str] = []

    if not case.case_id:
        errors.append("case_id is empty")

    signal_node_ids = [signal.node_id for signal in case.gold.extraction.active_signals]
    for node_id in signal_node_ids:
        if node_id not in valid_node_ids:
            errors.append(f"{case.case_id}: unknown signal node {node_id}")

    duplicate_signal_nodes = [
        node_id for node_id, count in Counter(signal_node_ids).items() if count > 1
    ]
    for node_id in duplicate_signal_nodes:
        errors.append(f"{case.case_id}: duplicate signal for node {node_id}")

    assessment_node_ids = [
        assessment.node_id for assessment in case.gold.assessment.assessments
    ]
    for node_id in assessment_node_ids:
        if node_id not in valid_node_ids:
            errors.append(f"{case.case_id}: unknown assessment node {node_id}")

    duplicate_assessment_nodes = [
        node_id for node_id, count in Counter(assessment_node_ids).items() if count > 1
    ]
    for node_id in duplicate_assessment_nodes:
        errors.append(f"{case.case_id}: duplicate assessment for node {node_id}")

    for assessment in case.gold.assessment.assessments:
        errors.extend(validate_assessment(assessment))

    signal_nodes = set(signal_node_ids)
    assessment_nodes = set(assessment_node_ids)
    for node_id in signal_nodes - assessment_nodes:
        errors.append(f"{case.case_id}: signal {node_id} has no assessment")

    safety = case.gold.safety
    if safety.is_high_risk and safety.risk_category == "SAFE":
        errors.append(f"{case.case_id}: high-risk case cannot have SAFE category")

    if not safety.is_high_risk and safety.risk_category != "SAFE":
        errors.append(f"{case.case_id}: non-high-risk case must have SAFE category")

    route = case.gold.routing.expected_route
    if safety.is_high_risk:
        if route != "emergency_response":
            errors.append(f"{case.case_id}: high-risk case must route to emergency_response")
    elif route == "emergency_response":
        errors.append(f"{case.case_id}: SAFE case cannot route to emergency_response")

    if route == "questioner" and case.gold.routing.confidence_class != "low":
        errors.append(f"{case.case_id}: questioner route requires low confidence class")

    if route == "advisor" and case.gold.routing.confidence_class != "high":
        errors.append(f"{case.case_id}: advisor route requires high confidence class")

    return errors


def validate_rendered_input(case: EvaluationCase, *, require_rendered: bool = False) -> list[str]:
    """Validate generated natural-language input and evidence alignment.

    Args:
        case: EvaluationCase instance containing rendered messages and evidence spans.
        require_rendered: If True, enforce that rendered input messages are present in the case.

    Returns:
        List of input and evidence validation error messages, or an empty list if valid.
    """
    errors: list[str] = []
    messages = case.input.messages
    expected_turn_count = case.scenario.turn_count

    if require_rendered:
        if len(messages) != expected_turn_count:
            errors.append(
                f"{case.case_id}: expected {expected_turn_count} messages, got {len(messages)}"
            )

    for index, message in enumerate(messages):
        if not message.strip():
            errors.append(f"{case.case_id}: message {index} is empty")

    if require_rendered:
        signals = case.gold.extraction.active_signals
        for signal in signals:
            if not signal.evidence:
                errors.append(f"{case.case_id}: missing evidence for {signal.node_id}")
                continue

            if signal.evidence_message_index is None:
                errors.append(
                    f"{case.case_id}: missing evidence message index for {signal.node_id}"
                )
                continue

            message_index = signal.evidence_message_index
            if not 0 <= message_index < len(messages):
                errors.append(
                    f"{case.case_id}: invalid evidence message index {message_index} for {signal.node_id}"
                )
                continue

            message = messages[message_index]
            if signal.evidence_start is None or signal.evidence_end is None:
                errors.append(
                    f"{case.case_id}: missing evidence span for {signal.node_id}"
                )
                continue

            span = message[signal.evidence_start : signal.evidence_end]
            if span != signal.evidence:
                errors.append(
                    f"{case.case_id}: evidence span mismatch for {signal.node_id}"
                )

    return errors


def validate_scenario_gold_alignment(case: EvaluationCase) -> list[str]:
    errors: list[str] = []
    scenario = case.scenario
    expected = {
        "severity": {"low": 22, "moderate": 16, "high": 8}[scenario.severity_level],
        "frequency": {"rare": 22, "episodic": 16, "chronic": 8}[scenario.frequency_level],
        "functional": {"none": 24, "mild": 18, "moderate": 12, "severe": 6}[scenario.functional_level],
        "coping": {"strong": 24, "moderate": 16, "weak": 8}[scenario.coping_level],
    }

    for assessment in case.gold.assessment.assessments:
        rubric = assessment.rubric
        for dimension, expected_value in expected.items():
            actual_value = getattr(rubric, dimension)
            if actual_value != expected_value:
                errors.append(
                    f"{case.case_id}: {assessment.node_id} {dimension}={actual_value}, "
                    f"expected {expected_value} from scenario profile"
                )

    return errors


def validate_dataset(cases: Iterable[EvaluationCase], valid_node_ids: set[str], require_rendered: bool = False) -> None:
    """Validate a complete evaluation dataset.

    Args:
        cases: Iterable of EvaluationCase objects.
        valid_node_ids: Set of valid resilience graph node identifiers.

    Raises:
        ValueError: If the dataset is empty or if any validation checks fail.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("Dataset is empty")

    errors: list[str] = []
    case_ids = [case.case_id for case in cases]

    duplicate_case_ids = [
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    ]
    for case_id in duplicate_case_ids:
        errors.append(f"Duplicate case_id: {case_id}")

    for case in cases:
        errors.extend(validate_case_structure(case, valid_node_ids=valid_node_ids))
        errors.extend(validate_scenario_gold_alignment(case))
        errors.extend(validate_rendered_input(case, require_rendered=require_rendered))

    if errors:
        formatted_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n{formatted_errors}"
        )
