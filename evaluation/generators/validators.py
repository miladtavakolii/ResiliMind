from __future__ import annotations

from collections import Counter
from typing import Iterable

from evaluation.schemas import (
    EvaluationCase,
    GoldAssessment,
)


def validate_assessment(assessment: GoldAssessment) -> list[str]:
    """Validate the deterministic properties of a single assessment.

    The validation checks that all rubric dimensions are within the supported
    range, that the total score is valid, and that the assessment status
    matches the total score according to the ResiliMind scoring thresholds.

    Args:
        assessment: Ground-truth assessment to validate.

    Returns:
        A list of validation error messages. An empty list indicates that the
        assessment is valid.
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
            errors.append(
                f"{assessment.node_id}: {name}={value} is outside [0, 25]"
            )

    total = rubric.total_score

    if not 0 <= total <= 100:
        errors.append(
            f"{assessment.node_id}: total score {total} is outside [0, 100]"
        )

    expected_status = (
        "GREEN"
        if total >= 70
        else "YELLOW"
        if total >= 40
        else "RED"
    )

    if assessment.status != expected_status:
        errors.append(
            f"{assessment.node_id}: invalid status "
            f"{assessment.status}; expected {expected_status}"
        )

    return errors


def validate_case(
    case: EvaluationCase,
    valid_node_ids: set[str],
) -> list[str]:
    """Validate the internal consistency of a complete evaluation case.

    The validation covers case identifiers, signal and assessment node
    references, duplicate signals and assessments, signal-to-assessment
    consistency, safety labels, and routing decisions.

    Assessment-to-signal consistency is intentionally not enforced because
    future evaluation versions may support derived or multi-node assessments
    that do not correspond directly to a single extracted signal.

    Args:
        case: Evaluation case to validate.
        valid_node_ids: Set of valid node identifiers from the knowledge graph.

    Returns:
        A list of validation error messages. An empty list indicates that the
        case is valid.
    """
    errors: list[str] = []

    if not case.case_id:
        errors.append("case_id is empty")

    signal_node_ids = []

    for signal in case.gold.extraction.active_signals:
        signal_node_ids.append(signal.node_id)

        if signal.node_id not in valid_node_ids:
            errors.append(
                f"{case.case_id}: unknown signal node "
                f"{signal.node_id}"
            )

    duplicates = [
        node_id
        for node_id, count in Counter(signal_node_ids).items()
        if count > 1
    ]

    for node_id in duplicates:
        errors.append(
            f"{case.case_id}: duplicate signal for node {node_id}"
        )

    assessment_node_ids = []

    for assessment in case.gold.assessment.assessments:
        assessment_node_ids.append(assessment.node_id)

        if assessment.node_id not in valid_node_ids:
            errors.append(
                f"{case.case_id}: unknown assessment node "
                f"{assessment.node_id}"
            )

        errors.extend(validate_assessment(assessment))

    duplicates = [
        node_id
        for node_id, count in Counter(assessment_node_ids).items()
        if count > 1
    ]

    for node_id in duplicates:
        errors.append(
            f"{case.case_id}: duplicate assessment for node {node_id}"
        )

    signal_nodes = set(signal_node_ids)
    assessment_nodes = set(assessment_node_ids)

    missing_assessments = signal_nodes - assessment_nodes

    for node_id in missing_assessments:
        errors.append(
            f"{case.case_id}: signal {node_id} has no assessment"
        )

    safety = case.gold.safety

    if safety.is_high_risk and safety.risk_category == "SAFE":
        errors.append(
            f"{case.case_id}: high-risk flag cannot have SAFE category"
        )

    if not safety.is_high_risk and safety.risk_category != "SAFE":
        errors.append(
            f"{case.case_id}: non-high-risk case must have SAFE category"
        )

    route = case.gold.routing.expected_route

    if safety.is_high_risk and route != "emergency_response":
        errors.append(
            f"{case.case_id}: high-risk case must route to "
            "emergency_response"
        )

    if not safety.is_high_risk and route == "emergency_response":
        errors.append(
            f"{case.case_id}: SAFE case cannot route to "
            "emergency_response"
        )

    if route == "questioner":
        if case.gold.routing.confidence_class != "low":
            errors.append(
                f"{case.case_id}: questioner route should use low "
                "confidence class"
            )

    if route == "advisor":
        if case.gold.routing.confidence_class != "high":
            errors.append(
                f"{case.case_id}: advisor route should use high "
                "confidence class"
            )

    return errors


def validate_dataset(
    cases: Iterable[EvaluationCase],
    valid_node_ids: set[str],
) -> None:
    """Validate the complete generated evaluation dataset.

    The function verifies that the dataset is non-empty, that all case
    identifiers are unique, and that every individual case satisfies the
    structural and semantic validation rules defined by ``validate_case``.

    Validation errors from all cases are collected before raising an
    exception, allowing multiple dataset problems to be diagnosed in a
    single run.

    Args:
        cases: Iterable containing the evaluation cases to validate.
        valid_node_ids: Set of valid node identifiers from the knowledge graph.

    Raises:
        ValueError: If the dataset is empty or contains one or more
            validation errors.
    """
    all_errors: list[str] = []

    cases = list(cases)

    if not cases:
        raise ValueError("Dataset is empty")

    case_ids = [case.case_id for case in cases]

    duplicates = [
        case_id
        for case_id, count in Counter(case_ids).items()
        if count > 1
    ]

    for case_id in duplicates:
        all_errors.append(
            f"Duplicate case_id: {case_id}"
        )

    for case in cases:
        all_errors.extend(
            validate_case(
                case,
                valid_node_ids=valid_node_ids,
            )
        )

    if all_errors:
        formatted = "\n".join(
            f"- {error}"
            for error in all_errors
        )

        raise ValueError(
            f"Dataset validation failed with "
            f"{len(all_errors)} error(s):\n{formatted}"
        )
