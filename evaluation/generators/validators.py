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

def _validate_assessment_profile_polarity(case: EvaluationCase) -> list[str]:
    """Validate that per-node assessment profiles align with gold signal polarities.

    Checks that every configured assessment profile corresponds to an active gold signal,
    that the signal's polarity is recognized, and that qualitative dimension ratings
    (severity, frequency, functional impact, coping capacity) adhere strictly to the
    clinical bounds permitted for that polarity.

    Args:
        case: EvaluationCase instance containing scenario assessment profiles
            and gold extraction signals.

    Returns:
        list[str]: Formatted error messages detailing missing signals, unknown polarities,
            or incompatible qualitative dimensions, empty if all profiles are valid.
    """
    errors: list[str] = []
    profiles = case.scenario.assessment_profiles

    allowed = {
        "positive": {
            "severity": {"low", "moderate"},
            "frequency": {"rare", "episodic"},
            "functional": {"none", "mild"},
            "coping": {"strong", "moderate"},
        },
        "negative": {
            "severity": {"moderate", "high"},
            "frequency": {"episodic", "chronic"},
            "functional": {"mild", "moderate", "severe"},
            "coping": {"weak", "moderate"},
        },
        "mixed": {
            "severity": {"moderate"},
            "frequency": {"episodic"},
            "functional": {"mild", "moderate"},
            "coping": {"moderate"},
        },
    }

    signals = {
        signal.node_id: signal.detected_signal
        for signal in case.gold.extraction.active_signals
    }

    for node_id, profile in profiles.items():
        polarity = signals.get(node_id)
        if polarity is None:
            errors.append(
                f"{case.case_id}: assessment profile exists for "
                f"node without gold signal: {node_id}"
            )
            continue

        if polarity not in allowed:
            errors.append(
                f"{case.case_id}: unknown polarity '{polarity}' "
                f"for {node_id}"
            )
            continue

        values = {
            "severity": profile.severity,
            "frequency": profile.frequency,
            "functional": profile.functional,
            "coping": profile.coping,
        }

        for dimension, value in values.items():
            if value not in allowed[polarity][dimension]:
                errors.append(
                    f"{case.case_id}: {node_id} polarity={polarity} "
                    f"has incompatible {dimension}={value}"
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

    if case.scenario.case_type == "ambiguous":
        if case.gold.extraction.active_signals:
            errors.append(
                f"{case.case_id}: ambiguous case must not contain gold signals"
            )

        if case.gold.assessment.assessments:
            errors.append(
                f"{case.case_id}: ambiguous case must not contain gold assessments"
            )

        if case.gold.routing.expected_route != "questioner":
            errors.append(
                f"{case.case_id}: ambiguous case must route to questioner"
            )

        if case.scenario.turn_count != 1:
            errors.append(
                f"{case.case_id}: ambiguous case must contain exactly one turn"
            )
    if safety.is_high_risk:
        if case.gold.extraction.active_signals:
            errors.append(
                f"{case.case_id}: high-risk case must not contain resilience signals"
            )

        if case.gold.assessment.assessments:
            errors.append(
                f"{case.case_id}: high-risk case must not contain assessments"
            )

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
    """Validate internal consistency between scenario assessment profiles and gold rubrics.

    Ensures that every gold assessment node possesses a corresponding assessment
    profile in the scenario specification, checks for unexpected or missing node profiles,
    and validates that rubric point scores match the qualitative dimension levels
    (severity, frequency, functional impact, coping capacity).

    Args:
        case: EvaluationCase instance containing scenario metadata and gold assessments.

    Returns:
        list[str]: Formatted error messages describing alignment discrepancies or
            missing profile attributes, empty if alignment is completely consistent.
    """
    errors: list[str] = []
    scenario = case.scenario
    assessment_profiles = getattr(scenario, "assessment_profiles", {})

    expected_node_ids = {
        assessment.node_id
        for assessment in case.gold.assessment.assessments
    }
    profile_node_ids = set(assessment_profiles)

    if not expected_node_ids:
        if assessment_profiles:
            for node_id in sorted(profile_node_ids):
                errors.append(
                    f"{case.case_id}: unexpected assessment profile for "
                    f"{node_id} in case without assessments"
                )
        return errors

    missing_profiles = expected_node_ids - profile_node_ids
    unexpected_profiles = profile_node_ids - expected_node_ids

    for node_id in sorted(missing_profiles):
        errors.append(
            f"{case.case_id}: missing assessment profile for {node_id}"
        )

    for node_id in sorted(unexpected_profiles):
        errors.append(
            f"{case.case_id}: unexpected assessment profile for {node_id}"
        )

    dimension_maps = {
        "severity": {"low": 22, "moderate": 16, "high": 8},
        "frequency": {"rare": 22, "episodic": 16, "chronic": 8},
        "functional": {"none": 24, "mild": 18, "moderate": 12, "severe": 6},
        "coping": {"strong": 24, "moderate": 16, "weak": 8},
    }

    polarity_constraints = {
        "positive": {
            "severity": {"low", "moderate"},
            "frequency": {"rare", "episodic"},
            "functional": {"none", "mild"},
            "coping": {"strong", "moderate"},
        },
        "negative": {
            "severity": {"moderate", "high"},
            "frequency": {"episodic", "chronic"},
            "functional": {"mild", "moderate", "severe"},
            "coping": {"weak", "moderate"},
        },
        "mixed": {
            "severity": {"moderate"},
            "frequency": {"episodic"},
            "functional": {"mild", "moderate"},
            "coping": {"moderate"},
        },
    }

    signal_polarities = {
        signal.node_id: signal.detected_signal
        for signal in case.gold.extraction.active_signals
    }

    for assessment in case.gold.assessment.assessments:
        node_id = assessment.node_id
        profile = assessment_profiles.get(node_id)

        if profile is None:
            continue

        profile_values = {
            dimension: (
                profile.get(dimension)
                if isinstance(profile, dict)
                else getattr(profile, dimension, None)
            )
            for dimension in dimension_maps
        }

        for dimension, val_map in dimension_maps.items():
            profile_val = profile_values[dimension]
            expected_value = val_map.get(profile_val)
            actual_value = getattr(assessment.rubric, dimension, None)

            if expected_value is None:
                errors.append(
                    f"{case.case_id}: {node_id} has invalid {dimension}="
                    f"{profile_val}"
                )
                continue

            if actual_value != expected_value:
                errors.append(
                    f"{case.case_id}: {node_id} {dimension}="
                    f"{actual_value}, expected {expected_value} "
                    f"from assessment profile"
                )

        polarity = signal_polarities.get(node_id)

        if polarity is None:
            errors.append(
                f"{case.case_id}: assessment profile for {node_id} "
                f"has no corresponding gold signal"
            )
            continue

        constraints = polarity_constraints.get(polarity)

        if constraints is None:
            errors.append(
                f"{case.case_id}: {node_id} has unknown signal polarity "
                f"{polarity}"
            )
            continue

        for dimension, allowed_values in constraints.items():
            profile_val = profile_values[dimension]

            if profile_val not in allowed_values:
                allowed = ", ".join(sorted(allowed_values))
                errors.append(
                    f"{case.case_id}: {node_id} polarity={polarity} "
                    f"has incompatible {dimension}={profile_val}; "
                    f"allowed: {allowed}"
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
        case_id
        for case_id, count in Counter(case_ids).items()
        if count > 1
    ]
    for case_id in duplicate_case_ids:
        errors.append(f"Duplicate case_id: {case_id}")

    for case in cases:
        errors.extend(
            validate_case_structure(
                case,
                valid_node_ids=valid_node_ids,
            )
        )
        errors.extend(validate_scenario_gold_alignment(case))
        errors.extend(
            validate_rendered_input(
                case,
                require_rendered=require_rendered,
            )
        )

    if errors:
        formatted_errors = "\n".join(
            f"- {error}"
            for error in errors
        )
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            f"{formatted_errors}"
        )
