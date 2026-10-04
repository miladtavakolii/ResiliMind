from __future__ import annotations

import random
from collections import Counter
from typing import Any

from evaluation.schemas import AssessmentProfile, AssessmentRubric

SEVERITY_MAP: dict[str, int] = {"low": 22, "moderate": 16, "high": 8}
FREQUENCY_MAP: dict[str, int] = {"rare": 22, "episodic": 16, "chronic": 8}
FUNCTIONAL_MAP: dict[str, int] = {"none": 24, "mild": 18, "moderate": 12, "severe": 6}
COPING_MAP: dict[str, int] = {"strong": 24, "moderate": 16, "weak": 8}

POLARITY_OPTIONS: dict[str, dict[str, list[str]]] = {
    "positive": {
        "severity": ["low"],
        "frequency": ["rare", "episodic"],
        "functional": ["none", "mild"],
        "coping": ["strong", "moderate"],
    },
    "negative": {
        "severity": ["moderate", "high"],
        "frequency": ["episodic", "chronic"],
        "functional": ["mild", "moderate", "severe"],
        "coping": ["weak", "moderate"],
    },
    "mixed": {
        "severity": ["moderate"],
        "frequency": ["episodic"],
        "functional": ["mild", "moderate"],
        "coping": ["moderate"],
    },
}

DIFFICULTY_TARGETS: dict[str, dict[str, int]] = {
    "easy": {
        "severity": 0,
        "frequency": 0,
        "functional": 0,
        "coping": 2,
    },
    "moderate": {
        "severity": 1,
        "frequency": 1,
        "functional": 1,
        "coping": 1,
    },
    "hard": {
        "severity": 2,
        "frequency": 2,
        "functional": 2,
        "coping": 0,
    },
    "adversarial": {
        "severity": 2,
        "frequency": 2,
        "functional": 2,
        "coping": 0,
    },
}

RANKS: dict[str, dict[str, int]] = {
    "severity": {"low": 0, "moderate": 1, "high": 2},
    "frequency": {"rare": 0, "episodic": 1, "chronic": 2},
    "functional": {"none": 0, "mild": 1, "moderate": 2, "severe": 3},
    "coping": {"weak": 0, "moderate": 1, "strong": 2},
}

DIMENSIONS: tuple[str, ...] = ("severity", "frequency", "functional", "coping")


def generate_rubric_from_profile(profile: AssessmentProfile) -> AssessmentRubric:
    """Map qualitative latent assessment dimensions to quantitative rubric scores.

    Converts the discrete levels across severity, frequency, functional impact,
    and coping capacity into calibrated rubric point values.

    Args:
        profile: AssessmentProfile instance containing qualitative ratings.

    Returns:
        AssessmentRubric: Initialized rubric model with mapped integer scores.
    """
    return AssessmentRubric(
        severity=SEVERITY_MAP[profile.severity],
        frequency=FREQUENCY_MAP[profile.frequency],
        functional=FUNCTIONAL_MAP[profile.functional],
        coping=COPING_MAP[profile.coping],
    )


def sample_assessment_profile(
    rng: random.Random,
    profile_counts: Counter[tuple[str, str, str, str, str]],
    *,
    difficulty: str,
    case_type: str,
    polarity: str,
) -> AssessmentProfile:
    """Sample independent latent assessment dimensions based on difficulty, case type, and polarity.

    Combines difficulty-constrained and polarity-aligned options across clinical
    dimensions (severity, frequency, functional impact, coping capacity). Rejects
    sampled configurations falling within boundary uncertainty zones (near 40 or 70)
    to guarantee stable resilience risk categorization.

    Args:
        rng: Dedicated random number generator instance.
        profile_counts: Frequency counter tracking profile reuse.
        difficulty: Scenario difficulty level ('easy', 'moderate', 'hard', 'adversarial').
        case_type: Case categorization ('high_risk', 'normal', 'ambiguous', etc.).
        polarity: Active signal polarity ('positive', 'negative', or 'mixed').

    Returns:
        AssessmentProfile: Sampled qualitative assessment profile.
    """
    if case_type == "high_risk":
        return AssessmentProfile(
            severity="high",
            frequency="chronic",
            functional="severe",
            coping="weak",
        )

    candidates = []

    for severity in POLARITY_OPTIONS[polarity]["severity"]:
        for frequency in POLARITY_OPTIONS[polarity]["frequency"]:
            for functional in POLARITY_OPTIONS[polarity]["functional"]:
                for coping in POLARITY_OPTIONS[polarity]["coping"]:
                    profile = {
                        "severity": severity,
                        "frequency": frequency,
                        "functional": functional,
                        "coping": coping,
                    }

                    total = (
                        SEVERITY_MAP[severity]
                        + FREQUENCY_MAP[frequency]
                        + FUNCTIONAL_MAP[functional]
                        + COPING_MAP[coping]
                    )

                    margin = min(abs(total - 40), abs(total - 70))

                    difficulty_distance = sum(
                        abs(
                            RANKS[dimension][profile[dimension]]
                            - DIFFICULTY_TARGETS[difficulty][dimension]
                        )
                        for dimension in DIMENSIONS
                    )

                    key = (
                        polarity,
                        severity,
                        frequency,
                        functional,
                        coping,
                    )
                    usage = profile_counts[key]

                    candidates.append(
                        (
                            margin,
                            difficulty_distance,
                            usage,
                            profile,
                        )
                    )

    stable = [
        candidate
        for candidate in candidates
        if candidate[0] >= 5
    ]

    pool = stable or candidates

    best_distance = min(candidate[1] for candidate in pool)
    difficulty_candidates = [
        candidate
        for candidate in pool
        if candidate[1] == best_distance
    ]

    min_usage = min(candidate[2] for candidate in difficulty_candidates)
    balanced_candidates = [
        candidate
        for candidate in difficulty_candidates
        if candidate[2] == min_usage
    ]

    selected = rng.choice(balanced_candidates)
    profile = selected[3]

    key = (
        polarity,
        profile["severity"],
        profile["frequency"],
        profile["functional"],
        profile["coping"],
    )
    profile_counts[key] += 1

    return AssessmentProfile(**profile)
