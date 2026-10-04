from __future__ import annotations

import random
from typing import Any

DEFAULT_DISTRIBUTION: dict[str, int] = {
    "easy_normal": 20,
    "moderate_normal": 20,
    "hard_ambiguous": 15,
    "moderate_mixed_signal": 15,
    "hard_multi_domain": 10,
    "hard_high_risk": 10,
    "adversarial": 10,
}

BUCKET_CONFIG: dict[str, tuple[str, str]] = {
    "easy_normal": ("easy", "normal"),
    "moderate_normal": ("moderate", "normal"),
    "hard_ambiguous": ("hard", "ambiguous"),
    "moderate_mixed_signal": ("moderate", "mixed_signal"),
    "hard_multi_domain": ("hard", "multi_domain"),
    "hard_high_risk": ("hard", "high_risk"),
    "adversarial": ("adversarial", "adversarial"),
}


def scale_distribution(
    count: int,
    base_distribution: dict[str, int] = DEFAULT_DISTRIBUTION,
) -> dict[str, int]:
    """Scale the default scenario distribution to a target size using largest remainder.

    Args:
        count: Target total number of evaluation scenarios.
        base_distribution: Base distribution weights mapping bucket to counts.

    Returns:
        dict[str, int]: Scaled bucket distribution summing to the requested count.
    """
    if count == 100 and base_distribution == DEFAULT_DISTRIBUTION:
        return dict(DEFAULT_DISTRIBUTION)

    keys = list(base_distribution.keys())
    weights = list(base_distribution.values())
    total_weights = sum(weights)

    raw = [count * weight / total_weights for weight in weights]
    floors = [int(value) for value in raw]
    remainder = count - sum(floors)

    fractional = sorted(
        range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True
    )
    for i in fractional[:remainder]:
        floors[i] += 1

    return dict(zip(keys, floors))


def choose_domain(
    rng: random.Random,
    nodes: dict[str, dict[str, Any]],
    *,
    case_type: str,
) -> str:
    """Select a domain randomly from available graph nodes.

    Args:
        rng: Random number generator.
        nodes: Mapping of graph node IDs to their attributes.
        case_type: Scenario case type category.

    Returns:
        str: Selected domain name.

    Raises:
        ValueError: If no valid domains are found in the graph.
    """
    domains = sorted(
        {node["domain"] for node in nodes.values() if "domain" in node}
    )
    if not domains:
        raise ValueError("No domains found in knowledge graph")
    return rng.choice(domains)


def choose_turn_count(
    rng: random.Random,
    *,
    case_type: str,
) -> int:
    """Determine the number of user turns based on the case type.

    Args:
        rng: Random number generator.
        case_type: Scenario case type category.

    Returns:
        int: Number of conversation turns.
    """
    if case_type == "multi_domain":
        return rng.choice([2, 3])
    if case_type == "ambiguous":
        return 1
    if case_type == "adversarial":
        return rng.choice([1, 2])
    return 1
