from __future__ import annotations

from collections import Counter
import json
from typing import Any

from evaluation.schemas import EvaluationCase


def build_assessment_requirements(node_id: str, profile: Any, semantic_boundary: str | None = None,) -> dict[str, str]:
    """Describe the observable first-person evidence required by an assessment profile."""
    values = {
        "severity": profile.severity,
        "frequency": profile.frequency,
        "functional": profile.functional,
        "coping": profile.coping,
    }

    requirements = {
        "severity": {
            "low": "حداقل یا خفیف بودن شدت را به‌صورت مستقیم بیان کن.",
            "moderate": "شدت قابل توجه اما نه طاقت‌فرسا را مستقیم بیان کن.",
            "high": "شدت زیاد، شدید یا طاقت‌فرسا را مستقیم بیان کن.",
        },
        "frequency": {
            "rare": "رخداد نادر، منفرد یا یک‌باره را صریح بیان کن.",
            "episodic": "رخداد گهگاهی، دوره‌ای یا وابسته به موقعیت را صریح بیان کن.",
            "chronic": "تداوم، استمرار، تکرار مداوم یا طولانی‌مدت بودن را صریح بیان کن.",
        },
        "functional": {
            "none": "حفظ عملکرد عادی را صریح بیان کن.",
            "mild": "اختلال جزئی و محدود در عملکرد را صریح بیان کن.",
            "moderate": "اثر معنادار بر فعالیت‌های روزمره را صریح بیان کن.",
            "severe": "اختلال شدید یا ناتوانی در انجام فعالیت‌های عادی را صریح بیان کن.",
        },
        "coping": {
            "strong": "مقابله یا سازگاری فعال و مؤثر را صریح بیان کن.",
            "moderate": "تلاش یا سازگاری وجود دارد ولی محدود است.",
            "weak": "درماندگی، تسلیم، ناتوانی در مقابله یا نبود راهکار مؤثر را صریح بیان کن.",
        },
    }

    return {
        **{
            dimension: requirements[dimension][value]
            for dimension, value in values.items()
        },
        "target_scope": semantic_boundary,
    }


def build_user_prompt(
    case: EvaluationCase,
    nodes: dict[str, dict[str, Any]],
) -> str:
    """Construct a formatted JSON prompt describing the latent scenario for the renderer.

    Aggregates target graph node signals, semantic cues, domain-confusable distractor
    nodes, and latent clinical assessment levels into a structured payload for Gemini.

    Args:
        case: Target EvaluationCase containing scenario constraints and gold signals.
        nodes: Mapping of graph node IDs to their attributes.

    Returns:
        str: Indented JSON string representing the complete scenario rendering prompt.

    Raises:
        ValueError: If a gold active signal references a node ID missing from the graph.
    """
    active_signals = []

    target_ids = [
        signal.node_id
        for signal in case.gold.extraction.active_signals
    ]

    duplicate_targets = [
        node_id
        for node_id, count in Counter(target_ids).items()
        if count > 1
    ]

    if duplicate_targets:
        raise ValueError(
            f"{case.case_id}: duplicate gold signal nodes: "
            f"{sorted(duplicate_targets)}"
        )

    target_ids_set = set(target_ids)

    for signal in case.gold.extraction.active_signals:
        node = nodes.get(signal.node_id)
        if node is None:
            raise ValueError(
                f"{case.case_id}: unknown graph node {signal.node_id}"
            )

        profile = case.scenario.assessment_profiles.get(signal.node_id)
        if profile is None:
            raise ValueError(
                f"{case.case_id}: missing assessment profile for {signal.node_id}"
            )

        cues = node.get("cues", {})
        semantic_boundary = node.get("semantic_boundary", "").strip()

        if signal.detected_signal == "positive":
            polarity_cues = cues.get("positive_keywords", [])
        elif signal.detected_signal == "negative":
            polarity_cues = cues.get("negative_keywords", [])
        else:
            polarity_cues = (
                cues.get("positive_keywords", [])
                + cues.get("negative_keywords", [])
            )

        active_signals.append(
            {
                "node_id": signal.node_id,
                "name_fa": node.get("name_fa", ""),
                "name_en": node.get("name_en", ""),
                "domain_fa": node.get("domain_fa", ""),
                "description": node.get("description", ""),
                "semantic_boundary": semantic_boundary,
                "polarity": signal.detected_signal,
                "semantic_cues": polarity_cues,
                "assessment_profile": {
                    "severity": profile.severity,
                    "frequency": profile.frequency,
                    "functional": profile.functional,
                    "coping": profile.coping,
                },
                "assessment_requirements": build_assessment_requirements(signal.node_id, profile, semantic_boundary),
            }
        )

    target_domains = {
        nodes[signal.node_id].get("domain")
        for signal in case.gold.extraction.active_signals
        if signal.node_id in nodes
    }

    confusable_nodes = []

    for node_id, node in nodes.items():
        if node_id in target_ids_set:
            continue

        if node.get("domain") not in target_domains:
            continue

        confusable_nodes.append(
            {
                "node_id": node_id,
                "name_fa": node.get("name_fa", ""),
                "name_en": node.get("name_en", ""),
                "description": node.get("description", ""),
            }
        )

    scenario = {
        "case_type": case.scenario.case_type,
        "difficulty": case.scenario.difficulty,
        "turn_count": case.scenario.turn_count,
        "safety_category": case.gold.safety.risk_category,
        "target_signals": active_signals,
        "confusable_nodes": confusable_nodes,
    }

    return json.dumps(
        scenario,
        ensure_ascii=False,
        indent=2,
    )


def build_retry_prompt(base_prompt: str, error: str) -> str:
    """Construct a refined retry prompt incorporating validation failure details.

    Appends the validation error message and formatting constraints to the base
    generation prompt to instruct the model to produce a valid conversation structure.

    Args:
        base_prompt: Original base generation prompt provided to the model.
        error: Validation error message or trace from the failed attempt.

    Returns:
        str: Augmented prompt string formatted for retry execution.
    """
    return (
        f"{base_prompt}\n\n"
        "=== PREVIOUS GENERATION FAILED VALIDATION ===\n"
        f"{error}\n\n"
        "Regenerate the entire conversation from scratch.\n"
        "Do not patch or minimally edit the previous generation.\n\n"
        "CRITICAL CORRECTION RULES:\n"
        "- Respect every target polarity exactly.\n"
        "- Respect every target assessment profile exactly.\n"
        "- Make severity explicitly observable.\n"
        "- Make frequency explicitly observable.\n"
        "- Make functional impact explicitly observable.\n"
        "- Make coping explicitly observable.\n"
        "- Do not infer any dimension from silence.\n"
        "- Do not infer functional=none from absence of impairment.\n"
        "- Do not infer coping=strong from absence of coping difficulty.\n"
        "- Do not infer frequency=rare from absence of temporal information.\n"
        "- Do not infer severity from polarity alone.\n"
        "- Do not use evidence belonging to one target to satisfy another target.\n"
        "- Do not introduce unrelated resilience concepts to express a profile.\n"
        "- For positive targets, do not use helplessness, surrender, severe impairment,\n"
        "  or inability to cope.\n"
        "- For negative targets, explicitly express the required difficulty or impairment.\n"
        "- For mixed targets, express both positive and negative aspects of the same node.\n"
        "- Preserve exactly one evidence marker for each target node.\n"
    )
