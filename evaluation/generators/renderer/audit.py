from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from evaluation.schemas import EvaluationCase
from src.resilimind.llm.providers.base import BaseLLMProvider
from .evidence import normalize_match_text
from .models import ScenarioRenderAudit
from .prompts import build_assessment_requirements

logger = logging.getLogger(__name__)


def validate_ambiguous_output(
    case: EvaluationCase,
    messages: list[str],
    nodes: dict[str, dict[str, Any]],
) -> None:
    """Reject ambiguous scenario renders containing explicit graph cue phrases.

    For cases categorized under the 'ambiguous' case type, scans the rendered
    messages against semantic polarity keywords across all knowledge graph nodes.
    If explicit cues appear, rejects the output to prevent unambiguous signal leakage.

    Args:
        case: Target EvaluationCase instance containing scenario metadata.
        messages: List of rendered dialogue messages to scan.
        nodes: Mapping of graph node IDs to their attributes.

    Raises:
        ValueError: If an ambiguous scenario contains explicit node-specific
            semantic cue keywords.
    """
    if case.scenario.case_type != "ambiguous":
        return

    normalized_message = normalize_match_text(" ".join(messages))
    hits: list[str] = []

    for node_id, node in nodes.items():
        cues = node.get("cues", {})

        for polarity in ("positive_keywords", "negative_keywords"):
            for cue in cues.get(polarity, []):
                normalized_cue = normalize_match_text(cue)

                if len(normalized_cue) < 4:
                    continue

                if normalized_cue in normalized_message:
                    hits.append(f"{node_id}:{cue}")

    if hits:
        raise ValueError(
            f"{case.case_id}: ambiguous render contains "
            f"node-specific cue(s): {hits[:10]}"
        )


def audit_rendered_output(
    client: Any = None,
    model_name: str = "",
    audit_prompt: str = "",
    case: EvaluationCase | None = None,
    messages: list[str] | None = None,
    nodes: dict[str, dict[str, Any]] | None = None,
    *,
    provider: BaseLLMProvider | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    request_delay: float = 1.0,
) -> ScenarioRenderAudit:
    """Perform a semantic audit on rendered scenario messages.

    Evaluates rendered text against target graph signals and non-target candidate
    nodes to verify that intended resilience concepts are present and unintended
    extraneous signals were not inadvertently leaked into the conversation.

    Args:
        client: Optional Google GenAI Client instance (for backward compatibility).
        model_name: Name of the model.
        audit_prompt: The loaded semantic audit prompt string.
        case: Target EvaluationCase containing expected gold signals.
        messages: Cleaned dialogue messages generated for the scenario.
        nodes: Mapping of graph node IDs to their attributes.
        provider: Optional BaseLLMProvider instance.
        max_retries: Max retry attempts.
        retry_delay: Base delay between retries.
        request_delay: Politeness delay before audit request.

    Returns:
        ScenarioRenderAudit: Parsed validation model assessing signal alignment
            and flagging unintended graph nodes.

    Raises:
        ValueError: If empty response or validation fails.
    """
    if case is None or messages is None or nodes is None:
        raise ValueError("case, messages, and nodes are required for semantic audit.")

    target_ids = {
        signal.node_id
        for signal in case.gold.extraction.active_signals
    }

    unknown_targets = sorted(target_ids - set(nodes))
    if unknown_targets:
        raise ValueError(
            f"{case.case_id}: audit references unknown graph nodes: "
            f"{unknown_targets}"
        )

    target_nodes = []

    for signal in case.gold.extraction.active_signals:
        profile = case.scenario.assessment_profiles.get(signal.node_id)

        if profile is None:
            raise ValueError(
                f"{case.case_id}: missing assessment profile for {signal.node_id}"
            )
        
        node = nodes[signal.node_id]
        semantic_boundary = node.get("semantic_boundary", "")
        target_nodes.append(
            {
                "node_id": signal.node_id,
                "polarity": signal.detected_signal,
                "name_fa": node.get("name_fa", ""),
                "description": node.get("description", ""),
                "semantic_boundary": semantic_boundary,
                "assessment_profile": {
                    "severity": profile.severity,
                    "frequency": profile.frequency,
                    "functional": profile.functional,
                    "coping": profile.coping,
                },
                "assessment_requirements": build_assessment_requirements(
                    signal.node_id, profile, semantic_boundary
                ),
            }
        )

    candidates = []

    for node_id, node in nodes.items():
        if node_id in target_ids:
            continue

        candidates.append(
            {
                "node_id": node_id,
                "name_fa": node.get("name_fa", ""),
                "description": node.get("description", ""),
                "semantic_boundary": node.get("semantic_boundary", ""),
            }
        )

    audit_input = {
        "case_type": case.scenario.case_type,
        "target_signals": target_nodes,
        "candidate_unintended_nodes": candidates,
        "messages": messages,
    }

    active_provider = provider or (client if isinstance(client, BaseLLMProvider) else None)
    if active_provider is None:
        raise ValueError("A BaseLLMProvider instance is required for semantic audit.")

    prompt_tmpl = ChatPromptTemplate.from_messages([
        SystemMessage(content=audit_prompt),
        ("human", "{audit_input}"),
    ])
    audit_runner = prompt_tmpl | active_provider.get_structured_model(
        schema=ScenarioRenderAudit,
        temperature=0.0,
    )

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if attempt == 0 and request_delay > 0:
                logger.info(
                    "%s: sleeping %.1f seconds before semantic audit...",
                    case.case_id,
                    request_delay,
                )
                time.sleep(request_delay)

            raw_response = audit_runner.invoke({
                "audit_input": json.dumps(
                    audit_input,
                    ensure_ascii=False,
                    indent=2,
                )
            })

            if isinstance(raw_response, dict):
                if raw_response.get("parsed") is not None:
                    return raw_response["parsed"]
                elif "raw" in raw_response:
                    raise ValueError(
                        f"{case.case_id}: semantic audit returned invalid output: {raw_response.get('raw')!r}"
                    )
                else:
                    return ScenarioRenderAudit.model_validate(raw_response)
            elif isinstance(raw_response, ScenarioRenderAudit):
                return raw_response
            else:
                raise ValueError(
                    f"{case.case_id}: unexpected semantic audit response type: {type(raw_response)}"
                )

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            delay = min(retry_delay * (attempt + 1), 20.0)

            logger.warning(
                "%s: semantic audit failed on attempt %d/%d: %s. "
                "Retrying in %.1f seconds...",
                case.case_id,
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )

            time.sleep(delay)

    raise RuntimeError(
        f"{case.case_id}: semantic audit failed after "
        f"{max_retries + 1} attempts"
    ) from last_error
