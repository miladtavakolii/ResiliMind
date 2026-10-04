from __future__ import annotations

from collections import Counter
import re
from typing import Any
import unicodedata

from evaluation.schemas import EvaluationCase
from .models import RenderedMessage, ScenarioRenderOutput

EVIDENCE_PATTERN = re.compile(
    r"\[\[EVIDENCE:(?P<node_id>[A-Za-z0-9_]+)\]\]"
    r"(?P<text>.*?)"
    r"\[\[/EVIDENCE:(?P=node_id)\]\]",
    re.DOTALL,
)

EVIDENCE_MARKER_PATTERN = re.compile(
    r"\[\[/?EVIDENCE:[A-Za-z0-9_]+\]\]"
)


def normalize_match_text(text: str) -> str:
    """Normalize text for deterministic lexical matching and cue auditing.

    Applies Unicode NFKC normalization, unifies Arabic/Persian character variants
    (Yeh, Kaf, Teh Marbuta, Heh Goal), strips diacritical marks (Tashkeel/Harakat),
    removes non-printing format characters (e.g., ZWNJ/ZWJ), and collapses whitespace
    into a single lowercase string.

    Args:
        text: Raw input string to normalize.

    Returns:
        str: Canonical, lowercased, and whitespace-collapsed text suitable
            for deterministic substring and keyword matching.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(
        str.maketrans(
            {
                "ي": "ی",
                "ى": "ی",
                "ئ": "ی",
                "ك": "ک",
                "ۀ": "ه",
                "ة": "ه",
            }
        )
    )
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = "".join(
        char for char in text if unicodedata.category(char) != "Cf"
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def validate_evidence_markup(
    case: EvaluationCase,
    output: ScenarioRenderOutput,
) -> None:
    """Validate that evidence markup is complete and structurally well-formed."""
    for message_index, message_obj in enumerate(output.messages):
        message = message_obj.content

        marker_count = len(
            EVIDENCE_MARKER_PATTERN.findall(message)
        )
        complete_marker_count = 2 * len(
            EVIDENCE_PATTERN.findall(message)
        )

        if marker_count != complete_marker_count:
            raise ValueError(
                f"{case.case_id}: malformed evidence markup in message "
                f"{message_index}"
            )


def clean_messages(messages: list[RenderedMessage]) -> list[str]:
    """Strip evidence annotation markup from rendered messages.

    Replaces evidence markup patterns with their underlying plain-text content
    to produce clean, readable user messages.

    Args:
        messages: List of RenderedMessage instances containing annotated text.

    Returns:
        list[str]: Cleaned message content strings without annotation tags.
    """
    cleaned = []

    for message in messages:
        text = EVIDENCE_PATTERN.sub(
            lambda m: m.group("text"),
            message.content,
        )
        cleaned.append(text)

    return cleaned


def extract_evidence(
    case: EvaluationCase,
    output: ScenarioRenderOutput,
) -> list[dict[str, Any]]:
    """Extract evidence spans from rendered scenario messages based on regex patterns.

    Args:
        case: Target EvaluationCase instance containing gold active signals.
        output: ScenarioRenderOutput containing rendered message objects.

    Returns:
        list[dict[str, Any]]: List of extracted evidence records containing
            node_id, message_index, character span offsets, and raw text.

    Raises:
        ValueError: If an extracted evidence node_id is not part of the expected
            gold active signals for the case.
    """
    expected = {
        signal.node_id
        for signal in case.gold.extraction.active_signals
    }

    extracted = []

    for message_index, message_obj in enumerate(output.messages):
        message = message_obj.content
        cursor = 0
        clean_length = 0

        for match in EVIDENCE_PATTERN.finditer(message):
            prefix = message[cursor:match.start()]
            clean_length += len(prefix)

            node_id = match.group("node_id")

            if node_id not in expected:
                raise ValueError(
                    f"{case.case_id}: unexpected evidence node {node_id}"
                )

            evidence_text = match.group("text")

            if not evidence_text:
                raise ValueError(
                    f"{case.case_id}: empty evidence for {node_id}"
                )

            if evidence_text != evidence_text.strip():
                raise ValueError(
                    f"{case.case_id}: evidence for {node_id} contains "
                    "leading/trailing whitespace"
                )

            start = clean_length
            clean_length += len(evidence_text)
            end = clean_length

            extracted.append(
                {
                    "node_id": node_id,
                    "message_index": message_index,
                    "start": start,
                    "end": end,
                    "evidence": evidence_text,
                }
            )

            cursor = match.end()

        if cursor < len(message):
            clean_length += len(message[cursor:])

    return extracted


def validate_rendered_output(
    case: EvaluationCase,
    messages: list[str],
    evidence: list[dict[str, Any]],
) -> None:
    """Validate rendered messages and extracted evidence spans against case expectations.

    Args:
        case: EvaluationCase instance containing scenario requirements and gold signals.
        messages: List of rendered user message strings.
        evidence: List of extracted evidence span dictionaries.

    Raises:
        ValueError: If message counts do not match expected turn counts, unknown signals
            are referenced, required signals lack evidence, duplicate evidence records
            are detected, message indices are out of bounds, or character offsets do not
            match the underlying message substrings.
    """
    expected_turns = case.scenario.turn_count

    if len(messages) != expected_turns:
        raise ValueError(
            f"{case.case_id}: expected {expected_turns} messages, got {len(messages)}"
        )

    for message_index, message in enumerate(messages):
        if not message.strip():
            raise ValueError(
                f"{case.case_id}: empty message at index {message_index}"
            )

        if (
            "[[EVIDENCE:" in message
            or "[[/EVIDENCE:" in message
        ):
            raise ValueError(
                f"{case.case_id}: malformed evidence marker in message "
                f"{message_index}"
            )

    signal_ids = {
        s.node_id
        for s in case.gold.extraction.active_signals
    }

    evidence_ids = {
        item["node_id"]
        for item in evidence
    }

    if unexpected := evidence_ids - signal_ids:
        raise ValueError(
            f"{case.case_id}: evidence returned for unknown signals: "
            f"{sorted(unexpected)}"
        )

    if missing := signal_ids - evidence_ids:
        raise ValueError(
            f"{case.case_id}: missing evidence for signals: "
            f"{sorted(missing)}"
        )

    evidence_counts = Counter(
        item["node_id"]
        for item in evidence
    )

    duplicates = [
        node_id
        for node_id, count in evidence_counts.items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"{case.case_id}: duplicate evidence markers for nodes: "
            f"{sorted(duplicates)}"
        )

    for item in evidence:
        message_index = item["message_index"]
        node_id = item["node_id"]
        start = item["start"]
        end = item["end"]

        if not 0 <= message_index < len(messages):
            raise ValueError(
                f"{case.case_id}: invalid message index "
                f"{message_index} for {node_id}"
            )

        if not 0 <= start <= end <= len(messages[message_index]):
            raise ValueError(
                f"{case.case_id}: invalid evidence bounds for {node_id}"
            )

        if not item["evidence"]:
            raise ValueError(
                f"{case.case_id}: empty evidence for {node_id}"
            )

        if item["evidence"] != item["evidence"].strip():
            raise ValueError(
                f"{case.case_id}: evidence for {node_id} contains "
                "leading/trailing whitespace"
            )

        message = messages[message_index]

        if message[start:end] != item["evidence"]:
            raise ValueError(
                f"{case.case_id}: invalid evidence span for {node_id}"
            )


def attach_evidence(
    case: EvaluationCase,
    evidence: list[dict[str, Any]],
) -> None:
    """Attach validated evidence spans to their corresponding gold signals.

    Args:
        case: The evaluation case whose signals will be updated in-place.
        evidence: List of verified evidence records containing offsets and text.

    Raises:
        ValueError: If a required gold signal is missing associated evidence.
    """
    evidence_by_node: dict[str, dict[str, Any]] = {}

    for item in evidence:
        node_id = item["node_id"]

        if node_id in evidence_by_node:
            raise ValueError(
                f"{case.case_id}: duplicate evidence for {node_id}"
            )

        evidence_by_node[node_id] = item

    for signal in case.gold.extraction.active_signals:
        item = evidence_by_node.get(signal.node_id)

        if item is None:
            raise ValueError(
                f"{case.case_id}: missing evidence for {signal.node_id}"
            )

        signal.evidence = item["evidence"]
        signal.evidence_message_index = item["message_index"]
        signal.evidence_start = item["start"]
        signal.evidence_end = item["end"]
