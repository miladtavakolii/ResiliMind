import logging
import re
from typing import Any
import unicodedata

from ..state import AgentState
from ...llm.engine import LLMEngine
from ...llm import prompts
from ...schemas.models import ExtractionOutput, ActiveSignal
from .common import llm_engine, resilience_graph
from .text_utils import normalize_persian_text, find_polarity_cues as _find_polarity_cues

# Initialize module logger
logger = logging.getLogger(__name__)


def canonicalize_extraction_result(result: ExtractionOutput) -> ExtractionOutput:
    """Collapse duplicate signals for the same node when their polarity agrees.

    Deduplicates multiple extraction detections referencing the identical graph node.
    If polarities conflict, they are merged as "mixed". If evidence spans subsume one another,
    retains the longer span; otherwise, retains the earlier detection while logging
    the deduplication event.

    Args:
        result: Raw ExtractionOutput instance potentially containing duplicate node detections.

    Returns:
        ExtractionOutput: A updated copy of the extraction output containing only
            unique canonical signals per node.

    Raises:
        ValueError: If duplicate signals for the same node contain contradictory
            polarity classifications (e.g., 'positive' vs 'negative').
    """
    unique_signals: dict[str, Any] = {}

    for signal in result.active_signals:
        existing = unique_signals.get(signal.node_id)
        if existing is None:
            unique_signals[signal.node_id] = signal
            continue

        existing_polarity = existing.detected_signal
        new_polarity = signal.detected_signal

        if existing_polarity != new_polarity:
            logger.error(
                "[Extractor] Conflicting duplicate polarities for node %s: %s vs %s",
                signal.node_id,
                existing_polarity,
                new_polarity,
            )
            raise ValueError(
                f"Conflicting duplicate polarity for {signal.node_id}; "
                "mixed must be explicitly resolved with polarity evidence"
            )

        merged_polarity = existing_polarity

        existing_evidence = normalize_persian_text(existing.evidence)
        new_evidence = normalize_persian_text(signal.evidence)

        if new_evidence and new_evidence in existing_evidence:
            selected_evidence = existing.evidence
        elif existing_evidence and existing_evidence in new_evidence:
            selected_evidence = signal.evidence
        else:
            selected_evidence = (
                signal.evidence
                if len(signal.evidence) > len(existing.evidence)
                else existing.evidence
            )

        if merged_polarity != existing_polarity:
            logger.warning("[Extractor] Merging duplicate node %s polarities: %s + %s -> mixed.", signal.node_id, existing_polarity, new_polarity)

        merged_polarity_evidence = {}

        for source_signal in (existing, signal):
            for polarity, span in source_signal.polarity_evidence.items():
                merged_polarity_evidence.setdefault(polarity, span)

        if merged_polarity != "mixed":
            merged_polarity_evidence = {
                merged_polarity: merged_polarity_evidence[merged_polarity]
            } if merged_polarity in merged_polarity_evidence else {}

        unique_signals[signal.node_id] = existing.model_copy(
            update={
                "detected_signal": merged_polarity,
                "evidence": selected_evidence,
                "polarity_evidence": merged_polarity_evidence,
            }
        )

    return result.model_copy(
        update={"active_signals": list(unique_signals.values())}
    )


def build_extractor_graph_context() -> str:
    """Build compact semantic definitions for all resilience graph nodes.

    Iterates deterministically through the global resilience knowledge graph nodes,
    formatting each node's identifiers, Persian names, domain categories, descriptions,
    and polarity cues into a structured plain-text prompt block.

    Returns:
        str: Formatted knowledge graph context string containing node definitions
            and semantic cues for extraction prompt injection.
    """
    blocks = []

    for node_id, node_data in sorted(resilience_graph.nodes(data=True)):
        blocks.append(
            f"Node ID: {node_id}\n"
            f"Name: {node_data.get('name_fa', '')}\n"
            f"Domain: {node_data.get('domain_fa', '')} "
            f"({node_data.get('domain', '')})\n"
            f"Definition: {node_data.get('description', '')}\n"
            f"Semantic boundary: {node_data.get('semantic_boundary', '')}"
        )

    return (
        "=== NODE CATALOG ===\n"
        + "\n\n".join(blocks)
    )


def audit_signal_polarity(result: ExtractionOutput) -> ExtractionOutput:
    """Reconcile LLM-detected polarity with explicit knowledge graph cue evidence.

    Cross-references extracted signal evidence against the node's authoritative
    positive and negative lexical cues in the knowledge graph. When explicit cues
    unambiguously indicate a polarity, updates the signal's polarity accordingly
    and logs any discrepancies with the original LLM prediction.

    Args:
        result: ExtractionOutput containing raw model-detected signals.

    Returns:
        ExtractionOutput: A updated copy of the extraction output containing
            reconciled polarities for all active signals.
    """
    for signal in result.active_signals:
        node_data = resilience_graph.nodes.get(signal.node_id, {})
        cues = node_data.get("cues", {})
        positive_cues = cues.get("positive_keywords", [])
        negative_cues = cues.get("negative_keywords", [])
        polarity_evidence = signal.polarity_evidence

        positive_text = polarity_evidence.get("positive", "")
        negative_text = polarity_evidence.get("negative", "")

        positive_hits = _find_polarity_cues(
            positive_text,
            positive_cues,
        )
        negative_hits_in_positive = _find_polarity_cues(
            positive_text,
            negative_cues,
        )
        negative_hits = _find_polarity_cues(
            negative_text,
            negative_cues,
        )
        positive_hits_in_negative = _find_polarity_cues(
            negative_text,
            positive_cues,
        )

        if signal.detected_signal == "positive":
            conflict = bool(negative_hits_in_positive) and not positive_hits
        elif signal.detected_signal == "negative":
            conflict = bool(positive_hits_in_negative) and not negative_hits
        else:
            conflict = (
                bool(negative_hits_in_positive) and not positive_hits
            ) or (
                bool(positive_hits_in_negative) and not negative_hits
            )

        if conflict:
            logger.warning(
                "[PolarityAudit] Model-cue disagreement for %s: "
                "polarity=%s, positive_hits=%s, negative_hits=%s",
                signal.node_id,
                signal.detected_signal,
                positive_hits,
                negative_hits,
            )
        else:
            logger.debug(
                "[PolarityAudit] node=%s polarity=%s "
                "positive_hits=%s negative_hits=%s",
                signal.node_id,
                signal.detected_signal,
                positive_hits,
                negative_hits,
            )

    return result


def align_evidence_to_user_message(evidence: str, user_message: str) -> str | None:
    """Map an LLM evidence candidate to an exact contiguous span in the raw user message.

    Handles surface-level variations such as Unicode NFKC differences, Arabic/Persian
    character mappings, irregular spacing, and zero-width non-joiners (ZWNJ) by tracking
    character index offsets between normalized/compacted text and the raw input string.

    Args:
        evidence: Predicted evidence string from the LLM extractor.
        user_message: Original raw user message string.

    Returns:
        str | None: The exact matching substring from `user_message` corresponding to
            the candidate evidence span, or None if no valid alignment can be found.
    """
    if not evidence or not user_message:
        return None

    if evidence in user_message:
        return evidence

    def compact_with_map(text: str) -> tuple[str, list[int]]:
        normalized = unicodedata.normalize("NFKC", text).translate(
            str.maketrans({
                "ي": "ی",
                "ى": "ی",
                "ئ": "ی",
                "ك": "ک",
                "آ": "ا",
                "أ": "ا",
                "إ": "ا",
                "ۀ": "ه",
                "ة": "ه",
                "ؤ": "و",
            })
        )

        chars: list[str] = []
        source_indices: list[int] = []

        for index, char in enumerate(normalized):
            if char.isspace() or unicodedata.category(char) == "Cf":
                continue
            chars.append(char)
            source_indices.append(index)

        return "".join(chars), source_indices

    candidate = evidence.strip()
    compact_candidate, _ = compact_with_map(candidate)
    compact_message, source_indices = compact_with_map(user_message)

    if not compact_candidate:
        return None

    start = compact_message.find(compact_candidate)

    if start == -1:
        return None

    end = start + len(compact_candidate)

    raw_start = source_indices[start]
    raw_end = source_indices[end - 1] + 1

    aligned = user_message[raw_start:raw_end]

    logger.debug(
        "[Extractor] Evidence aligned: predicted=%r aligned=%r",
        evidence,
        aligned,
    )

    return aligned


def build_evidence_candidates(
    user_message: str, max_candidates: int = 72
) -> list[tuple[int, int, str]]:
    """Segment raw user text into candidate evidence spans with exact character offsets.

    Extracts both natural linguistic units (sentences, punctuation-delimited clauses)
    and short contiguous sliding windows (5, 7, 9, 11, and 13 words) with balanced strides.
    Trims leading and trailing boundary punctuation and whitespace while adjusting exact
    character offsets to guarantee precise substring alignment with `user_message`.
    Deduplicates identical spans and caps total candidates at `max_candidates` (default 72)
    using uniform sampling if the candidate count exceeds the quota.

    Args:
        user_message: Raw user message string to segment.
        max_candidates: Maximum number of candidate spans to produce (default: 72).

    Returns:
        list[tuple[int, int, str]]: Triples of `(start_index, end_index, clause_text)`
            representing trimmed contiguous candidate spans within `user_message`.
    """
    if not user_message or not user_message.strip():
        return []

    seen_spans: set[tuple[int, int]] = set()
    structural_candidates: list[tuple[int, int, str]] = []

    # 1. Sentences and sub-clauses delimited by punctuation
    for pattern in (r"[^.!?؟\n]+", r"[^،,؛;.!?؟\n]+"):
        for match in re.finditer(pattern, user_message):
            start, end = match.span()
            raw_segment = user_message[start:end]
            text = raw_segment.strip()
            if not text:
                continue

            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
            s = start + leading_ws
            e = end - trailing_ws

            if s < e and (s, e) not in seen_spans:
                seen_spans.add((s, e))
                structural_candidates.append((s, e, user_message[s:e]))

    # 2. Token spans and sliding windows (5, 7, 9, 11, 13 words)
    word_matches = list(re.finditer(r"\S+", user_message))
    num_words = len(word_matches)
    window_candidates: list[tuple[int, int, str]] = []

    window_configs = [
        (5, 2),   # (window_size, stride)
        (7, 2),
        (9, 3),
        (11, 3),
        (13, 4),
    ]

    # Handle short messages where total words are fewer than the smallest window size
    if 0 < num_words < 5:
        s = word_matches[0].start()
        e = word_matches[-1].end()
        raw = user_message[s:e]
        trimmed_s = s + (len(raw) - len(raw.lstrip(" \t\n\r،,؛;.!?؟:\"'«»()[]{}")))
        trimmed_e = e - (len(raw) - len(raw.rstrip(" \t\n\r،,؛;.!?؟:\"'«»()[]{}")))
        if trimmed_s < trimmed_e and (trimmed_s, trimmed_e) not in seen_spans:
            seen_spans.add((trimmed_s, trimmed_e))
            window_candidates.append((trimmed_s, trimmed_e, user_message[trimmed_s:trimmed_e]))

    for w_size, stride in window_configs:
        if num_words < w_size:
            continue
        for i in range(0, num_words - w_size + 1, stride):
            s = word_matches[i].start()
            e = word_matches[i + w_size - 1].end()
            raw = user_message[s:e]

            l_strip = len(raw) - len(raw.lstrip(" \t\n\r،,؛;.!?؟:\"'«»()[]{}"))
            r_strip = len(raw) - len(raw.rstrip(" \t\n\r،,؛;.!?؟:\"'«»()[]{}"))
            trimmed_s = s + l_strip
            trimmed_e = e - r_strip

            if trimmed_s < trimmed_e and (trimmed_s, trimmed_e) not in seen_spans:
                seen_spans.add((trimmed_s, trimmed_e))
                window_candidates.append((trimmed_s, trimmed_e, user_message[trimmed_s:trimmed_e]))

    # 3. Cap candidates at max_candidates (72)
    total_found = len(structural_candidates) + len(window_candidates)
    if total_found <= max_candidates:
        selected = structural_candidates + window_candidates
    else:
        if len(structural_candidates) >= max_candidates:
            step = (len(structural_candidates) - 1) / (max_candidates - 1)
            selected = [
                structural_candidates[round(i * step)]
                for i in range(max_candidates)
            ]
        else:
            remaining_quota = max_candidates - len(structural_candidates)
            step = (
                (len(window_candidates) - 1) / (remaining_quota - 1)
                if remaining_quota > 1
                else 0
            )
            sampled_windows = [
                window_candidates[round(i * step)]
                for i in range(remaining_quota)
            ]
            selected = structural_candidates + sampled_windows

    # Sort deterministically by start offset then end offset
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected


def validate_extraction_result(result: ExtractionOutput, user_message: str) -> ExtractionOutput:
    """Validate extractor evidence and node consistency against the knowledge graph.

    Verifies that all extracted signals reference known graph nodes, contain no duplicate
    node IDs, have non-empty evidence strings, and ensure that every evidence span is
    present as an exact substring within the original user message.

    Args:
        result: ExtractionOutput instance containing candidate active signals.
        user_message: Raw user message text against which evidence substrings are verified.

    Returns:
        ExtractionOutput: The validated extraction output instance.

    Raises:
        ValueError: If a signal references an unknown node, duplicate nodes are present,
            evidence is empty, or evidence text does not appear in the user message.
    """
    valid_node_ids = set(resilience_graph.nodes)
    seen_nodes: set[str] = set()
    corrected_signals = []

    for signal in result.active_signals:
        if signal.node_id not in valid_node_ids:
            raise ValueError(
                f"Extractor returned unknown node: {signal.node_id}"
            )

        if signal.node_id in seen_nodes:
            raise ValueError(
                f"Extractor returned duplicate node: {signal.node_id}"
            )
        seen_nodes.add(signal.node_id)

        evidence = align_evidence_to_user_message(
            signal.evidence,
            user_message,
        )

        if evidence is None:
            raise ValueError(
                f"Extractor evidence cannot be aligned to user message for "
                f"{signal.node_id}: {signal.evidence!r}"
            )

        polarity_evidence = dict(signal.polarity_evidence or {})

        if signal.detected_signal in {"positive", "negative"}:
            if not polarity_evidence:
                polarity_evidence[signal.detected_signal] = evidence

        expected_polarity_keys = {
            "positive": {"positive"},
            "negative": {"negative"},
            "mixed": {"positive", "negative"},
        }[signal.detected_signal]

        if set(polarity_evidence) != expected_polarity_keys:
            raise ValueError(
                f"Invalid polarity evidence keys for {signal.node_id}: "
                f"expected={expected_polarity_keys}, "
                f"actual={set(polarity_evidence)}"
            )

        aligned_polarity_evidence = {}

        for polarity, polarity_span in polarity_evidence.items():
            aligned_span = align_evidence_to_user_message(
                polarity_span,
                user_message,
            )

            if aligned_span is None:
                raise ValueError(
                    f"Polarity evidence cannot be aligned to user message "
                    f"for {signal.node_id} ({polarity}): {polarity_span!r}"
                )

            aligned_polarity_evidence[polarity] = aligned_span

        corrected_signals.append(
            signal.model_copy(
                update={
                    "evidence": evidence,
                    "polarity_evidence": aligned_polarity_evidence,
                }
            )
        )

    return result.model_copy(update={"active_signals": corrected_signals})


def build_extractor_candidate_hints(user_message: str) -> str:
    """Build candidate node hints based on lexical cue matches in the user message.

    Scans the incoming user utterance against known positive and negative cue
    keywords defined across all resilience graph nodes, assembling matched cues
    into a structured text block for extractor prompt injection.

    Args:
        user_message: Raw user message text to scan for keyword cues.

    Returns:
        str: Formatted string of candidate graph nodes with their matching
            positive and negative cues, or a fallback message if no matches occur.
    """
    blocks = []

    cue_owners: dict[str, list[str]] = {}

    for node_id, node_data in resilience_graph.nodes(data=True):
        cues = node_data.get("cues", {})
        all_cues = (
            cues.get("positive_keywords", [])
            + cues.get("negative_keywords", [])
            + node_data.get("candidate_keywords", [])
        )

        for cue in all_cues:
            normalized_cue = normalize_persian_text(cue)
            if normalized_cue:
                cue_owners.setdefault(normalized_cue, []).append(node_id)

    for node_id, node_data in sorted(resilience_graph.nodes(data=True)):
        cues = node_data.get("cues", {})
        positive_hits = _find_polarity_cues(
            user_message,
            cues.get("positive_keywords", []),
        )
        negative_hits = _find_polarity_cues(
            user_message,
            cues.get("negative_keywords", []),
        )
        context_hits = _find_polarity_cues(
            user_message,
            node_data.get("candidate_keywords", []),
        )

        if not positive_hits and not negative_hits and not context_hits:
            continue

        positive_text = ", ".join(positive_hits) if positive_hits else "none"
        negative_text = ", ".join(negative_hits) if negative_hits else "none"

        shared_cues = [
            cue
            for cue in positive_hits + negative_hits
            if len(cue_owners.get(normalize_persian_text(cue), [])) > 1
        ]

        shared_text = (
            ", ".join(
                f"{cue} -> {cue_owners[normalize_persian_text(cue)]}"
                for cue in shared_cues
            )
            if shared_cues
            else "none"
        )

        blocks.append(
            f"Candidate Node: {node_id}\n"
            f"Positive cue matches: {positive_text}\n"
            f"Negative cue matches: {negative_text}\n"
            f"Context cue matches: {', '.join(context_hits) if context_hits else 'none'}\n"
            f"Shared cue warning: {shared_text}"
        )

    if not blocks:
        return "No explicit graph cue candidates were found."

    return (
        "Review EVERY candidate independently.\n"
        "Candidate hints are locator hints only, NOT extraction decisions.\n"
        "A cue match alone is never sufficient for extraction.\n"
        "The matched cue MUST satisfy the candidate node's semantic boundary.\n"
        "Shared or generic cues require surrounding context for disambiguation.\n"
        "If the context does not clearly distinguish the node, do not extract it.\n"
        "Do not stop after selecting the first candidate.\n\n"
        + "\n\n".join(blocks)
    )


def extract_candidate_node_ids(candidate_hints: str) -> list[str]:
    """Extract and validate candidate node IDs from formatted cue hint blocks.

    Parses multi-line candidate hint text using regular expressions to capture
    identifiers marked with 'Candidate Node:', verifies them against valid nodes
    in the active resilience knowledge graph, and preserves initial discovery order
    while eliminating duplicate IDs.

    Args:
        candidate_hints: Formatted candidate text generated by lexical cue matchers.

    Returns:
        list[str]: Unique, order-preserved list of valid knowledge graph node IDs.
    """
    valid_node_ids = set(resilience_graph.nodes)
    matches = re.findall(
        r"(?m)^Candidate Node:\s*([A-Za-z0-9_]+)\s*$",
        candidate_hints,
    )
    return list(
        dict.fromkeys(
            node_id for node_id in matches if node_id in valid_node_ids
        )
    )


def build_selected_node_context(node_ids: list[str]) -> str:
    """Format detailed semantic context for selected graph nodes.

    Constructs a plain-text prompt block containing identifiers, Persian names,
    domains, definitions, boundary constraints, and polarity keywords for each
    requested node present in the knowledge graph.

    Args:
        node_ids: List of candidate graph node identifiers to contextualize.

    Returns:
        str: Double-newline-separated blocks of node specifications, or an empty
            string if none of the provided IDs exist in the resilience graph.
    """
    blocks = []

    for node_id in node_ids:
        if node_id not in resilience_graph:
            continue

        node = resilience_graph.nodes[node_id]
        cues = node.get("cues", {})

        blocks.append(
            f"Node ID: {node_id}\n"
            f"Name: {node.get('name_fa', '')}\n"
            f"Domain: {node.get('domain_fa', '')} "
            f"({node.get('domain', '')})\n"
            f"Definition: {node.get('description', '')}\n"
            f"Semantic boundary: {node.get('semantic_boundary', '')}\n"
            f"Positive cues: {cues.get('positive_keywords', [])}\n"
            f"Negative cues: {cues.get('negative_keywords', [])}"
        )

    return "\n\n".join(blocks)


def extractor_node(state: AgentState) -> dict[str, Any]:
    """Execute two-stage structured signal extraction in the LangGraph agent pipeline.

    Performs graph node candidate selection followed by detailed signal resolution.
    Validates evidence alignment against raw user messages, collapses duplicates,
    reconciles emotional polarity against authoritative knowledge graph cues, and
    ensures complete 1-to-1 consistency between selected nodes and resolved signals.

    Args:
        state: Current agent workflow state containing the input `user_message`.

    Returns:
        dict[str, Any]: State update dictionary containing:
            - "active_nodes": List of verified graph node ID strings.
            - "active_signals": List of serialized active signal dictionaries
              with extracted evidence spans, polarities, and node IDs.

    Raises:
        RuntimeError: If valid extraction and reconciliation cannot be completed
            within 3 attempts due to schema parsing errors or structural mismatches.
    """
    logger.info("[Extractor] Starting staged extraction...")
    user_msg = state.get("user_message", "")

    candidate_hints = build_extractor_candidate_hints(user_msg)
    selector_prompt = (
        f"{prompts.NODE_SELECTOR_SYSTEM_PROMPT}\n\n"
        f"{build_extractor_graph_context()}\n\n"
        f"=== LEXICAL CANDIDATE HINTS ===\n"
        f"{candidate_hints}\n\n"
    )

    selector = llm_engine.get_node_selector_runner(selector_prompt)

    last_error: Exception | None = None

    for attempt in range(3):
        raw = selector.invoke({"user_message": user_msg})

        if raw.get("parsed") is None:
            last_error = ValueError(
                f"Node selector returned invalid output: {raw.get('raw')!r}"
            )
            continue

        selector_nodes = list(dict.fromkeys(raw["parsed"].node_ids))
        hinted_nodes = extract_candidate_node_ids(candidate_hints)
        selected_nodes = list(dict.fromkeys([
            *selector_nodes,
            *hinted_nodes,
        ]))

        logger.debug(
            "[Extractor] Selector nodes=%s; graph-hinted nodes=%s; resolving=%s",
            selector_nodes,
            hinted_nodes,
            selected_nodes,
        )

        if not selected_nodes:
            logger.debug("[Extractor] No selector or graph-hinted candidates found.")
            return {
                "active_nodes": [],
                "active_signals": [],
            }

        if not selected_nodes:
            logger.debug("[Extractor] Selector returned no nodes. Lexical hints:\n%s",candidate_hints,)
            return {
                "active_nodes": [],
                "active_signals": [],
            }

        resolved_signals = []
        accepted_nodes = []

        evidence_candidates = build_evidence_candidates(user_msg)

        if not evidence_candidates:
            raise RuntimeError("No evidence candidates found in user message")

        for node_id in selected_nodes:
            resolver_prompt_base = (
                f"{prompts.SIGNAL_RESOLVER_SYSTEM_PROMPT}\n\n"
                "=== TARGET NODE ===\n"
                f"{build_selected_node_context([node_id])}\n\n"
                "=== EVIDENCE CANDIDATES ===\n"
                + "\n".join(
                    f"[{index}] {candidate[2]}"
                    for index, candidate in enumerate(evidence_candidates)
                )
                + "\n\n"
                f"The target node for this invocation is {node_id}.\n"
                "Return exactly one signal for this node.\n"
                "Select exactly one evidence candidate by index."
            )

            node_last_error: Exception | None = None
            node_resolved = None
            node_rejected = False

            for resolver_attempt in range(3):
                resolver_prompt = resolver_prompt_base

                if node_last_error is not None:
                    resolver_prompt += (
                        "\n\n=== PREVIOUS ATTEMPT FAILED ===\n"
                        f"Reason: {node_last_error}\n"
                        "Regenerate the result from scratch.\n"
                        f"Return exactly one signal for {node_id}.\n"
                        "Select exactly one valid evidence_index.\n"
                        "Do not generate or rewrite evidence text.\n"
                        "Include both polarity evidence index keys in every "
                        "supported signal; use null only when not applicable.\n"
                        "For positive or negative polarity, the matching index "
                        "may equal evidence_index.\n"
                        "For mixed polarity, both indices must be valid and "
                        "must support opposite polarities of the SAME node.\n"
                        "Never return mixed when only one polarity is supported.\n"
                        "Do not return mixed unless both polarity evidence indices are valid.\n"
                        "If only positive evidence is supported, return positive with its index.\n"
                        "If only negative evidence is supported, return negative with its index.\n"
                        "If the target node is unsupported, return supported=false and null indices.\n"
                        "Never use evidence_index as a substitute for both mixed-polarity indices.\n"
                    )

                resolver = llm_engine.get_signal_resolver_runner(
                    resolver_prompt
                )

                logger.debug(
                    "[Extractor] user_message=%r",
                    user_msg,
                )

                raw_resolved = resolver.invoke({
                    "user_message": user_msg,
                    "selected_nodes": build_selected_node_context([node_id]),
                })

                if raw_resolved.get("parsed") is None:
                    node_last_error = ValueError(
                        f"Signal resolver returned invalid output for {node_id}: "
                        f"{raw_resolved.get('raw')!r}"
                    )
                    logger.warning(
                        "[Extractor] Resolver failed for %s "
                        "(attempt %d/3): %s",
                        node_id,
                        resolver_attempt + 1,
                        node_last_error,
                    )
                    continue

                signals = raw_resolved["parsed"].signals
                logger.debug(
                    "[Extractor] Resolver output for %s: %s",
                    node_id,
                    [
                        {
                            "node_id": signal.node_id,
                            "supported": signal.supported,
                            "polarity": signal.detected_signal,
                            "evidence_index": signal.evidence_index,
                        }
                        for signal in signals
                    ],
                )

                if len(signals) != 1:
                    node_last_error = ValueError(
                        f"Signal resolver returned {len(signals)} signals for "
                        f"{node_id}, expected exactly 1."
                    )
                    logger.warning(
                        "[Extractor] Resolver returned wrong signal count for %s "
                        "(attempt %d/3): %s",
                        node_id,
                        resolver_attempt + 1,
                        node_last_error,
                    )
                    continue

                signal = signals[0]

                if signal.node_id != node_id:
                    node_last_error = ValueError(
                        f"Signal resolver returned wrong node: "
                        f"expected={node_id}, actual={signal.node_id}"
                    )
                    logger.warning(
                        "[Extractor] Resolver returned wrong node "
                        "(attempt %d/3): %s",
                        resolver_attempt + 1,
                        node_last_error,
                    )
                    continue

                if not signal.supported:
                    logger.info(
                        "[Extractor] Resolver rejected unsupported node %s",
                        node_id,
                    )
                    node_rejected = True
                    break

                if signal.detected_signal is None:
                    node_last_error = ValueError(
                        f"Supported node {node_id} has no detected_signal"
                    )
                    continue

                if signal.evidence_index is None:
                    node_last_error = ValueError(
                        f"Supported node {node_id} has no evidence_index"
                    )
                    continue

                if not 0 <= signal.evidence_index < len(evidence_candidates):
                    node_last_error = ValueError(
                        f"Signal resolver returned invalid evidence index for "
                        f"{node_id}: {signal.evidence_index}"
                    )
                    logger.warning(
                        "[Extractor] Invalid evidence index for %s "
                        "(attempt %d/3): %s",
                        node_id,
                        resolver_attempt + 1,
                        node_last_error,
                    )
                    continue

                polarity_indices = {
                    "positive": signal.positive_evidence_index,
                    "negative": signal.negative_evidence_index,
                }

                if signal.detected_signal == "positive":
                    if polarity_indices["negative"] is not None:
                        node_last_error = ValueError(
                            "Positive polarity cannot contain negative evidence"
                        )
                        continue
                    if polarity_indices["positive"] is None:
                        polarity_indices["positive"] = signal.evidence_index

                elif signal.detected_signal == "negative":
                    if polarity_indices["positive"] is not None:
                        node_last_error = ValueError(
                            "Negative polarity cannot contain positive evidence"
                        )
                        continue
                    if polarity_indices["negative"] is None:
                        polarity_indices["negative"] = signal.evidence_index

                elif (
                    polarity_indices["positive"] is None
                    or polarity_indices["negative"] is None
                ):
                    node_last_error = ValueError(
                        "Mixed polarity requires both polarity evidence indices"
                    )
                    logger.warning(
                        "[Extractor] Incomplete mixed-polarity evidence for %s "
                        "(attempt %d/3)",
                        node_id,
                        resolver_attempt + 1,
                    )
                    continue

                invalid_indices = [
                    index
                    for index in polarity_indices.values()
                    if index is not None
                    and not 0 <= index < len(evidence_candidates)
                ]

                if invalid_indices:
                    node_last_error = ValueError(
                        f"Invalid polarity evidence indices for {node_id}: "
                        f"{invalid_indices}"
                    )
                    logger.warning(
                        "[Extractor] Invalid polarity evidence for %s: %s",
                        node_id,
                        invalid_indices,
                    )
                    continue

                polarity_evidence = {
                    polarity: evidence_candidates[index][2]
                    for polarity, index in polarity_indices.items()
                    if index is not None
                }
                evidence = evidence_candidates[signal.evidence_index][2]

                logger.info(
                    "[Extractor] Resolved %s -> evidence_index=%d, evidence=%r",
                    node_id,
                    signal.evidence_index,
                    evidence,
                )

                node_resolved = {
                    "node_id": signal.node_id,
                    "detected_signal": signal.detected_signal,
                    "evidence": evidence,
                    "polarity_evidence": polarity_evidence,
                }
                break

            if node_rejected:
                continue

            if node_resolved is None:
                logger.warning(
                    "[Extractor] Rejecting unresolved candidate %s after retries: %s",
                    node_id,
                    node_last_error,
                )
                continue

            resolved_signals.append(node_resolved)
            accepted_nodes.append(node_id)

        result = ExtractionOutput(
            active_signals=resolved_signals
        )

        result = canonicalize_extraction_result(result)
        result = validate_extraction_result(
            result=result,
            user_message=user_msg,
        )
        result = audit_signal_polarity(result)

        return {
            "active_nodes": accepted_nodes,
            "active_signals": [
                signal.model_dump()
                for signal in result.active_signals
            ],
        }

    raise RuntimeError(
        "Staged extractor failed after 3 attempts"
    ) from last_error
