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

EXTRACTOR_NODE_BOUNDARIES: dict[str, dict[str, str]] = {
    "IND_PER_01": {
        "positive": (
            "Explicit belief in one's ability to overcome a personal challenge, "
            "or explicit persistence despite difficulty."
        ),
        "negative": (
            "Explicit inability, helplessness, giving up, or loss of confidence "
            "specifically about overcoming a personal challenge."
        ),
    },
    "IND_PER_02": {
        "positive": (
            "Explicit emotional control, calming, composure, or successful "
            "regulation of anxiety, anger, panic, or distress."
        ),
        "negative": (
            "Explicit anxiety, anger, panic, emotional overwhelm, or loss of "
            "emotional control."
        ),
    },
    "IND_PER_03": {
        "positive": (
            "Explicit future goals, hope, plans, purposeful direction, or "
            "continued investment in a desired future."
        ),
        "negative": (
            "Explicit hopelessness, loss of future outlook, lack of purpose, "
            "or giving up on future goals."
        ),
    },
    "IND_POL_01": {
        "positive": (
            "Explicit critical evaluation of political or news information, "
            "source checking, fact checking, comparison of sources, or resistance "
            "to misinformation."
        ),
        "negative": (
            "Explicit uncritical acceptance of misinformation, sweeping distrust "
            "of all sources, or black-and-white interpretation of political/news information."
        ),
    },
    "IND_POL_02": {
        "positive": (
            "Explicit belief that the person can influence society, participate "
            "civically or politically, contribute to a community, or take collective action."
        ),
        "negative": (
            "Explicit political or civic helplessness, alienation, or belief that "
            "one's civic actions have no influence on society."
        ),
    },
    "IND_ECO_01": {
        "positive": (
            "Explicit financial management, budgeting, expense control, income "
            "adaptation, saving, or other concrete financial coping."
        ),
        "negative": (
            "Explicit financial difficulty involving inability to manage expenses, "
            "debt, insufficient income, or inability to meet financial needs."
        ),
    },
    "IND_ECO_02": {
        "positive": (
            "Explicit professional adaptation, skill development, career planning, "
            "job searching, or confidence in handling employment challenges."
        ),
        "negative": (
            "Explicit job insecurity, dismissal, unemployment risk, severe workplace "
            "stress, or inability to cope with career instability."
        ),
    },
    "IND_PHY_01": {
        "positive": (
            "Explicit stable energy, physical recovery, physical well-being, or "
            "successful recovery from bodily stress."
        ),
        "negative": (
            "Explicit physical stress symptoms, fatigue, low energy, bodily pain, "
            "or impaired physical recovery."
        ),
    },
    "IND_PHY_02": {
        "positive": (
            "Explicitly stable sleep, appetite, biological rhythm, or healthy "
            "sleep/nutrition routines."
        ),
        "negative": (
            "Explicit sleep disturbance, insomnia, nightmares, appetite disturbance, "
            "or disrupted biological rhythms."
        ),
    },
    "IND_SOC_01": {
        "positive": (
            "Explicit family support, family cohesion, constructive family interaction, "
            "or a sense of safety within the family."
        ),
        "negative": (
            "Explicit family conflict, lack of family support, family insecurity, "
            "or deterioration of family relationships."
        ),
    },
    "IND_SOC_02": {
        "positive": (
            "Explicit support from friends, peers, social network, or active social connection."
        ),
        "negative": (
            "Explicit social isolation, loss of friends/peers, lack of social support, "
            "or deterioration of the person's social network."
        ),
    },
    "IND_SPI_01": {
        "positive": (
            "Explicit faith, prayer, religious belief, spiritual meaning, God, "
            "or spiritual coping as a source of resilience."
        ),
        "negative": (
            "Explicit spiritual crisis, loss of faith, spiritual alienation, or "
            "difficulty finding spiritual meaning."
        ),
    },
    "IND_SPI_02": {
        "positive": (
            "Explicit cultural identity, connection to roots, traditions, heritage, "
            "or cultural belonging."
        ),
        "negative": (
            "Explicit cultural alienation, loss of cultural connection, identity "
            "conflict, or disconnection from roots or heritage."
        ),
    },
}


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

        if existing_polarity == new_polarity:
            merged_polarity = existing_polarity
        else:
            merged_polarity = "mixed"

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

        unique_signals[signal.node_id] = existing.model_copy(
            update={
                "detected_signal": merged_polarity,
                "evidence": selected_evidence,
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
        boundary = EXTRACTOR_NODE_BOUNDARIES.get(node_id, {})
        blocks.append(
            f"Node ID: {node_id}\n"
            f"Name: {node_data.get('name_fa', '')}\n"
            f"Domain: {node_data.get('domain', '')}\n"
            f"Domain (FA): {node_data.get('domain_fa', '')}\n"
            f"Definition: {node_data.get('description', '')}\n"
            f"Positive semantic boundary: {boundary.get('positive', '')}\n"
            f"Negative semantic boundary: {boundary.get('negative', '')}"
        )

    return (
        "=== KNOWLEDGE GRAPH SEMANTIC DEFINITIONS ===\n"
        + "\n\n".join(blocks)
    )


def build_node_selection_criteria() -> str:
    """Format the global node selection criteria prompt block.

    Iterates over all nodes defined in the resilience knowledge graph, extracting
    their explicit semantic boundaries (falling back to node descriptions if an
    explicit boundary override is not registered), and builds structured prompt
    instructions for the first-stage candidate selector.

    Returns:
        str: Formatted multi-line string containing general selection guidelines
            followed by per-node semantic boundary definitions.
    """
    blocks = []

    for node_id, node_data in sorted(resilience_graph.nodes(data=True)):
        boundary = EXTRACTOR_NODE_BOUNDARIES.get(
            node_id,
            node_data.get("description", ""),
        )

        blocks.append(
            f"{node_id}: {boundary}"
        )

    return (
        "=== NODE SELECTION CRITERIA ===\n"
        "Select a node only when the user's message explicitly supports "
        "its specific semantic boundary.\n"
        "Clear semantic paraphrases are valid.\n"
        "Generic or unrelated statements are not valid.\n\n"
        + "\n".join(blocks)
    )


def reconcile_signal_polarity(result: ExtractionOutput) -> ExtractionOutput:
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
    reconciled_signals = []

    for signal in result.active_signals:
        node_data = resilience_graph.nodes.get(signal.node_id, {})
        cues = node_data.get("cues", {})

        positive_hits = _find_polarity_cues(
            signal.evidence,
            cues.get("positive_keywords", []),
        )
        negative_hits = _find_polarity_cues(
            signal.evidence,
            cues.get("negative_keywords", []),
        )

        inferred_polarity = signal.detected_signal

        if positive_hits and not negative_hits:
            if signal.detected_signal in {"positive", "mixed"}:
                inferred_polarity = "positive"
            else:
                inferred_polarity = signal.detected_signal
        elif negative_hits and not positive_hits:
            if signal.detected_signal in {"negative", "mixed"}:
                inferred_polarity = "negative"
            else:
                inferred_polarity = signal.detected_signal
        elif positive_hits and negative_hits:
            inferred_polarity = "mixed"
        else:
            inferred_polarity = signal.detected_signal

        if inferred_polarity != signal.detected_signal:
            logger.warning(
                "[Extractor] Reconciling polarity for %s: %s -> %s "
                "(positive_cues=%s, negative_cues=%s)",
                signal.node_id,
                signal.detected_signal,
                inferred_polarity,
                positive_hits,
                negative_hits,
            )

        reconciled_signals.append(
            signal.model_copy(
                update={"detected_signal": inferred_polarity}
            )
        )

    return result.model_copy(
        update={"active_signals": reconciled_signals}
    )


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


def build_evidence_candidates(user_message: str) -> list[tuple[int, int, str]]:
    """Segment raw user text into candidate evidence clauses with character offsets.

    Splits the message using Persian and English clause terminators and punctuation
    marks (commas, semicolons, full stops, question marks, exclamation marks, and
    newlines). Trims leading and trailing whitespace while adjusting exact start and
    end slice indices to guarantee precise substring alignment with the original text.

    Args:
        user_message: Raw user message string to segment.

    Returns:
        list[tuple[int, int, str]]: Triples of `(start_index, end_index, clause_text)`
            representing trimmed contiguous candidate spans within `user_message`.
    """
    candidates: list[tuple[int, int, str]] = []

    for match in re.finditer(
        r"[^،,؛;.!?؟\n]+",
        user_message,
    ):
        start, end = match.span()
        raw_segment = user_message[start:end]

        text = raw_segment.strip()
        if not text:
            continue

        leading_ws = len(raw_segment) - len(raw_segment.lstrip())
        trailing_ws = len(raw_segment) - len(raw_segment.rstrip())

        start += leading_ws
        end -= trailing_ws

        candidates.append(
            (start, end, user_message[start:end])
        )

    return candidates


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

        corrected_signals.append(
            signal.model_copy(update={"evidence": evidence})
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

        for cue in (
            cues.get("positive_keywords", [])
            + cues.get("negative_keywords", [])
        ):
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

        if not positive_hits and not negative_hits:
            continue

        positive_text = ", ".join(positive_hits) if positive_hits else "none"
        negative_text = ", ".join(negative_hits) if negative_hits else "none"

        matched_cues = {
            normalize_persian_text(cue)
            for cue in positive_hits + negative_hits
        }

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
            f"Semantic boundary: {EXTRACTOR_NODE_BOUNDARIES.get(node_id, '')}\n"
            f"Positive cue matches: {positive_text}\n"
            f"Negative cue matches: {negative_text}\n"
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
        boundary = EXTRACTOR_NODE_BOUNDARIES.get(node_id, {})
        blocks.append(
            f"Node ID: {node_id}\n"
            f"Name: {node.get('name_fa', '')}\n"
            f"Domain: {node.get('domain', '')}\n"
            f"Definition: {node.get('description', '')}\n"
            f"Semantic boundary: {EXTRACTOR_NODE_BOUNDARIES.get(node_id, '')}\n"
            f"Positive semantic boundary: {boundary.get('positive', '')}\n"
            f"Negative semantic boundary: {boundary.get('negative', '')}"
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

    selector_prompt = (
        f"{prompts.NODE_SELECTOR_SYSTEM_PROMPT}\n\n"
        f"{build_extractor_graph_context()}\n\n"
        f"=== CANDIDATE HINTS ===\n"
        f"{build_extractor_candidate_hints(user_msg)}\n\n"
        f"{build_node_selection_criteria()}"
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

        selected_nodes = list(dict.fromkeys(raw["parsed"].node_ids))

        if not selected_nodes:
            return {
                "active_nodes": [],
                "active_signals": [],
            }

        resolved_signals = []

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
                    f"[{index}] {candidate}"
                    for index, candidate in enumerate(evidence_candidates)
                )
                + "\n\n"
                f"The target node for this invocation is {node_id}.\n"
                "Return exactly one signal for this node.\n"
                "Select exactly one evidence candidate by index."
            )

            node_last_error: Exception | None = None
            node_resolved = None

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

                evidence = evidence_candidates[signal.evidence_index][2]
                logger.info(
                    "[Extractor] Resolved %s -> evidence_index=%d, evidence=%r",
                    node_id,
                    signal.evidence_index,
                    evidence_candidates[signal.evidence_index][2],
                )
                node_resolved = {
                    "node_id": signal.node_id,
                    "detected_signal": signal.detected_signal,
                    "evidence": evidence,
                }
                break

            if node_resolved is None:
                raise RuntimeError(
                    f"Signal resolution failed for node {node_id}"
                ) from node_last_error

            resolved_signals.append(node_resolved)

        result = ExtractionOutput(
            active_signals=resolved_signals
        )

        result = canonicalize_extraction_result(result)
        result = validate_extraction_result(
            result=result,
            user_message=user_msg,
        )
        result = reconcile_signal_polarity(result)

        return {
            "active_nodes": selected_nodes,
            "active_signals": [
                signal.model_dump()
                for signal in result.active_signals
            ],
        }

    raise RuntimeError(
        "Staged extractor failed after 3 attempts"
    ) from last_error
