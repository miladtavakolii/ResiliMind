from collections import Counter
import logging
from typing import Dict, Any, List
import networkx as nx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
import re
import unicodedata

from .state import AgentState
from .database import get_user_node_timeline
from ..graph.ingestion import load_resilience_graph
from ..graph.retriever import retrieve_subgraph_context
from ..llm.engine import LLMEngine
from ..llm import prompts
from ..schemas.models import ExtractionOutput, AssessmentOutput, SafetyOutput
from .config import settings

# Initialize module logger
logger = logging.getLogger(__name__)

# Initialize LLM Engine singleton and load graph into memory once
llm_engine: LLMEngine = LLMEngine()
resilience_graph: nx.DiGraph = load_resilience_graph()

EXTRACTOR_NODE_BOUNDARIES: dict[str, str] = {
    "IND_PER_01": (
        "Requires explicit self-efficacy or hardiness about overcoming a challenge. "
        "Generic task completion, planning, calmness, social support, or coping effort alone "
        "does not activate this node."
    ),
    "IND_PER_02": (
        "Requires emotional regulation or dysregulation such as anxiety, anger, panic, "
        "loss of emotional control, or explicit calming. Generic confusion alone is insufficient."
    ),
    "IND_PER_03": (
        "Requires future goals, future outlook, hope, or purposeful personal direction. "
        "Generic task completion is insufficient."
    ),
    "IND_POL_01": (
        "Requires evaluating political/news information, source verification, misinformation, "
        "or critical analysis of news."
    ),
    "IND_POL_02": (
        "Requires civic or political agency/alienation. Generic 'I cannot do anything' "
        "is insufficient unless the statement is explicitly about political or civic impact."
    ),
    "IND_ECO_01": (
        "Requires financial management or financial adaptation. General stress about work "
        "is not sufficient."
    ),
    "IND_ECO_02": (
        "Requires employment, career, job security, dismissal, or professional adaptation. "
        "Generic self-efficacy is not sufficient."
    ),
    "IND_PHY_01": (
        "Requires bodily symptoms, fatigue, energy, or physical recovery. Psychological "
        "stress alone is insufficient."
    ),
    "IND_PHY_02": (
        "Requires sleep, wakefulness, appetite, or biological-rhythm evidence. Generic fatigue "
        "alone is insufficient."
    ),
    "IND_SOC_01": (
        "Requires family-specific support, cohesion, conflict, or safety."
    ),
    "IND_SOC_02": (
        "Requires friends, peers, social network, or social-support evidence."
    ),
    "IND_SPI_01": (
        "Requires spiritual/religious meaning, faith, prayer, God, or spiritual coping."
    ),
    "IND_SPI_02": (
        "Requires cultural identity, roots, traditions, heritage, or cultural belonging."
    ),
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
            logger.warning(f"[Extractor] Merging duplicate node {signal.node_id} polarities: {existing_polarity} + {new_polarity} -> mixed.")

        unique_signals[signal.node_id] = existing.model_copy(
            update={
                "detected_signal": merged_polarity,
                "evidence": selected_evidence,
            }
        )

    return result.model_copy(
        update={"active_signals": list(unique_signals.values())}
    )

def _find_polarity_cues(text: str, cues: list[str]) -> list[str]:
    """Find whole-word cue matches within normalized text.

    Args:
        text: Target text segment (e.g., extracted signal evidence).
        cues: List of lexical cue phrases defined for a resilience node.

    Returns:
        list[str]: Cues from the input list that appear as isolated words or phrases
            in the normalized text.
    """
    normalized_text = normalize_persian_text(text)
    hits: list[str] = []

    for cue in cues:
        normalized_cue = normalize_persian_text(cue)
        if not normalized_cue:
            continue

        pattern = rf"(?<!\w){re.escape(normalized_cue)}(?!\w)"
        if re.search(pattern, normalized_text):
            hits.append(cue)

    return hits


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
            f"Domain: {node_data.get('domain', '')}\n"
            f"Domain (FA): {node_data.get('domain_fa', '')}\n"
            f"Definition: {node_data.get('description', '')}\n"
            f"Semantic boundary: {EXTRACTOR_NODE_BOUNDARIES.get(node_id, '')}"
        )

    return (
        "=== KNOWLEDGE GRAPH SEMANTIC DEFINITIONS ===\n"
        + "\n\n".join(blocks)
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

        if positive_hits and negative_hits:
            inferred_polarity = "mixed"
        elif positive_hits:
            inferred_polarity = "positive"
        elif negative_hits:
            inferred_polarity = "negative"
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
    if not evidence:
        return None

    if evidence in user_message:
        return evidence

    def compact_with_map(text: str) -> tuple[str, list[int]]:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.translate(
            str.maketrans({
                "ي": "ی",
                "ى": "ی",
                "ئ": "ی",
                "ك": "ک",
                "ة": "ه",
                "ۀ": "ه",
            })
        )

        compact_chars: list[str] = []
        source_indices: list[int] = []

        for index, char in enumerate(normalized):
            if char.isspace() or char == "\u200c":
                continue
            compact_chars.append(char)
            source_indices.append(index)

        return "".join(compact_chars), source_indices

    candidate = evidence.strip()
    compact_candidate, _ = compact_with_map(candidate)
    compact_message, source_indices = compact_with_map(user_message)

    if not compact_candidate or compact_candidate not in compact_message:
        return None

    start = compact_message.index(compact_candidate)
    end = start + len(compact_candidate)

    raw_start = source_indices[start]
    raw_end = source_indices[end - 1] + 1

    return user_message[raw_start:raw_end]

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

def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's input message to extract active resilience nodes 
    using the Gemma LLM with structured output enforcement.

    Args:
        state (AgentState): Current state containing 'user_message'.

    Returns:
        Dict[str, Any]: Updated state dict with 'active_nodes'.
    """
    logger.info("[Extractor] Extractor Agent is analyzing input...")
    user_msg: str = state.get("user_message", "")

    extractor_prompt_base = (
        f"{prompts.EXTRACTOR_SYSTEM_PROMPT}\n\n"
        f"{build_extractor_graph_context()}\n\n"
        f"=== EXPLICIT CANDIDATE HINTS ===\n"
        f"{build_extractor_candidate_hints(user_msg)}"
    )

    last_error: Exception | None = None

    for attempt in range(3):
        extractor_prompt = extractor_prompt_base

        if last_error is not None:
            extractor_prompt += (
                "\n\n=== VALIDATION RETRY ===\n"
                "The previous extraction failed validation.\n"
                "Do not reuse the previous evidence.\n"
                "For every signal, evidence must be copied exactly from the USER INPUT.\n"
                "Choose the shortest contiguous span that directly supports the node.\n"
                "Do not combine multiple phrases into one evidence span.\n"
                "If you cannot identify a valid exact span, omit the signal.\n"
            )

        extractor_chain = llm_engine.get_extractor_runner(extractor_prompt)
        raw_result = extractor_chain.invoke({"user_message": user_msg})

        if raw_result.get("parsed") is None:
            raw = raw_result.get("raw")
            last_error = ValueError(
                f"Extractor returned invalid structured output: {raw!r}"
            )
            logger.warning(
                "[Extractor] Structured output validation failed "
                "(attempt %d/3): %s",
                attempt + 1,
                last_error,
            )
            continue

        try:
            result = canonicalize_extraction_result(
                raw_result["parsed"]
            )
            result = validate_extraction_result(
                result=result,
                user_message=user_msg,
            )
            result = reconcile_signal_polarity(result)
            break
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "[Extractor] Extraction validation failed "
                "(attempt %d/3): %s",
                attempt + 1,
                exc,
            )
    else:
        raise RuntimeError(
            f"Extractor failed after 3 attempts"
        ) from last_error
    
    signals_list: List[Dict[str, Any]] = [
        signal.model_dump() for signal in result.active_signals
    ]
    
    # Extract unique node IDs from active signals
    active_node_ids = [signal.node_id for signal in result.active_signals]
    logger.debug(f"[Extractor] Extracted {len(active_node_ids)} nodes and {len(signals_list)} signals.")
    
    return {"active_nodes": active_node_ids, "active_signals": signals_list}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves formatted sub-graph context for active nodes from NetworkX.

    Args:
        state (AgentState): Current state containing 'active_nodes'.

    Returns:
        Dict[str, Any]: Updated state dict with 'subgraph_context'.
    """
    logger.info("[Retriever] Graph Retriever is fetching node knowledge...")
    active_nodes: List[str] = state.get("active_nodes", [])
    
    # Fetch structured string representation from NetworkX graph
    context: str = retrieve_subgraph_context(resilience_graph, active_nodes)
    logger.debug(f"[Retriever] Fetched subgraph context for nodes: {active_nodes}")
    
    return {"subgraph_context": context}


def calculate_composite_confidence(
    raw_confidence: float,
    signal_polarity: str,
    evidence_text: str,
    node_id: str,
    user_message: str,
    assessment_score: int,
) -> float:
    """Compute deterministic routing confidence from four independent signals.

    Combines self-reported model confidence, Persian substring evidence validity,
    lexical cue-polarity consistency, and distance-to-boundary assessment certainty:
        - 35% LLM self-reported confidence
        - 20% Evidence substring validity (in normalized user message)
        - 25% Polarity consistency (concordance with knowledge graph cues)
        - 20% Assessment certainty (distance from 40/70 clinical threshold boundaries)

    Args:
        raw_confidence: Model self-reported extraction confidence score (0.0 to 1.0).
        signal_polarity: Detected emotional/situational polarity ('positive', 'negative', 'mixed').
        evidence_text: Extracted supporting evidence phrase or substring.
        node_id: Target knowledge graph node identifier.
        user_message: Raw user message text against which evidence is verified.
        assessment_score: Total quantitative rubric score (e.g., 0 to 100).

    Returns:
        float: Calibrated routing confidence score bounded between 0.0 and 1.0,
            rounded to two decimal places.
    """
    llm_confidence = min(max(raw_confidence, 0.0), 1.0)

    node_data = resilience_graph.nodes.get(node_id, {})
    cues = node_data.get("cues", {})

    positive_hits = _find_polarity_cues(
        evidence_text,
        cues.get("positive_keywords", []),
    )
    negative_hits = _find_polarity_cues(
        evidence_text,
        cues.get("negative_keywords", []),
    )

    normalized_evidence = normalize_persian_text(evidence_text)
    normalized_message = normalize_persian_text(user_message)

    if not normalized_evidence:
        evidence_validity = 0.0
    elif normalized_evidence in normalized_message:
        evidence_validity = 1.0
    else:
        evidence_validity = 0.0

    polarity_lower = signal_polarity.lower()

    if positive_hits and negative_hits:
        polarity_consistency = 1.0 if polarity_lower == "mixed" else 0.0
    elif positive_hits:
        polarity_consistency = 1.0 if polarity_lower == "positive" else 0.0
    elif negative_hits:
        polarity_consistency = 1.0 if polarity_lower == "negative" else 0.0
    else:
        polarity_consistency = 0.5

    distance_to_boundary = min(
        abs(assessment_score - 40),
        abs(assessment_score - 70),
    )
    assessment_certainty = min(distance_to_boundary / 15.0, 1.0)

    composite_score = (
        llm_confidence * 0.35
        + evidence_validity * 0.20
        + polarity_consistency * 0.25
        + assessment_certainty * 0.20
    )

    final_confidence = round(
        min(max(composite_score, 0.0), 1.0),
        2,
    )

    logger.debug(
        "[Confidence] node=%s confidence=%s raw=%s "
        "evidence_validity=%s polarity_consistency=%s "
        "assessment_certainty=%s score=%s",
        node_id,
        final_confidence,
        raw_confidence,
        evidence_validity,
        polarity_consistency,
        assessment_certainty,
        assessment_score,
    )

    return final_confidence

def validate_assessment_result(result: AssessmentOutput, active_signals: list[dict[str, Any]]) -> AssessmentOutput:
    """Validate assessor output against active signals and graph topology.

    Verifies 1-to-1 parity between assessed nodes and active extraction signals,
    confirms node existence in the knowledge graph, checks for duplicate
    assessments, and ensures each assessment's category matches the node's domain.

    Args:
        result: AssessmentOutput instance containing scored node assessments.
        active_signals: List of extracted active signal dictionaries for the turn.

    Returns:
        AssessmentOutput: The validated assessment output instance.

    Raises:
        ValueError: If duplicate assessments are found, unknown or inactive nodes
            are assessed, active nodes are omitted, or an assessment category
            mismatches the node's domain in the resilience graph.
    """
    valid_node_ids = set(resilience_graph.nodes)
    signal_set = {signal["node_id"] for signal in active_signals}

    assessment_ids = [assessment.node_id for assessment in result.assessments]
    assessment_set = set(assessment_ids)

    if len(assessment_ids) != len(assessment_set):
        counts = Counter(assessment_ids)
        duplicate_ids = sorted(
            node_id for node_id, count in counts.items() if count > 1
        )
        raise ValueError(
            f"Assessor returned duplicate assessments: {duplicate_ids}"
        )

    if unknown_ids := sorted(assessment_set - valid_node_ids):
        raise ValueError(f"Assessor returned unknown nodes: {unknown_ids}")

    if extra_ids := sorted(assessment_set - signal_set):
        raise ValueError(
            f"Assessor returned assessments for inactive nodes: {extra_ids}"
        )

    if missing_ids := sorted(signal_set - assessment_set):
        raise ValueError(
            f"Assessor missing assessments for active nodes: {missing_ids}"
        )

    for assessment in result.assessments:
        expected_domain = resilience_graph.nodes[assessment.node_id].get("domain")
        if assessment.category != expected_domain:
            raise ValueError(
                f"Assessor returned wrong category for {assessment.node_id}: "
                f"expected={expected_domain}, got={assessment.category}"
            )

    return result

def assessor_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates resilience levels using extracted evidence and polarity signals,
    grounded in the retrieved subgraph context.

    Args:
        state (AgentState): Current state containing 'user_message', 
                            'subgraph_context', and 'active_signals'.

    Returns:
        Dict[str, Any]: Updated state with 'assessments' and 'requires_disambiguation'.
    """
    logger.info("[Assessor] Assessor Agent is evaluating resilience status with evidence...")
    user_msg: str = state.get("user_message", "")
    context: str = state.get("subgraph_context", "")
    active_signals: List[Dict[str, Any]] = state.get("active_signals", [])
    if not active_signals:
        logger.info("[Assessor] No active signals. Skipping assessment.")
        return {
            "assessments": [],
            "requires_disambiguation": False,
        }

    # 1. Format extracted signals & evidence substrings for prompt ingestion
    if active_signals:
        evidence_blocks = []
        for index, sig in enumerate(active_signals, start=1):
            evidence_blocks.append(
                f"TARGET {index}/{len(active_signals)}\n"
                f"Node ID: {sig.get('node_id')}\n"
                f"Polarity: {sig.get('detected_signal', 'mixed').upper()}\n"
                f"Exact Evidence: \"{sig.get('evidence', '')}\""
            )
        formatted_evidence = "\n\n".join(evidence_blocks)
    else:
        logger.warning("[Assessor] No explicit extracted signals provided.")
        formatted_evidence = "No explicit extracted signals provided."

    # 2. Construct clear evidence-aware payload conforming to assessor.txt prompt
    enriched_input = (
        "=== PRIMARY EVIDENCE BY TARGET NODE ===\n"
        f"{formatted_evidence}\n\n"
        "=== FULL USER MESSAGE ===\n"
        f"{user_msg}\n\n"
        "RULE:\n"
        "The exact evidence assigned to each node is the primary evidence for that node.\n"
        "Use the full message only to resolve local context or polarity.\n"
        "Do not borrow severity, frequency, functional impact, or coping evidence from "
        "another target node.\n"
    )

    # 3. Invoke LLM chain with evidence payload
    assessor_chain = llm_engine.get_assessor_runner(prompts.ASSESSOR_SYSTEM_PROMPT)
    max_assessment_attempts = 3
    last_validation_error: Exception | None = None

    for attempt in range(max_assessment_attempts):
        attempt_input = enriched_input

        if last_validation_error is not None:
            attempt_input += (
                "\n\n=== PREVIOUS ASSESSMENT VALIDATION FAILURE ===\n"
                f"{last_validation_error}\n\n"
                "Regenerate the assessment output completely.\n"
                "You MUST return exactly one assessment for every active node.\n"
                "Do not omit any active node.\n"
                "Do not return an empty assessments list.\n"
            )

        raw_result = assessor_chain.invoke({
            "user_message": attempt_input,
            "subgraph_context": context,
        })

        if raw_result.get("parsed") is None:
            raw = raw_result.get("raw")
            last_validation_error = ValueError(
                f"Assessor returned invalid structured output: {raw!r}"
            )
            logger.warning(
                "[Assessor] Structured output validation failed "
                "(attempt %d/%d). Retrying...",
                attempt + 1,
                max_assessment_attempts,
            )
            continue

        result = raw_result["parsed"]

        try:
            result = validate_assessment_result(
                result=result,
                active_signals=active_signals,
            )
            break
        except ValueError as exc:
            last_validation_error = exc
            logger.warning(
                "[Assessor] Assessment coverage validation failed "
                "(attempt %d/%d): %s",
                attempt + 1,
                max_assessment_attempts,
                exc,
            )
    else:
        raise RuntimeError(
            f"[Assessor] Failed to produce complete assessments after "
            f"{max_assessment_attempts} attempts"
        ) from last_validation_error
    
    # Create a quick lookup for active signals to match with assessments
    signal_lookup = {sig["node_id"]: sig for sig in active_signals}

    requires_disambiguation_override = False
    assessments_list: List[Dict[str, Any]] = []

    # 4. Compute heuristic routing confidence from multiple signals
    for assessment in result.assessments:
        assessment_dict = assessment.model_dump()
        assessment_dict["score"] = assessment.score
        assessment_dict["status"] = assessment.status
        node_id = assessment_dict["node_id"]
        
        # Get corresponding signal data from Extractor
        signal_data = signal_lookup.get(node_id, {})
        signal_polarity = signal_data.get("detected_signal", "mixed")
        evidence_text = signal_data.get("evidence", "")

        # Recalculate true confidence
        routing_confidence = calculate_composite_confidence(
            raw_confidence=assessment_dict["confidence"],
            signal_polarity=signal_polarity,
            evidence_text=evidence_text,
            node_id=node_id,
            user_message=user_msg,
            assessment_score=assessment_dict["score"],
        )
        
        # Override the LLM's self-reported confidence
        assessment_dict["confidence"] = routing_confidence
        assessments_list.append(assessment_dict)

        # Re-evaluate routing logic based on the heuristic confidence score
        if routing_confidence < settings.RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD:
            logger.warning(f"[Assessor] Low confidence detected for node {node_id} ({routing_confidence}). Flagging disambiguation.")
            requires_disambiguation_override = True

    return {
        "assessments": assessments_list,
        "requires_disambiguation": requires_disambiguation_override 
    }


def questioner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a targeted, empathetic clarification question when input is ambiguous,
    using the full conversational memory.

    Args:
        state (AgentState): Current state with 'messages' and 'subgraph_context'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.info("[Questioner] Questioner Agent is formulating clarification...")
    context: str = state.get("subgraph_context", "")
    
    # Fetch the full chat history from the graph state
    messages_history = state.get("messages", [])

    stream_handler = config.get("configurable", {}).get("stream_handler")
    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_questioner_prompt()
    chain = prompt | conversational_llm    
    question_text = ""
    for chunk in chain.stream({
        "user_message": state.get("user_message", ""),
        "subgraph_context": context,
        "messages": messages_history
    }):
        question_text += chunk.content
        if stream_handler:
            stream_handler.on_llm_new_token(chunk.content)

    logger.debug("[Questioner] Clarification question successfully streamed.")
    return {
        "route": "questioner",
        "final_response": question_text,
        "messages": [AIMessage(content=question_text)]
    }


def advisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates tailored psychological advice and interventions using active graph guidance
    combined with the user's historical resilience profile and full conversational memory.

    Args:
        state (AgentState): Current state with 'user_id', 'messages', 'subgraph_context', and 'assessments'.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.info("[Advisor] Advisor Agent is generating psychological interventions with full memory...")
    user_id: int = state.get("user_id", 0)
    subgraph_context: str = state.get("subgraph_context", "")
    assessments: List[Dict[str, Any]] = state.get("assessments", [])
    
    # Fetch the full chat history from the graph state
    messages_history = state.get("messages", [])
    
    # Fetch user's historical resilience timeline from SQLite database
    history_logs: List[Dict[str, Any]] = get_user_node_timeline(user_id) if user_id else []
    
    # Format historical profile into a chronological timeline block
    history_context: str = "No prior historical timeline recorded."
    if history_logs:
        timeline_by_node = {}
        for log in history_logs:
            nid = log.get('node_id')
            if nid not in timeline_by_node:
                timeline_by_node[nid] = []
            
            date_str = str(log.get('created_at', ''))[:10]
            status = log.get('status', 'UNKNOWN')
            score = log.get('score', 'N/A')
            timeline_by_node[nid].append(f"[{date_str}] {status}({score})")
        
        formatted_logs = []
        for nid, timeline in timeline_by_node.items():
            path_str = " ➔ ".join(timeline)
            formatted_logs.append(f"• Node {nid} Timeline: {path_str}")
            
        history_context = "\n".join(formatted_logs)
        logger.debug(f"[Advisor] Loaded timeline history for user {user_id}.")
    
    # Combine real-time graph context with the user's historical profile
    full_context: str = (
        f"=== CURRENT GRAPH KNOWLEDGE ===\n{subgraph_context}\n\n"
        f"=== USER HISTORICAL RESILIENCE PROFILE ===\n{history_context}"
    )

    stream_handler = config.get("configurable", {}).get("stream_handler")
    conversational_llm = llm_engine.get_conversational_llm()
    prompt = prompts.get_advisor_prompt()
    chain = prompt | conversational_llm    
    advice_text = ""
    for chunk in chain.stream({
        "user_message": state.get("user_message", ""),
        "subgraph_context": full_context,
        "assessments": str(assessments),
        "messages": messages_history
    }):
        advice_text += chunk.content
        if stream_handler:
            stream_handler.on_llm_new_token(chunk.content)

    if not advice_text or not advice_text.strip():
        logger.error("[Advisor] LLM returned an empty response.")
        advice_text = "متأسفانه در پردازش پاسخ خطایی رخ داد. لطفاً صفحه را رفرش کرده یا نشست جدیدی آغاز کنید."
    else:
        logger.debug("[Advisor] Advice successfully streamed and generated.")
    
    return {
        "route": "advisor",
        "final_response": advice_text,
        "messages": [AIMessage(content=advice_text)]
    }


def normalize_persian_text(text: str) -> str:
    """
    Standardizes Persian text by unifying characters, removing diacritics, 
    punctuations, ZWNJ (\u200c), and normalizing whitespaces.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    translation_table = str.maketrans({
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ۀ": "ه",
        "ة": "ه",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    })
    text = text.translate(translation_table)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = re.sub(r'[!?,.:;؛؟"\'()\-\[\]{}<>/\\]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def safety_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Primary zero-tolerance safety gate.
    Pipeline: Raw Input -> Persian Normalization -> High-Recall Heuristic -> LLM Classifier -> Policy Router
    """
    logger.info("[Safety] Safety Gate is checking for high-risk signals...")
    user_msg: str = state.get("user_message", "")
    
    # 1. Text Normalization
    normalized_msg = normalize_persian_text(user_msg)
    
    # 2. Context-Aware Crisis Intent Phrases (High-Recall, Low-False-Positive)
    crisis_phrases = [
        "خودکشی",
        "خودمو بکشم",
        "خودم رو بکشم",
        "خودکشی کنم",
        "نمیخوام زنده باشم",
        "نمی خواهم زنده باشم",
        "میخوام بمیرم",
        "می خواهم بمیرم",
        "پایان بدم به زندگی",
        "پایان دادن به زندگی",
        "رگمو بزنم",
        "رگم رو بزنم",
        "رگم بزنم",
        "قصدم خودکشیه",
        "خسته شدم از زندگی میخوام بمیرم"
    ]

    # Check for multi-word phrases or explicit suicide intent
    if any(phrase in normalized_msg for phrase in crisis_phrases):
        logger.warning(f"[Safety] Fast heuristic triggered high-risk flag on normalized input.")
        return {
            "safety_status": "HIGH_RISK",
            "safety_flag": True,
            "safety_risk_category": "SELF_HARM",
        }
        
    # 3. LLM-based Safety Classification (Context-aware fallback)
    try:
        safety_chain: Any = llm_engine.get_safety_runner(prompts.SAFETY_CLASSIFIER_PROMPT)
        raw_result: SafetyOutput = safety_chain.invoke({"user_message": user_msg})
        
        if raw_result.get("parsed") is None:
            raw = raw_result.get("raw")
            logger.error("[Safety] Structured output parsing failed. Raw model output: %r", raw)
            raise ValueError(f"Safety returned invalid structured output: {raw!r}")

        result: SafetyOutput = raw_result["parsed"]

        effective_high_risk = (
            result.is_high_risk or result.risk_category != "SAFE"
        )

        if result.is_high_risk != (result.risk_category != "SAFE"):
            logger.warning(f"[Safety] Inconsistent classifier output: is_high_risk={result.is_high_risk}, risk_category={result.risk_category}. Applying fail-safe high-risk interpretation.")

        if effective_high_risk:
            logger.warning(f"[Safety] LLM Safety Classifier flagged high-risk signal. Category: {result.risk_category}")
            return {
                "safety_status": "HIGH_RISK",
                "safety_flag": True,
                "safety_risk_category": result.risk_category,
            }

        logger.debug("[Safety] Input evaluated as SAFE.")
        return {
            "safety_status": "SAFE",
            "safety_flag": False,
            "safety_risk_category": "SAFE",
        }
        
    except Exception as e:
        logger.error(f"[Safety] Safety LLM execution failed ({e}). Defaulting to SAFETY_UNAVAILABLE status.")
        return {
            "safety_status": "UNAVAILABLE",
            "safety_flag": False,
            "safety_risk_category": "SAFE",
        }


def service_unavailable_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles cases where the safety subsystem or LLM backend is unavailable, 
    preventing un-vetted processing while gracefully informing the user.

    Args:
        state (AgentState): The current state dictionary of the workflow.

    Returns:
        Dict[str, Any]: A state update dictionary containing the formatted service 
                        unavailable message and appended AI message history.
    """
    logger.error("[ServiceUnavailable] Pipeline halted: Safety subsystem is unavailable.")
    unavailable_text: str = (
        "⚠️ **The system is temporarily experiencing some issues with the safety assessment section.**\n\n"
        "For security reasons, it is not possible to continue the psychological analysis at this time. "
        "Please try again in a few minutes or contact the help desk if you need immediate support."
    )
    return {
        "route": "service_unavailable",
        "final_response": unavailable_text,
        "messages": [AIMessage(content=unavailable_text)]
    }


def emergency_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a deterministic emergency intervention response with official crisis hotlines,
    bypassing LLM generation completely to avoid hallucinations or clinical risks.

    Args:
        state (AgentState): Current graph state.

    Returns:
        Dict[str, Any]: Updated state dict with 'final_response' and 'messages'.
    """
    logger.critical("[Emergency] 🚨 HIGH RISK DETECTED: Routing to Emergency Protocols!")
    
    # Load static, clinically approved crisis text directly from prompt configuration
    emergency_text: str = prompts.EMERGENCY_RESPONSE_TEMPLATE
    
    return {
        "route": "emergency_response",
        "final_response": emergency_text,
        "messages": [AIMessage(content=emergency_text)]
    }
