import logging
from typing import List, Dict, Any
import networkx as nx

# Initialize module logger
logger = logging.getLogger(__name__)

def retrieve_subgraph_context(graph: nx.DiGraph, active_node_ids: List[str]) -> str:
    """
    Extracts node attributes and cross-domain relationships for active nodes 
    to form a formatted context string for LLM prompts.

    Args:
        graph (nx.DiGraph): The loaded NetworkX resilience graph.
        active_node_ids (List[str]): List of node IDs identified in the user message.

    Returns:
        str: Formatted context text containing node definitions, levels, and outgoing edges.
    """
    if not active_node_ids:
        logger.debug("[Retriever] No active node IDs provided for context extraction.")
        return "No specific resilience domains were activated."

    context_blocks: List[str] = []
    logger.debug(f"[Retriever] Extracting context for active nodes: {active_node_ids}")

    for node_id in active_node_ids:
        if node_id not in graph.nodes:
            logger.warning(f"[Retriever] Node ID '{node_id}' not found in the resilience graph.")
            continue

        node_data = graph.nodes[node_id]

        status_levels = node_data.get("status_levels", {})
        cues = node_data.get("cues", {})

        context_blocks.append(
            f"=== Node: {node_id} ({node_data.get('name_fa', '')}) ===\n"
            f"Domain: {node_data.get('domain', '')}\n"
            f"Domain (FA): {node_data.get('domain_fa', '')}\n"
            f"Description: {node_data.get('description', '')}\n"
            f"Positive cues: {cues.get('positive_keywords', [])}\n"
            f"Negative cues: {cues.get('negative_keywords', [])}\n"
            f"Status GREEN: {status_levels.get('green', {})}\n"
            f"Status YELLOW: {status_levels.get('yellow', {})}\n"
            f"Status RED: {status_levels.get('red', {})}"
        )

    if not context_blocks:
        return "No specific resilience domains were activated."

    logger.debug(f"[Retriever] Successfully built context blocks for {len(context_blocks)} nodes.")
    return "\n\n".join(context_blocks)
