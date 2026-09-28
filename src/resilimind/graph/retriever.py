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
        block = (
            f"=== Node: {node_id} ({node_data.get('name_fa', '')}) ===\n"
            f"Domain: {node_data.get('domain', '')}\n"
            f"Domain (FA): {node_data.get('domain_fa', '')}\n"
            f"Description: {node_data.get('description', '')}\n\n"
            "Status Level Definitions:\n"
        )

        status_levels = node_data.get("status_levels", {})
        for level_color, level_info in status_levels.items():
            if isinstance(level_info, dict):
                block += (
                    f"  - [{level_color.upper()}] "
                    f"({level_info.get('code', '')}): "
                    f"{level_info.get('description', '')}\n"
                )

        context_blocks.append(block)

    logger.debug(f"[Retriever] Successfully built context blocks for {len(context_blocks)} nodes.")
    return "\n" + "=" * 50 + "\n".join(context_blocks)
