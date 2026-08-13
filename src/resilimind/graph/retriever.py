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

        node_data: Dict[str, Any] = graph.nodes[node_id]
        
        # Format Node Basic Info
        block: str = f"=== Node: {node_id} ({node_data.get('name_fa', '')}) ===\n"
        block += f"Domain: {node_data.get('domain', '')}\n"
        block += f"Description: {node_data.get('description', '')}\n\n"

        # Format Status Levels
        status_levels: Dict[str, Any] = node_data.get("status_levels", {})
        block += "Status Level Definitions:\n"
        for level_color, level_info in status_levels.items():
            if isinstance(level_info, dict):
                block += f"  - [{level_color.upper()}] ({level_info.get('code', '')}): {level_info.get('description', '')}\n"
        
        # Format Interventions
        interventions: Dict[str, str] = node_data.get("interventions", {})
        block += "\nRecommended Interventions:\n"
        for cond, action in interventions.items():
            block += f"  - {cond}: {action}\n"

        # Format Outgoing Cross-Domain Edges
        successors: List[str] = list(graph.successors(node_id))
        if successors:
            block += "\nCross-Domain Impacts on Other Nodes:\n"
            for succ in successors:
                edge_data: Dict[str, Any] = graph.get_edge_data(node_id, succ)
                succ_name: str = graph.nodes[succ].get("name_fa", succ)
                block += f"  -> Affects '{succ_name}' [{succ}] | Relation: {edge_data.get('relation_type', '')} ({edge_data.get('description', '')})\n"

        context_blocks.append(block)

    logger.debug(f"[Retriever] Successfully built context blocks for {len(context_blocks)} nodes.")
    return "\n" + "=" * 50 + "\n".join(context_blocks)
