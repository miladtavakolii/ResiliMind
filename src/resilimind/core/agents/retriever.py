import logging
from typing import Dict, Any, List

from ..state import AgentState
from ...graph.retriever import retrieve_subgraph_context
from .common import resilience_graph

logger = logging.getLogger(__name__)


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
    logger.debug("[Retriever] Fetched subgraph context for nodes: %s", active_nodes)
    
    return {"subgraph_context": context}
