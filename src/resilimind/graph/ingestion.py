import json
import networkx as nx
import importlib.resources as pkg_resources
from typing import Dict, Any, List, Optional

def load_resilience_graph(json_file_path: Optional[str] = None) -> nx.DiGraph:
    """
    Reads the resilience graph JSON file and returns a directed graph (DiGraph).

    Args:
        json_file_path (Optional[str]): The path to a custom JSON file. 
                                        If None, loads the default bundled asset via pkg_resources.

    Returns:
        nx.DiGraph: A NetworkX directed graph populated with nodes and edges.
    """
    # Initialize an empty directed graph
    G: nx.DiGraph = nx.DiGraph()

    # 1. Read the JSON data
    try:
        if json_file_path:
            # Custom path provided (e.g., for external testing)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                graph_data: Dict[str, Any] = json.load(f)
        else:
            # Default: Load safely from the bundled package assets
            json_text = pkg_resources.files("resilimind.assets").joinpath("final_resilience_graph.json").read_text(encoding="utf-8")
            graph_data: Dict[str, Any] = json.loads(json_text)
            
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return G

    # 2. Add Nodes
    nodes: Dict[str, Any] = graph_data.get("nodes", {})
    for node_id, node_attrs in nodes.items():
        # Store all attributes (e.g., cues, status levels, interventions) as node data
        G.add_node(
            node_id,
            name_fa=node_attrs.get("name_fa"),
            domain=node_attrs.get("domain_fa"),
            description=node_attrs.get("description"),
            cues=node_attrs.get("cues", {}),
            status_levels=node_attrs.get("status_levels", {}),
            interventions=node_attrs.get("interventions", {})
        )
    
    print(f"Successfully loaded {G.number_of_nodes()} nodes.")

    # 3. Add Edges
    edges: List[Dict[str, str]] = graph_data.get("edges", [])
    for edge in edges:
        source: Optional[str] = edge.get("source")
        target: Optional[str] = edge.get("target")
        
        # Verify that both source and target nodes exist in the graph before adding the edge
        if source and target and source in G.nodes and target in G.nodes:
            G.add_edge(
                source, 
                target, 
                relation_type=edge.get("type"),
                description=edge.get("description")
            )
    
    print(f"Successfully loaded {G.number_of_edges()} edges.")
    
    return G

# Script Testing and Data Retrieval Example
if __name__ == "__main__":
    # Build the graph using default path resolution
    resilience_graph: nx.DiGraph = load_resilience_graph()
    
    # 🔍 Example: Extracting data for the RAG system (Node: IND_ECO_01)
    target_node: str = "IND_ECO_01"
    
    if target_node in resilience_graph:
        print(f"\n--- Fetching data for node: {target_node} ---")
        node_data: Dict[str, Any] = resilience_graph.nodes[target_node]
        
        print(f"Domain: {node_data.get('domain')}")
        print(f"Description: {node_data.get('description')}")
        
        # Find nodes affected by this node (outgoing edges / successors)
        successors: List[str] = list(resilience_graph.successors(target_node))
        if successors:
            print(f"\nAffects the following nodes:")
            for succ in successors:
                edge_data: Dict[str, Any] = resilience_graph.get_edge_data(target_node, succ)
                succ_name: str = resilience_graph.nodes[succ].get('name_fa', 'Unknown')
                print(f" -> {succ_name} (Relation Type: {edge_data.get('relation_type')})")
