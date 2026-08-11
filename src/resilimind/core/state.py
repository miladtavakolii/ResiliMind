from typing import List, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Represents the internal state of the LangGraph workflow.
    Data is passed and mutated through this dictionary across all agent nodes.
    """
    
    # 1. User Input
    user_message: str
    
    # 2. Extractor Agent Outputs
    active_nodes: List[str]
    
    # 3. Graph Retriever Output
    subgraph_context: str
    
    # 4. Assessor Agent Outputs
    assessments: List[Dict[str, Any]]
    requires_disambiguation: bool
    
    # 5. Final Output (from Advisor or Questioner)
    final_response: str
