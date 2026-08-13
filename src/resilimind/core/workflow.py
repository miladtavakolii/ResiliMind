import sqlite3
from pathlib import Path
from typing import Literal, Any, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import the shared state structure and individual agent node functions
from .state import AgentState
from .agents import (
    safety_classifier_node,
    emergency_response_node,
    extractor_node,
    retriever_node,
    assessor_node,
    questioner_node,
    advisor_node
)

# Define path for SQLite persistent checkpoint database
CHECKPOINT_DB_PATH: Path = Path.cwd() / "data" / "checkpoints.db"


def route_safety(state: AgentState) -> Literal["emergency_response", "extractor"]:
    """
    Determines if the input contains a high-risk crisis signal requiring 
    immediate emergency intervention, bypassing normal resilience analysis.

    Args:
        state (AgentState): The current state dictionary of the workflow.

    Returns:
        Literal["emergency_response", "extractor"]: Target node identifier.
    """
    if state.get("safety_flag", False):
        print("🚨 Routing to emergency response protocol...")
        return "emergency_response"
    return "extractor"


def route_after_assessment(state: AgentState) -> Literal["questioner", "advisor"]:
    """
    Conditional edge router that determines whether to ask a clarifying question
    or to provide final psychological advice based on disambiguation status and confidence threshold.

    Args:
        state (AgentState): The current state dictionary of the workflow.

    Returns:
        Literal["questioner", "advisor"]: Target agent node identifier.
    """
    # 1. Check explicit disambiguation flag set by Assessor Agent
    if state.get("requires_disambiguation", False):
        return "questioner"
    
    # 2. Retrieve extracted node assessments
    assessments: List[Dict[str, Any]] = state.get("assessments", [])
    
    # If no nodes were extracted or active, route to Questioner for clarification
    if not assessments:
        return "questioner"

    # If any confidence score falls below the 0.70 threshold, force disambiguation
    for item in assessments:
        confidence: float = item.get("confidence", 1.0)
        if confidence < 0.70:
            print(f"⚠️ Low confidence detected ({confidence}). Routing to Questioner...")
            return "questioner"
            
    return "advisor"


def build_workflow() -> Any:
    """
    Constructs, connects, attaches persistent SQLite checkpointer, and compiles 
    the LangGraph state machine workflow for the resilience assessment system.
    
    Returns:
        Any (CompiledStateGraph): The fully compiled, runnable, and persistent LangGraph application.
    """
    # 1. Initialize the state graph using the predefined AgentState schema
    workflow: StateGraph = StateGraph(AgentState)

    # 2. Register all agent functions as distinct nodes in the graph
    workflow.add_node("safety_classifier", safety_classifier_node)
    workflow.add_node("emergency_response", emergency_response_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("assessor", assessor_node)
    workflow.add_node("questioner", questioner_node)
    workflow.add_node("advisor", advisor_node)

    # 3. Entry Point: Route to Safety Gate first
    workflow.add_edge(START, "safety_classifier")

    # 4. Conditional Edge: Evaluate Safety
    workflow.add_conditional_edges(
        "safety_classifier",
        route_safety
    )

    # 5. Define standard execution pipeline (if safe)
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "assessor")

    # 6. Dynamic routing logic after assessment
    workflow.add_conditional_edges(
        "assessor",
        route_after_assessment
    )

    # 7. Connect terminal nodes to END
    workflow.add_edge("emergency_response", END)
    workflow.add_edge("questioner", END)
    workflow.add_edge("advisor", END)

    # 8. Ensure data directory exists and set up SQLite checkpointer connection
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    memory: SqliteSaver = SqliteSaver(conn)

    # 9. Compile the graph configuration into an executable application with memory persistence
    app: Any = workflow.compile(checkpointer=memory)
    
    return app
