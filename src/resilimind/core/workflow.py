import sqlite3
from pathlib import Path
from typing import Literal, Any, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import the shared state structure and individual agent node functions
from .state import AgentState
from .agents import (
    extractor_node,
    retriever_node,
    assessor_node,
    questioner_node,
    advisor_node
)

# Define path for SQLite persistent checkpoint database
CHECKPOINT_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "checkpoints.db"


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
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("assessor", assessor_node)
    workflow.add_node("questioner", questioner_node)
    workflow.add_node("advisor", advisor_node)

    # 3. Define the linear execution pipeline (Standard Edges)
    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "assessor")

    # 4. Define dynamic routing logic (Conditional Edges)
    workflow.add_conditional_edges(
        "assessor",
        route_after_assessment
    )

    # 5. Connect the final generation nodes to the termination point of the graph
    workflow.add_edge("questioner", END)
    workflow.add_edge("advisor", END)

    # 6. Ensure data directory exists and set up SQLite checkpointer connection
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    memory: SqliteSaver = SqliteSaver(conn)

    # 7. Compile the graph configuration into an executable application with memory persistence
    app: Any = workflow.compile(checkpointer=memory)
    
    return app
