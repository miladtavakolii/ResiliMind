from typing import Literal, Any
from langgraph.graph import StateGraph, START, END

# Import the shared state structure and individual agent nodes
from .state import AgentState
from .agents import (
    extractor_node,
    retriever_node,
    assessor_node,
    questioner_node,
    advisor_node
)


def route_after_assessment(state: AgentState) -> Literal["questioner", "advisor"]:
    """
    Conditional edge router that determines the next step after the resilience assessment.
    
    Args:
        state (AgentState): The current state dictionary of the workflow.
        
    Returns:
        Literal["questioner", "advisor"]: The exact string identifier of the next node to execute.
    """
    # Route to the 'questioner' node if the assessor flagged the input as ambiguous 
    # or if the confidence score fell below the required threshold.
    if state.get("requires_disambiguation", False):
        return "questioner"
    
    # Otherwise, route to the 'advisor' node to generate psychological interventions.
    return "advisor"


def build_workflow() -> Any:
    """
    Constructs, connects, and compiles the LangGraph state machine workflow 
    for the resilience assessment system.
    
    Returns:
        Any (CompiledStateGraph): The fully compiled and runnable LangGraph application.
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
    # LangGraph will automatically pass the state from 'assessor' to 'route_after_assessment'
    workflow.add_conditional_edges(
        "assessor",
        route_after_assessment
    )

    # 5. Connect the final generation nodes to the termination point of the graph
    workflow.add_edge("questioner", END)
    workflow.add_edge("advisor", END)

    # 6. Compile the graph configuration into an executable application
    app: Any = workflow.compile()
    
    return app
