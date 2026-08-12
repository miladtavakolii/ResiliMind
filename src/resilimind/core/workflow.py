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
    Conditional edge router that determines whether to ask a clarifying question
    or to provide final psychological advice.
    """
    # 1. Check explicit flag set by Assessor
    if state.get("requires_disambiguation", False):
        return "questioner"
    
    # 2. Check individual confidence scores in assessments
    assessments = state.get("assessments", [])
    
    # If no nodes were extracted or active, ask for clarification
    if not assessments:
        return "questioner"

    # If any confidence score is below 0.70 threshold, force disambiguation
    for item in assessments:
        confidence = item.get("confidence", 1.0)
        if confidence < 0.70:
            print(f"Low confidence detected ({confidence}). Routing to Questioner...")
            return "questioner"
            
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
