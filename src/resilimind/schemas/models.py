from typing import List, Literal
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

class ActiveSignal(BaseModel):
    """
    Represents an individual node detected from user input.
    """
    node_id: str = Field(
        ..., 
        description="The unique identifier of the node (e.g., 'IND_PER_01', 'IND_ECO_01')."
    )
    detected_signal: Literal["positive", "negative", "mixed"] = Field(
        ..., 
        description="The emotional or situational polarity detected in the user's statement."
    )
    evidence: str = Field(
        ..., 
        description="Exact phrase or substring from user message supporting this signal."
    )

class ExtractionOutput(BaseModel):
    """
    Structured output schema for the Extractor Agent.
    """
    active_signals: List[ActiveSignal] = Field(
        default_factory=list,
        description="List of nodes activated by the user's input message."
    )

class NodeAssessment(BaseModel):
    """
    Detailed evaluation of resilience status for a specific node.
    """
    node_id: str = Field(..., description="The unique identifier of the node.")
    status: Literal["GREEN", "YELLOW", "RED"] = Field(
        ..., 
        description="Assessed resilience status: GREEN (High), YELLOW (Moderate), RED (Critical)."
    )
    category: Literal[
        "Personal_Resilience", 
        "Political_Resilience", 
        "Economic_Resilience", 
        "Physical_Resilience", 
        "Social_Resilience", 
        "Spiritual_Cultural_Resilience"
    ] = Field(description="The exact domain string of the node as provided in the graph context.")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the assessment between 0.0 and 1.0."
    )
    score: int = Field(ge=0, le=100, description="Exact numerical score representing resilience capacity (0 to 100)")
    reasoning: str = Field(..., description="Psychological reasoning behind this status assessment.")

    @model_validator(mode="after")
    def validate_score_status_alignment(self) -> Self:
        """
        Enforces strict numerical alignment between status and score thresholds.
        Raises ValueError if the LLM hallucinates inconsistent status/score combinations.
        """
        status = self.status
        score = self.score

        if status == "GREEN" and not (70 <= score <= 100):
            raise ValueError(f"Inconsistent assessment: Status is {status} but score is {score}. GREEN must be between 70 and 100.")
        elif status == "YELLOW" and not (40 <= score <= 69):
            raise ValueError(f"Inconsistent assessment: Status is {status} but score is {score}. YELLOW must be between 40 and 69.")
        elif status == "RED" and not (0 <= score <= 39):
            raise ValueError(f"Inconsistent assessment: Status is {status} but score is {score}. RED must be between 0 and 39.")
        
        # In Pydantic v2 mode="after", returning self is mandatory
        return self

class AssessmentOutput(BaseModel):
    """
    Structured output schema for the Assessor Agent.
    """
    assessments: List[NodeAssessment] = Field(
        default_factory=list, 
        description="List of status assessments for active nodes."
    )
    requires_disambiguation: bool = Field(
        default=False,
        description="Flag indicating if confidence is low or input is ambiguous requiring clarification."
    )
