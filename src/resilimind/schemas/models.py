from typing import List, Literal
from pydantic import BaseModel, Field, computed_field

class SafetyOutput(BaseModel):
    """
    Structured output schema for the Safety Classifier Agent.
    Acts as a zero-tolerance gatekeeper to detect high-risk user intents.
    """
    is_high_risk: bool = Field(
        ..., 
        description="True ONLY if the user explicitly expresses intent for self-harm, suicide, or severe violence."
    )
    risk_category: Literal["SAFE", "SELF_HARM", "VIOLENCE", "SEVERE_ABUSE"] = Field(
        ...,
        description="Categorization of the detected risk based on safety guidelines."
    )

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
    Detailed evaluation of resilience status for a specific node based on a 4-dimensional rubric.
    The LLM outputs ONLY the raw dimensions; status and total score are deterministically computed.
    """
    node_id: str = Field(..., description="The unique identifier of the node.")
    category: Literal[
        "Personal_Resilience", 
        "Political_Resilience", 
        "Economic_Resilience", 
        "Physical_Resilience", 
        "Social_Resilience", 
        "Spiritual_Cultural_Resilience"
    ] = Field(description="The exact domain string of the node as provided in the graph context.")
    
    # --- Evidence-Based Rubric Dimensions (0-25 each) ---
    severity_score: int = Field(
        description="Absence of severity: 0 (Severe/Overwhelming distress) to 25 (None/Mild distress).", 
        ge=0, le=25
    )
    frequency_score: int = Field(
        description="Absence of frequency: 0 (Constant/Chronic distress) to 25 (Rare/Isolated distress).", 
        ge=0, le=25
    )
    functional_score: int = Field(
        description="Functional preservation: 0 (Severe impairment) to 25 (Fully functional/Adapted).", 
        ge=0, le=25
    )
    coping_score: int = Field(
        description="Coping capacity: 0 (No Coping/Surrender) to 25 (Excellent Coping Mechanisms).", 
        ge=0, le=25
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the assessment between 0.0 and 1.0."
    )
    reasoning: str = Field(..., description="Psychological reasoning behind the assigned dimension scores.")

    @computed_field
    @property
    def score(self) -> int:
        """
        Deterministically aggregates the final resilience score (0-100) 
        from the four evidence dimensions.
        """
        return self.severity_score + self.frequency_score + self.functional_score + self.coping_score

    @computed_field
    @property
    def status(self) -> str:
        """
        Deterministically evaluates the color status based on the aggregated score.
        """
        total = self.score
        if total >= 70:
            return "GREEN"
        elif total >= 40:
            return "YELLOW"
        else:
            return "RED"

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
