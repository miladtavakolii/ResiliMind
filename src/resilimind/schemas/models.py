from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field

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

class EvidenceScores(BaseModel):
    """
    4-dimensional evidence-based resilience rubric (0-25 each).
    Strictly contains only raw dimension scores for LLM generation.
    """
    severity: int = Field(
        ..., 
        ge=0, 
        le=25, 
        description="Absence of severity: 0 (Severe distress) to 25 (None/Mild distress)."
    )
    frequency: int = Field(
        ..., 
        ge=0, 
        le=25, 
        description="Absence of frequency: 0 (Constant/Chronic distress) to 25 (Rare/Isolated)."
    )
    functional: int = Field(
        ..., 
        ge=0, 
        le=25, 
        description="Functional preservation: 0 (Severe impairment) to 25 (Fully functional/Adapted)."
    )
    coping: int = Field(
        ..., 
        ge=0, 
        le=25, 
        description="Coping capacity: 0 (No Coping/Surrender) to 25 (Strong Coping Mechanisms)."
    )

    @property
    def total_score(self) -> int:
        """Calculates total score (0-100) deterministically in Python."""
        return self.severity + self.frequency + self.functional + self.coping

    @property
    def status(self) -> str:
        """Derives status color deterministically from total score in Python."""
        total = self.total_score
        if total >= 70:
            return "GREEN"
        elif total >= 40:
            return "YELLOW"
        return "RED"

class NodeAssessment(BaseModel):
    """
    Detailed evaluation of resilience status for a specific node.
    LLM only generates rubric scores, confidence, and reasoning.
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
    
    scores: EvidenceScores = Field(
        ..., 
        description="The 4-dimensional evidence rubric ratings."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the assessment between 0.0 and 1.0."
    )
    reasoning: str = Field(..., description="Psychological reasoning behind the assigned dimension scores.")

    @property
    def score(self) -> int:
        """Proxy property for total calculated score."""
        return self.scores.total_score

    @property
    def status(self) -> str:
        """Proxy property for derived status color."""
        return self.scores.status

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Overridden to inject computed 'score' and 'status' into dictionary outputs.
        Ensures 100% backward compatibility with DB persistence and Streamlit UI.
        """
        data: Dict[str, Any] = super().model_dump(*args, **kwargs)
        data["score"] = self.score
        data["status"] = self.status
        return data

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
