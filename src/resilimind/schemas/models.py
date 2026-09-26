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

NodeId = Literal[
    "IND_PER_01",
    "IND_PER_02",
    "IND_PER_03",
    "IND_POL_01",
    "IND_POL_02",
    "IND_ECO_01",
    "IND_ECO_02",
    "IND_PHY_01",
    "IND_PHY_02",
    "IND_SOC_01",
    "IND_SOC_02",
    "IND_SPI_01",
    "IND_SPI_02",
]

class ActiveSignal(BaseModel):
    """
    Represents an individual node detected from user input.
    """
    node_id: NodeId = Field(
        ...,
        description="Exact node ID from the knowledge graph. Do not modify, shorten, or invent node IDs."
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
    severity: Literal[8, 16, 22] = Field(
        ...,
        description="Allowed severity score: 8 (severe), 16 (moderate), 22 (mild/no distress)."
    )
    frequency: Literal[8, 16, 22] = Field(
        ...,
        description="Allowed frequency score: 8 (chronic), 16 (episodic), 22 (rare/isolated)."
    )
    functional: Literal[6, 12, 18, 24] = Field(
        ...,
        description="Allowed functional score: 6 (severe impairment), 12 (moderate), 18 (mild), 24 (fully functional)."
    )
    coping: Literal[8, 16, 24] = Field(
        ...,
        description="Allowed coping score: 8 (weak), 16 (moderate), 24 (strong/effective coping)."
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
        if total >= 40:
            return "YELLOW"
        return "RED"

class NodeAssessment(BaseModel):
    """
    Detailed evaluation of resilience status for a specific node.
    LLM only generates rubric scores, confidence, and reasoning.
    """
    node_id: NodeId = Field(
        ...,
        description="Exact node ID from the knowledge graph. Do not modify, shorten, or invent node IDs."
    )
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

class ProcessResult(BaseModel):
    """
    Represents the pure decoupled output of the resilience processing service.
    """
    final_response: str = Field(..., description="The textual response generated by the workflow.")
    new_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="Newly saved assessments.")
    state: Dict[str, Any] = Field(default_factory=dict, description="The complete final graph state dictionary.")
