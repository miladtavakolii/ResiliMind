from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Type aliases
SafetyCategory = Literal[
    "SAFE",
    "SELF_HARM",
    "VIOLENCE",
    "SEVERE_ABUSE",
]

Polarity = Literal[
    "positive",
    "negative",
    "mixed",
]

Status = Literal[
    "GREEN",
    "YELLOW",
    "RED",
]

Difficulty = Literal[
    "easy",
    "moderate",
    "hard",
    "adversarial",
]

CaseType = Literal[
    "normal",
    "ambiguous",
    "high_risk",
    "mixed_signal",
    "multi_domain",
    "adversarial",
]

Route = Literal[
    "emergency_response",
    "questioner",
    "advisor",
]

ConfidenceClass = Literal[
    "low",
    "high",
]


# Scenario
class ScenarioSpec(BaseModel):
    """
    Latent scenario specification.

    This is the ground-truth representation before converting the scenario
    into natural-language user messages.
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    difficulty: Difficulty
    case_type: CaseType

    # Number of user turns that will be generated in the next stage.
    turn_count: int = Field(default=1, ge=1, le=5)

    severity_level: Literal["low", "moderate", "high"]
    frequency_level: Literal["rare", "episodic", "chronic"]
    functional_level: Literal["none", "mild", "moderate", "severe"]
    coping_level: Literal["strong", "moderate", "weak"]


# Safety
class GoldSafety(BaseModel):
    """
    Ground-truth safety annotation.

    This follows the safety categories currently used by ResiliMind.
    """

    model_config = ConfigDict(extra="forbid")

    is_high_risk: bool
    risk_category: SafetyCategory


# Extraction
class GoldSignal(BaseModel):
    """
    Ground-truth signal.

    Evidence is intentionally optional at this stage because the scenario
    generator runs before natural-language text generation.

    Stage 3 will generate the actual user text and attach exact evidence
    substrings.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    detected_signal: Polarity

    # Filled during the text-generation stage.
    evidence: str | None = None
    evidence_message_index: int | None = Field(
        default=None,
        ge=0,
    )


class GoldExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_signals: list[GoldSignal] = Field(default_factory=list)


# Assessment
class AssessmentRubric(BaseModel):
    """
    Raw four-dimensional rubric.

    This exactly follows ResiliMind's current EvidenceScores schema:
    each dimension is in [0, 25].
    """

    model_config = ConfigDict(extra="forbid")

    severity: int = Field(ge=0, le=25)
    frequency: int = Field(ge=0, le=25)
    functional: int = Field(ge=0, le=25)
    coping: int = Field(ge=0, le=25)

    @property
    def total_score(self) -> int:
        return (
            self.severity
            + self.frequency
            + self.functional
            + self.coping
        )

    @property
    def status(self) -> Status:
        total = self.total_score

        if total >= 70:
            return "GREEN"

        if total >= 40:
            return "YELLOW"

        return "RED"


class GoldAssessment(BaseModel):
    """
    Ground-truth assessment for one resilience node.

    total_score and status are derived rather than independently generated.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    rubric: AssessmentRubric

    @property
    def total_score(self) -> int:
        return self.rubric.total_score

    @property
    def status(self) -> Status:
        return self.rubric.status


class GoldAssessmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[GoldAssessment] = Field(default_factory=list)


# Routing
class GoldRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_route: Route
    confidence_class: ConfidenceClass


# Response evaluation criteria
class ResponseCriteria(BaseModel):
    """
    Criteria for the later LLM-as-a-Judge stage.

    We deliberately do not store a single "gold response".
    """

    model_config = ConfigDict(extra="forbid")

    must_be_safe: bool = True
    must_be_grounded: bool = True
    must_not_diagnose: bool = True
    should_be_relevant: bool = True
    should_be_actionable: bool = True


# Full gold specification
class EvaluationGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safety: GoldSafety
    extraction: GoldExtraction
    assessment: GoldAssessmentOutput
    routing: GoldRouting
    response_criteria: ResponseCriteria = Field(
        default_factory=ResponseCriteria
    )


# Input
class EvaluationInput(BaseModel):
    """
    User input presented to ResiliMind.

    Empty at scenario-generation time and populated in Stage 3.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[str] = Field(default_factory=list)


# Metadata
class EvaluationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: str
    generator_version: str
    seed: int
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        generator_version: str = "1.0.0",
    ) -> "EvaluationMetadata":
        return cls(
            generator="deterministic_scenario_generator",
            generator_version=generator_version,
            seed=seed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


# Evaluation case
class EvaluationCase(BaseModel):
    """
    Complete evaluation case.

    Stage 2 creates the scenario + gold specification.
    Stage 3 fills input.messages.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_version: str

    scenario: ScenarioSpec
    input: EvaluationInput = Field(default_factory=EvaluationInput)
    gold: EvaluationGold
    metadata: EvaluationMetadata

    @model_validator(mode="after")
    def validate_case(self) -> "EvaluationCase":
        if self.scenario.turn_count != len(self.input.messages):
            # During Stage 2 messages are intentionally empty.
            if self.input.messages:
                raise ValueError(
                    "turn_count must match the number of input messages"
                )

        return self

class TurnPrediction(BaseModel):
    """Store the raw prediction produced for a single conversation turn."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    user_message: str
    safety_status: str | None = None
    safety_flag: bool = False
    safety_risk_category: str = "SAFE"
    route: str | None = None
    active_nodes: list[str] = Field(default_factory=list)
    active_signals: list[dict[str, Any]] = Field(default_factory=list)
    subgraph_context: str = ""
    assessments: list[dict[str, Any]] = Field(default_factory=list)
    requires_disambiguation: bool = False
    final_response: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class CasePrediction(BaseModel):
    """Store the complete raw prediction for one evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_version: str
    thread_id: str
    successful: bool
    turns: list[TurnPrediction] = Field(default_factory=list)
    final_response: str = ""
    error: str | None = None


class CaseEvaluationResult(BaseModel):
    """
    Stores evaluation results for a single benchmark case.

    Each evaluator contributes its own metric namespace.
    """

    case_id: str

    metrics: dict[str, Any] = Field(
        default_factory=dict
    )


class EvaluationSummary(BaseModel):
    """
    Aggregated benchmark evaluation report.

    Contains metrics calculated across the complete dataset.
    """

    dataset_size: int

    evaluators: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )


class ResponseJudgeResult(BaseModel):
    """
    Structured output returned by the LLM response judge.
    """

    model_config = ConfigDict(extra="forbid")

    empathy: float = Field(ge=1.0, le=10.0)
    relevance: float = Field(ge=1.0, le=10.0)
    safety: float = Field(ge=1.0, le=10.0)
    actionability: float = Field(ge=1.0, le=10.0)
    consistency: float = Field(ge=1.0, le=10.0)
    hallucination: float = Field(ge=1.0, le=10.0)
    overall: float = Field(ge=1.0, le=10.0)
    reason: str
