from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import random
from typing import Any

from evaluation.schemas import (
    AssessmentProfile,
    AssessmentRubric,
    EvaluationCase,
    EvaluationGold,
    EvaluationInput,
    EvaluationMetadata,
    GoldAssessment,
    GoldAssessmentOutput,
    GoldExtraction,
    GoldRouting,
    GoldSafety,
    GoldSignal,
    ResponseCriteria,
    ScenarioSpec,
)
from .sampler import (
    generate_rubric_from_profile,
    sample_assessment_profile,
)
from .gold import (
    generate_safety,
    generate_signals,
    generate_assessments,
    generate_routing,
    generate_response_criteria,
)
from .strategy import (
    DEFAULT_DISTRIBUTION,
    BUCKET_CONFIG,
    scale_distribution,
    choose_domain,
    choose_turn_count,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_PATH: Path = (
    PROJECT_ROOT / "src" / "resilimind" / "assets" / "final_resilience_graph.json"
)


class ScenarioGenerator:
    """Generate deterministic synthetic evaluation scenarios for ResiliMind.

    Attributes:
        VERSION (str): Generator version string.
        graph_path (Path): Path to the resilience knowledge graph JSON file.
        seed (int): Random seed used for deterministic generation.
        rng (random.Random): Dedicated random number generator instance.
        graph (dict[str, Any]): Loaded knowledge graph data.
        nodes (dict[str, dict[str, Any]]): Extracted node mapping from the graph.
    """

    VERSION: str = "1.0.0"

    def __init__(self, *, graph_path: Path = DEFAULT_GRAPH_PATH, seed: int = 42) -> None:
        """Initialize the scenario generator.

        Args:
            graph_path: Path to the knowledge graph JSON asset.
            seed: Seed value for deterministic random generation.

        Raises:
            FileNotFoundError: If the graph file does not exist.
            ValueError: If the graph format is invalid or contains no nodes.
        """
        self.graph_path = Path(graph_path)
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile_counts: Counter[tuple[str, str, str, str, str]] = Counter()

        self.graph = self._load_graph()
        self.nodes = self._load_nodes()

        if not self.nodes:
            raise ValueError(f"No nodes found in graph: {self.graph_path}")

    def _load_graph(self) -> dict[str, Any]:
        """Load the ResiliMind knowledge graph from disk.

        Returns:
            dict[str, Any]: Parsed knowledge graph dictionary.

        Raises:
            FileNotFoundError: If the graph file cannot be located.
            ValueError: If the top-level 'nodes' key is missing.
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {self.graph_path}")

        with self.graph_path.open("r", encoding="utf-8") as file:
            graph = json.load(file)

        if "nodes" not in graph:
            raise ValueError("Knowledge graph must contain a top-level 'nodes' object")

        return graph

    def _load_nodes(self) -> dict[str, dict[str, Any]]:
        """Extract and validate node definitions from the knowledge graph.

        Returns:
            dict[str, dict[str, Any]]: Mapping of node IDs to their attribute dictionaries.

        Raises:
            ValueError: If the 'nodes' element is not a dictionary.
        """
        nodes = self.graph["nodes"]
        if not isinstance(nodes, dict):
            raise ValueError("Graph 'nodes' must be a dictionary")
        return nodes

    def generate(
        self, count: int = 100, *, distribution: dict[str, int] | None = None
    ) -> list[EvaluationCase]:
        """Generate a deterministic collection of evaluation scenarios.

        Args:
            count: Total number of evaluation scenarios to create.
            distribution: Custom distribution mapping bucket names to target counts.
                If None, scales DEFAULT_DISTRIBUTION to the requested count.

        Returns:
            list[EvaluationCase]: Generated and shuffled list of evaluation cases.

        Raises:
            ValueError: If count is non-positive or distribution counts do not sum to count.
        """
        if count <= 0:
            raise ValueError("count must be greater than zero")

        distribution = distribution or self._scaled_distribution(count)
        if sum(distribution.values()) != count:
            raise ValueError("Distribution counts must sum to requested count")

        cases = []
        index = 1

        for bucket, bucket_count in distribution.items():
            for _ in range(bucket_count):
                cases.append(self._generate_case(index=index, bucket=bucket))
                index += 1

        self.rng.shuffle(cases)
        return cases

    def _scaled_distribution(self, count: int) -> dict[str, int]:
        """Scale the default scenario distribution to a target size using largest remainder."""
        return scale_distribution(count, DEFAULT_DISTRIBUTION)

    def _choose_domain(self, *, case_type: str) -> str:
        """Select a domain randomly from available graph nodes."""
        return choose_domain(self.rng, self.nodes, case_type=case_type)

    def _choose_turn_count(self, *, case_type: str) -> int:
        """Determine the number of user turns based on the case type."""
        return choose_turn_count(self.rng, case_type=case_type)

    def _sample_assessment_profile(
        self, *, difficulty: str, case_type: str, polarity: str
    ) -> AssessmentProfile:
        """Sample independent latent assessment dimensions."""
        return sample_assessment_profile(
            self.rng,
            self.profile_counts,
            difficulty=difficulty,
            case_type=case_type,
            polarity=polarity,
        )

    def _generate_safety(self, *, case_type: str) -> GoldSafety:
        """Generate ground-truth safety annotations."""
        return generate_safety(self.rng, case_type=case_type)

    def _generate_signals(
        self, *, domain: str, case_type: str, safety: GoldSafety
    ) -> list[GoldSignal]:
        """Generate ground-truth resilience signals based on domain and case type."""
        return generate_signals(
            self.rng,
            self.nodes,
            domain=domain,
            case_type=case_type,
            safety=safety,
        )

    def _generate_rubric_from_profile(
        self, profile: AssessmentProfile
    ) -> AssessmentRubric:
        """Map qualitative latent assessment dimensions to quantitative rubric scores."""
        return generate_rubric_from_profile(profile)

    def _generate_assessments(
        self, *, signals: Sequence[GoldSignal], scenario: ScenarioSpec
    ) -> list[GoldAssessment]:
        """Generate ground-truth rubric assessments for detected signals."""
        return generate_assessments(signals=signals, scenario=scenario)

    def _generate_routing(
        self,
        *,
        safety: GoldSafety,
        difficulty: str,
        case_type: str,
        assessments: Sequence[GoldAssessment],
    ) -> GoldRouting:
        """Generate the expected workflow routing decision."""
        return generate_routing(
            safety=safety,
            difficulty=difficulty,
            case_type=case_type,
            assessments=assessments,
        )

    def _generate_response_criteria(
        self,
        *,
        scenario: ScenarioSpec,
        safety: GoldSafety,
        signals: Sequence[GoldSignal],
        route: str,
    ) -> ResponseCriteria:
        """Generate scenario-specific response criteria for LLM Judge."""
        return generate_response_criteria(
            scenario=scenario,
            safety=safety,
            signals=signals,
            route=route,
        )

    def _generate_case(self, *, index: int, bucket: str) -> EvaluationCase:
        """Generate a single evaluation case for a specific scenario bucket.

        Args:
            index: Sequential integer index for scenario ID naming.
            bucket: Bucket identifier determining difficulty and case type.

        Returns:
            EvaluationCase: Fully constructed benchmark case with gold annotations.

        Raises:
            ValueError: If the bucket name is unrecognized.
        """
        try:
            difficulty, case_type = BUCKET_CONFIG[bucket]
        except KeyError as exc:
            raise ValueError(f"Unknown scenario bucket: {bucket}") from exc

        domain = self._choose_domain(case_type=case_type)
        turn_count = self._choose_turn_count(case_type=case_type)
        safety = self._generate_safety(case_type=case_type)

        signals = self._generate_signals(
            domain=domain,
            case_type=case_type,
            safety=safety,
        )

        assessment_profiles = {
            signal.node_id: self._sample_assessment_profile(
                difficulty=difficulty,
                case_type=case_type,
                polarity=signal.detected_signal,
            )
            for signal in signals
        }

        scenario = ScenarioSpec(
            domain=domain,
            difficulty=difficulty,
            case_type=case_type,
            turn_count=turn_count,
            severity_level="moderate",
            frequency_level="episodic",
            functional_level="mild",
            coping_level="moderate",
            assessment_profiles=assessment_profiles,
        )

        assessments = self._generate_assessments(
            signals=signals,
            scenario=scenario,
        )

        routing = self._generate_routing(
            safety=safety,
            difficulty=difficulty,
            case_type=case_type,
            assessments=assessments,
        )

        response_criteria = self._generate_response_criteria(
            scenario=scenario,
            safety=safety,
            signals=signals,
            route=routing.expected_route,
        )

        gold = EvaluationGold(
            safety=safety,
            extraction=GoldExtraction(active_signals=signals),
            assessment=GoldAssessmentOutput(assessments=assessments),
            routing=routing,
            response_criteria=response_criteria,
        )

        return EvaluationCase(
            case_id=f"RM-SC-{index:04d}",
            dataset_version="v1",
            scenario=scenario,
            input=EvaluationInput(messages=[]),
            gold=gold,
            metadata=EvaluationMetadata.create(
                seed=self.seed,
                generator_version=self.VERSION,
            ),
        )
