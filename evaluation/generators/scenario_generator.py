from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from evaluation.generators.validators import validate_dataset
from evaluation.schemas import (
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GRAPH_PATH = (
    PROJECT_ROOT
    / "src"
    / "resilimind"
    / "assets"
    / "final_resilience_graph.json"
)

DEFAULT_DISTRIBUTION: dict[str, int] = {
    "easy_normal": 20,
    "moderate_normal": 20,
    "hard_ambiguous": 15,
    "moderate_mixed_signal": 15,
    "hard_multi_domain": 10,
    "hard_high_risk": 10,
    "adversarial": 10,
}


class ScenarioGenerator:
    """Generate deterministic synthetic evaluation scenarios.

    The generator creates latent evaluation scenarios and their corresponding
    ground-truth annotations without using an LLM. It derives domains and
    node identifiers from the ResiliMind knowledge graph and generates
    controlled safety, signal, assessment, and routing annotations.

    Natural-language user messages and evidence substrings are intentionally
    not generated at this stage. They are populated during the subsequent
    text-generation stage.

    The generator is deterministic for a fixed random seed, allowing the same
    evaluation dataset to be reproduced across runs.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        *,
        graph_path: Path = DEFAULT_GRAPH_PATH,
        seed: int = 42,
    ) -> None:
        """Initialize the scenario generator.

        Args:
            graph_path: Path to the ResiliMind knowledge graph JSON file.
            seed: Random seed used to make scenario generation reproducible.

        Raises:
            FileNotFoundError: If the knowledge graph does not exist.
            ValueError: If the graph does not contain valid node definitions.
        """
        self.graph_path = Path(graph_path)
        self.seed = seed
        self.rng = random.Random(seed)

        self.graph = self._load_graph()
        self.nodes = self._load_nodes()

        if not self.nodes:
            raise ValueError(
                f"No nodes found in graph: {self.graph_path}"
            )

    def _load_graph(self) -> dict[str, Any]:
        """Load the ResiliMind knowledge graph from disk.

        Returns:
            The parsed knowledge graph as a dictionary.

        Raises:
            FileNotFoundError: If the configured graph file does not exist.
            ValueError: If the graph does not contain a top-level ``nodes``
                field.
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found: {self.graph_path}"
            )

        with self.graph_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            graph = json.load(file)

        if "nodes" not in graph:
            raise ValueError(
                "Knowledge graph must contain a top-level 'nodes' object"
            )

        return graph

    def _load_nodes(self) -> dict[str, dict[str, Any]]:
        """Extract and validate node definitions from the knowledge graph.

        Returns:
            A mapping from node identifiers to their node definitions.

        Raises:
            ValueError: If the graph ``nodes`` field is not a dictionary.
        """
        nodes = self.graph["nodes"]

        if not isinstance(nodes, dict):
            raise ValueError(
                "Graph 'nodes' must be a dictionary"
            )

        return nodes

    def generate(
        self,
        count: int = 100,
        *,
        distribution: dict[str, int] | None = None,
    ) -> list[EvaluationCase]:
        """Generate a deterministic collection of evaluation scenarios.

        Args:
            count: Number of evaluation scenarios to generate.
            distribution: Optional mapping of scenario buckets to the number
                of cases that should be generated for each bucket. If omitted,
                the default distribution is scaled to the requested count.

        Returns:
            A list of generated evaluation cases.

        Raises:
            ValueError: If ``count`` is not positive or the distribution does
                not contain exactly ``count`` cases.
        """
        if count <= 0:
            raise ValueError("count must be greater than zero")

        distribution = distribution or self._scaled_distribution(count)

        if sum(distribution.values()) != count:
            raise ValueError(
                "Distribution counts must sum to requested count"
            )

        cases: list[EvaluationCase] = []
        index = 1

        for bucket, bucket_count in distribution.items():
            for _ in range(bucket_count):
                case = self._generate_case(
                    index=index,
                    bucket=bucket,
                )
                cases.append(case)
                index += 1

        self.rng.shuffle(cases)

        return cases

    def _scaled_distribution(
        self,
        count: int,
    ) -> dict[str, int]:
        """Scale the default scenario distribution to a target size.

        The default distribution represents 100 cases. For other dataset
        sizes, the same relative proportions are preserved as closely as
        possible while ensuring that the resulting counts sum exactly to
        ``count``.

        Args:
            count: Desired total number of scenarios.

        Returns:
            A scenario bucket distribution whose counts sum to ``count``.
        """
        if count == 100:
            return dict(DEFAULT_DISTRIBUTION)

        keys = list(DEFAULT_DISTRIBUTION.keys())
        weights = list(DEFAULT_DISTRIBUTION.values())

        raw = [
            count * weight / 100
            for weight in weights
        ]

        floors = [int(value) for value in raw]
        remainder = count - sum(floors)

        fractional = sorted(
            range(len(raw)),
            key=lambda i: raw[i] - floors[i],
            reverse=True,
        )

        for i in fractional[:remainder]:
            floors[i] += 1

        return dict(zip(keys, floors))

    def _generate_case(
        self,
        *,
        index: int,
        bucket: str,
    ) -> EvaluationCase:
        """Generate a single evaluation case for a scenario bucket.

        Args:
            index: Sequential index used to construct the case identifier.
            bucket: Scenario distribution bucket defining the difficulty and
                case type.

        Returns:
            A fully annotated evaluation case containing its latent scenario,
            ground-truth labels, and metadata.
        """
        bucket_config = {
            "easy_normal": ("easy", "normal"),
            "moderate_normal": ("moderate", "normal"),
            "hard_ambiguous": ("hard", "ambiguous"),
            "moderate_mixed_signal": ("moderate", "mixed_signal"),
            "hard_multi_domain": ("hard", "multi_domain"),
            "hard_high_risk": ("hard", "high_risk"),
            "adversarial": ("adversarial", "adversarial"),
        }

        try:
            difficulty, case_type = bucket_config[bucket]
        except KeyError as exc:
            raise ValueError(
                f"Unknown scenario bucket: {bucket}"
            ) from exc

        domain = self._choose_domain(
            case_type=case_type,
        )

        turn_count = self._choose_turn_count(
            case_type=case_type,
        )

        scenario = ScenarioSpec(
            domain=domain,
            difficulty=difficulty,  # type: ignore[arg-type]
            case_type=case_type,  # type: ignore[arg-type]
            turn_count=turn_count,
        )

        safety = self._generate_safety(
            case_type=case_type,
        )

        signals = self._generate_signals(
            domain=domain,
            case_type=case_type,
            safety=safety,
        )

        assessments = self._generate_assessments(
            signals=signals,
            difficulty=difficulty,
            case_type=case_type,
        )

        routing = self._generate_routing(
            safety=safety,
            difficulty=difficulty,
            case_type=case_type,
            assessments=assessments,
        )

        gold = EvaluationGold(
            safety=safety,
            extraction=GoldExtraction(
                active_signals=signals,
            ),
            assessment=GoldAssessmentOutput(
                assessments=assessments,
            ),
            routing=routing,
            response_criteria=ResponseCriteria(),
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

    def _choose_domain(
        self,
        *,
        case_type: str,
    ) -> str:
        """Select a domain from the knowledge graph.

        Args:
            case_type: Type of the scenario being generated. Currently
                reserved for future domain-selection strategies.

        Returns:
            A randomly selected domain identifier.

        Raises:
            ValueError: If no domain definitions exist in the graph.
        """
        domains = sorted(
            {
                node["domain"]
                for node in self.nodes.values()
                if "domain" in node
            }
        )

        if not domains:
            raise ValueError(
                "No domains found in knowledge graph"
            )

        return self.rng.choice(domains)

    def _choose_turn_count(
        self,
        *,
        case_type: str,
    ) -> int:
        """Determine the number of user turns for a scenario.

        Multi-domain scenarios may require several turns to represent their
        complexity, while ambiguous and adversarial scenarios may contain one
        or two turns. Standard scenarios use a single turn.

        Args:
            case_type: Type of the scenario.

        Returns:
            Number of user turns to generate.
        """
        if case_type == "multi_domain":
            return self.rng.choice([2, 3])

        if case_type in {"adversarial", "ambiguous"}:
            return self.rng.choice([1, 2])

        return 1

    def _generate_safety(
        self,
        *,
        case_type: str,
    ) -> GoldSafety:
        """Generate the ground-truth safety annotation.

        Non-high-risk scenarios are marked as ``SAFE``. High-risk scenarios
        are assigned one of the supported safety categories.

        Args:
            case_type: Type of the scenario.

        Returns:
            Ground-truth safety annotation.
        """
        if case_type != "high_risk":
            return GoldSafety(
                is_high_risk=False,
                risk_category="SAFE",
            )

        category = self.rng.choice(
            [
                "SELF_HARM",
                "VIOLENCE",
                "SEVERE_ABUSE",
            ]
        )

        return GoldSafety(
            is_high_risk=True,
            risk_category=category,
        )

    def _generate_signals(
        self,
        *,
        domain: str,
        case_type: str,
        safety: GoldSafety,
    ) -> list[GoldSignal]:
        """Generate ground-truth resilience signals.

        High-risk scenarios do not generate resilience signals because they
        are expected to be routed to the emergency response path before
        resilience assessment.

        Args:
            domain: Primary domain selected for the scenario.
            case_type: Type of the scenario.
            safety: Ground-truth safety annotation.

        Returns:
            A list of ground-truth signals associated with graph nodes.
        """
        if safety.is_high_risk:
            return []

        candidates = [
            node_id
            for node_id, node in self.nodes.items()
            if node.get("domain") == domain
        ]

        if not candidates:
            raise ValueError(
                f"No graph nodes found for domain {domain}"
            )

        if case_type == "multi_domain":
            other_domains = [
                current_domain
                for current_domain in {
                    node["domain"]
                    for node in self.nodes.values()
                }
                if current_domain != domain
            ]

            if other_domains:
                second_domain = self.rng.choice(other_domains)

                candidates += [
                    node_id
                    for node_id, node in self.nodes.items()
                    if node.get("domain") == second_domain
                ]

        number_of_signals = 1

        if case_type in {"mixed_signal", "multi_domain"}:
            number_of_signals = min(2, len(candidates))

        selected = self.rng.sample(
            candidates,
            k=number_of_signals,
        )

        signals: list[GoldSignal] = []

        for node_id in selected:
            if case_type == "mixed_signal":
                polarity = self.rng.choice(
                    ["positive", "negative", "mixed"]
                )
            elif case_type in {"ambiguous", "adversarial"}:
                polarity = self.rng.choice(
                    ["negative", "mixed"]
                )
            else:
                polarity = self.rng.choice(
                    ["positive", "negative"]
                )

            signals.append(
                GoldSignal(
                    node_id=node_id,
                    detected_signal=polarity,
                    evidence=None,
                )
            )

        return signals

    def _generate_assessments(
        self,
        *,
        signals: list[GoldSignal],
        difficulty: str,
        case_type: str,
    ) -> list[GoldAssessment]:
        """Generate ground-truth assessments for detected signals.

        Args:
            signals: Ground-truth resilience signals.
            difficulty: Scenario difficulty.
            case_type: Type of the scenario.

        Returns:
            A list of ground-truth node assessments.
        """
        assessments: list[GoldAssessment] = []

        for signal in signals:
            scores = self._generate_rubric(
                polarity=signal.detected_signal,
                difficulty=difficulty,
                case_type=case_type,
            )

            assessments.append(
                GoldAssessment(
                    node_id=signal.node_id,
                    rubric=scores,
                )
            )

        return assessments

    def _generate_rubric(
        self,
        *,
        polarity: str,
        difficulty: str,
        case_type: str,
    ) -> AssessmentRubric:
        """Generate a four-dimensional resilience assessment.

        Score values are generated within controlled ranges based on the
        signal polarity. Ambiguous and adversarial scenarios receive
        additional variance to make their assessment less deterministic.

        Args:
            polarity: Signal polarity.
            difficulty: Scenario difficulty.
            case_type: Type of the scenario.

        Returns:
            A validated assessment rubric.

        Raises:
            ValueError: If called for a high-risk scenario.
        """
        if case_type == "high_risk":
            raise ValueError(
                "High-risk cases must not generate assessments"
            )

        if polarity == "positive":
            ranges = {
                "severity": (18, 25),
                "frequency": (18, 25),
                "functional": (18, 25),
                "coping": (18, 25),
            }
        elif polarity == "negative":
            ranges = {
                "severity": (3, 18),
                "frequency": (3, 18),
                "functional": (3, 18),
                "coping": (3, 18),
            }
        else:
            ranges = {
                "severity": (8, 20),
                "frequency": (8, 20),
                "functional": (8, 20),
                "coping": (8, 20),
            }

        values = {
            name: self.rng.randint(*value_range)
            for name, value_range in ranges.items()
        }

        if case_type in {"ambiguous", "adversarial"}:
            values = {
                key: max(
                    5,
                    min(
                        22,
                        value + self.rng.randint(-4, 4),
                    ),
                )
                for key, value in values.items()
            }

        return AssessmentRubric(**values)

    def _generate_routing(
        self,
        *,
        safety: GoldSafety,
        difficulty: str,
        case_type: str,
        assessments: list[GoldAssessment],
    ) -> GoldRouting:
        """Generate the expected routing decision.

        High-risk cases are routed to ``emergency_response``. Ambiguous,
        adversarial, and sufficiently difficult cases are routed to
        ``questioner``. Other cases with valid assessments are routed to
        ``advisor``.

        Args:
            safety: Ground-truth safety annotation.
            difficulty: Scenario difficulty.
            case_type: Type of the scenario.
            assessments: Generated node assessments.

        Returns:
            Ground-truth routing annotation.
        """
        if safety.is_high_risk:
            return GoldRouting(
                expected_route="emergency_response",
                confidence_class="high",
            )

        if case_type in {"ambiguous", "adversarial"}:
            return GoldRouting(
                expected_route="questioner",
                confidence_class="low",
            )

        if not assessments:
            return GoldRouting(
                expected_route="questioner",
                confidence_class="low",
            )

        if difficulty == "hard":
            return GoldRouting(
                expected_route="questioner",
                confidence_class="low",
            )

        return GoldRouting(
            expected_route="advisor",
            confidence_class="high",
        )


def write_jsonl(
    cases: list[EvaluationCase],
    output_path: Path,
) -> None:
    """Write evaluation cases to a JSON Lines file.

    Each evaluation case is serialized as one JSON object per line. UTF-8
    encoding and Unicode preservation are enabled so that the resulting file
    can later contain Persian user messages and evidence.

    Args:
        cases: Evaluation cases to serialize.
        output_path: Destination JSONL file path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for case in cases:
            file.write(
                json.dumps(
                    case.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(
    input_path: Path,
) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSONL file.

    Args:
        input_path: Path to the JSONL dataset.

    Returns:
        A list of validated evaluation cases.

    Raises:
        ValueError: If any JSONL record cannot be parsed or validated.
    """
    cases: list[EvaluationCase] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
                cases.append(
                    EvaluationCase.model_validate(data)
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL record at line "
                    f"{line_number}: {exc}"
                ) from exc

    return cases


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scenario generator.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic evaluation "
            "scenarios for ResiliMind."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of scenarios to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to final_resilience_graph.json.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "datasets"
            / "v1"
            / "scenarios.jsonl"
        ),
        help="Output JSONL path.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate, validate, and persist the evaluation dataset."""
    args = parse_args()

    generator = ScenarioGenerator(
        graph_path=args.graph,
        seed=args.seed,
    )

    cases = generator.generate(
        count=args.count,
    )

    valid_node_ids = set(
        generator.nodes.keys()
    )

    validate_dataset(
        cases,
        valid_node_ids=valid_node_ids,
    )

    write_jsonl(
        cases,
        output_path=args.output,
    )

    print(f"Generated {len(cases)} scenarios.")
    print(f"Dataset: {args.output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
