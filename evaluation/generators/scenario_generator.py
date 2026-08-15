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
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "src" / "resilimind" / "assets" / "final_resilience_graph.json"

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
    """Generate deterministic synthetic evaluation scenarios."""

    VERSION = "1.0.0"

    def __init__(self, *, graph_path: Path = DEFAULT_GRAPH_PATH, seed: int = 42) -> None:
        self.graph_path = Path(graph_path)
        self.seed = seed
        self.rng = random.Random(seed)

        self.graph = self._load_graph()
        self.nodes = self._load_nodes()

        if not self.nodes:
            raise ValueError(f"No nodes found in graph: {self.graph_path}")

    def _load_graph(self) -> dict[str, Any]:
        """Load the ResiliMind knowledge graph from disk."""
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {self.graph_path}")

        with self.graph_path.open("r", encoding="utf-8") as file:
            graph = json.load(file)

        if "nodes" not in graph:
            raise ValueError("Knowledge graph must contain a top-level 'nodes' object")

        return graph

    def _load_nodes(self) -> dict[str, dict[str, Any]]:
        """Extract and validate node definitions from the knowledge graph."""
        nodes = self.graph["nodes"]
        if not isinstance(nodes, dict):
            raise ValueError("Graph 'nodes' must be a dictionary")
        return nodes

    def generate(
        self, count: int = 100, *, distribution: dict[str, int] | None = None
    ) -> list[EvaluationCase]:
        """Generate a deterministic collection of evaluation scenarios."""
        if count <= 0:
            raise ValueError("count must be greater than zero")

        distribution = distribution or self._scaled_distribution(count)
        if sum(distribution.values()) != count:
            raise ValueError("Distribution counts must sum to requested count")

        cases: list[EvaluationCase] = []
        index = 1

        for bucket, bucket_count in distribution.items():
            for _ in range(bucket_count):
                cases.append(self._generate_case(index=index, bucket=bucket))
                index += 1

        self.rng.shuffle(cases)
        return cases

    def _scaled_distribution(self, count: int) -> dict[str, int]:
        """Scale the default scenario distribution to a target size."""
        if count == 100:
            return dict(DEFAULT_DISTRIBUTION)

        keys = list(DEFAULT_DISTRIBUTION.keys())
        weights = list(DEFAULT_DISTRIBUTION.values())

        raw = [count * weight / 100 for weight in weights]
        floors = [int(value) for value in raw]
        remainder = count - sum(floors)

        fractional = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in fractional[:remainder]:
            floors[i] += 1

        return dict(zip(keys, floors))

    def _generate_case(self, *, index: int, bucket: str) -> EvaluationCase:
        """Generate a single evaluation case for a scenario bucket."""
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
            raise ValueError(f"Unknown scenario bucket: {bucket}") from exc

        domain = self._choose_domain(case_type=case_type)
        turn_count = self._choose_turn_count(case_type=case_type)

        scenario = ScenarioSpec(
            domain=domain,
            difficulty=difficulty,  # type: ignore[arg-type]
            case_type=case_type,  # type: ignore[arg-type]
            turn_count=turn_count,
        )

        safety = self._generate_safety(case_type=case_type)
        signals = self._generate_signals(domain=domain, case_type=case_type, safety=safety)
        assessments = self._generate_assessments(
            signals=signals, difficulty=difficulty, case_type=case_type
        )
        routing = self._generate_routing(
            safety=safety,
            difficulty=difficulty,
            case_type=case_type,
            assessments=assessments,
        )

        gold = EvaluationGold(
            safety=safety,
            extraction=GoldExtraction(active_signals=signals),
            assessment=GoldAssessmentOutput(assessments=assessments),
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

    def _choose_domain(self, *, case_type: str) -> str:
        """Select a domain from the knowledge graph."""
        domains = sorted({node["domain"] for node in self.nodes.values() if "domain" in node})
        if not domains:
            raise ValueError("No domains found in knowledge graph")
        return self.rng.choice(domains)

    def _choose_turn_count(self, *, case_type: str) -> int:
        """Determine the number of user turns for a scenario."""
        if case_type == "multi_domain":
            return self.rng.choice([2, 3])
        if case_type in {"adversarial", "ambiguous"}:
            return self.rng.choice([1, 2])
        return 1

    def _generate_safety(self, *, case_type: str) -> GoldSafety:
        """Generate the ground-truth safety annotation."""
        if case_type != "high_risk":
            return GoldSafety(is_high_risk=False, risk_category="SAFE")

        category = self.rng.choice(["SELF_HARM", "VIOLENCE", "SEVERE_ABUSE"])
        return GoldSafety(is_high_risk=True, risk_category=category)

    def _generate_signals(
        self, *, domain: str, case_type: str, safety: GoldSafety
    ) -> list[GoldSignal]:
        """Generate ground-truth resilience signals."""
        if safety.is_high_risk:
            return []

        candidates = [
            node_id for node_id, node in self.nodes.items() if node.get("domain") == domain
        ]
        if not candidates:
            raise ValueError(f"No graph nodes found for domain {domain}")

        if case_type == "multi_domain":
            other_domains = [
                d for d in {node["domain"] for node in self.nodes.values()} if d != domain
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

        selected = self.rng.sample(candidates, k=number_of_signals)
        signals: list[GoldSignal] = []

        for node_id in selected:
            if case_type == "mixed_signal":
                polarity = self.rng.choice(["positive", "negative", "mixed"])
            elif case_type in {"ambiguous", "adversarial"}:
                polarity = self.rng.choice(["negative", "mixed"])
            else:
                polarity = self.rng.choice(["positive", "negative"])

            signals.append(GoldSignal(node_id=node_id, detected_signal=polarity, evidence=None))

        return signals

    def _generate_assessments(
        self, *, signals: list[GoldSignal], difficulty: str, case_type: str
    ) -> list[GoldAssessment]:
        """Generate ground-truth assessments for detected signals."""
        assessments: list[GoldAssessment] = []
        for signal in signals:
            scores = self._generate_rubric(
                polarity=signal.detected_signal,
                difficulty=difficulty,
                case_type=case_type,
            )
            assessments.append(GoldAssessment(node_id=signal.node_id, rubric=scores))
        return assessments

    def _generate_rubric(
        self, *, polarity: str, difficulty: str, case_type: str
    ) -> AssessmentRubric:
        """Generate a four-dimensional resilience assessment."""
        if case_type == "high_risk":
            raise ValueError("High-risk cases must not generate assessments")

        if polarity == "positive":
            ranges = {k: (18, 25) for k in ("severity", "frequency", "functional", "coping")}
        elif polarity == "negative":
            ranges = {k: (3, 18) for k in ("severity", "frequency", "functional", "coping")}
        else:
            ranges = {k: (8, 20) for k in ("severity", "frequency", "functional", "coping")}

        values = {name: self.rng.randint(*v_range) for name, v_range in ranges.items()}

        if case_type in {"ambiguous", "adversarial"}:
            values = {
                k: max(5, min(22, v + self.rng.randint(-4, 4))) for k, v in values.items()
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
        """Generate the expected routing decision."""
        if safety.is_high_risk:
            return GoldRouting(expected_route="emergency_response", confidence_class="high")

        if case_type in {"ambiguous", "adversarial"} or not assessments or difficulty == "hard":
            return GoldRouting(expected_route="questioner", confidence_class="low")

        return GoldRouting(expected_route="advisor", confidence_class="high")


def write_jsonl(cases: list[EvaluationCase], output_path: Path) -> None:
    """Write evaluation cases to a JSON Lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case.model_dump(), ensure_ascii=False) + "\n")


def load_jsonl(input_path: Path) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSONL file."""
    cases: list[EvaluationCase] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvaluationCase.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
    return cases


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scenario generator."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic evaluation scenarios for ResiliMind."
    )
    parser.add_argument("--count", type=int, default=100, help="Number of scenarios to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--graph", type=Path, default=DEFAULT_GRAPH_PATH, help="Path to final_resilience_graph.json."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "scenarios.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, validate, and persist the evaluation dataset."""
    args = parse_args()

    generator = ScenarioGenerator(graph_path=args.graph, seed=args.seed)
    cases = generator.generate(count=args.count)

    validate_dataset(cases, valid_node_ids=set(generator.nodes.keys()))
    write_jsonl(cases, output_path=args.output)

    print(f"Generated {len(cases)} scenarios.")
    print(f"Dataset: {args.output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
