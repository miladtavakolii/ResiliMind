from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from evaluation.generators.validators import validate_dataset
from evaluation.schemas import EvaluationCase
from .engine import (
    DEFAULT_GRAPH_PATH,
    PROJECT_ROOT,
    ScenarioGenerator,
)


def write_jsonl(cases: Sequence[EvaluationCase], output_path: Path) -> None:
    """Write evaluation cases to a JSON Lines file.

    Args:
        cases: Sequence of evaluation cases to serialize.
        output_path: Target file path on disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case.model_dump(), ensure_ascii=False) + "\n")


def load_jsonl(input_path: Path) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSONL file.

    Args:
        input_path: Path to the JSONL dataset file.

    Returns:
        list[EvaluationCase]: List of parsed and validated evaluation cases.

    Raises:
        ValueError: If a record fails schema validation or JSON decoding.
    """
    cases = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvaluationCase.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: {exc}"
                ) from exc
    return cases


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scenario generator.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic evaluation scenarios for ResiliMind."
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
        default=PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "scenarios.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, validate, and persist the evaluation dataset."""
    args = parse_args()

    generator = ScenarioGenerator(graph_path=args.graph, seed=args.seed)
    cases = generator.generate(count=args.count)

    validate_dataset(
        cases,
        valid_node_ids=set(generator.nodes.keys()),
        require_rendered=False,
    )
    write_jsonl(cases, output_path=args.output)

    print(f"Generated {len(cases)} scenarios.")
    print(f"Dataset: {args.output}")
    print(f"Seed: {args.seed}")
