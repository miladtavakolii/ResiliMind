from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.generators.validators import validate_dataset
from evaluation.schemas import EvaluationCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "src" / "resilimind" / "assets" / "final_resilience_graph.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "cases.jsonl"


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSONL file.

    Args:
        path: Path to the dataset JSONL file.

    Returns:
        list[EvaluationCase]: List of parsed and validated evaluation cases.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If any JSONL record fails validation or decoding.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvaluationCase.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid record at line {line_number}: {exc}") from exc

    return cases


def load_valid_node_ids(graph_path: Path) -> set[str]:
    """Load valid node identifiers from the ResiliMind knowledge graph.

    Args:
        graph_path: Path to the knowledge graph JSON file.

    Returns:
        set[str]: Set of valid node identifiers extracted from the graph.

    Raises:
        FileNotFoundError: If the knowledge graph file does not exist.
        ValueError: If the graph structure is invalid or lacks 'nodes'.
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {graph_path}")

    with graph_path.open("r", encoding="utf-8") as file:
        graph = json.load(file)

    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("Knowledge graph must contain a 'nodes' dictionary")

    return set(nodes)


def print_statistics(cases: list[EvaluationCase]) -> None:
    """Print basic dataset statistics across domains, difficulties, safety, and routes.

    Args:
        cases: List of evaluation cases to aggregate and display.
    """
    domains = Counter(case.scenario.domain for case in cases)
    difficulties = Counter(case.scenario.difficulty for case in cases)
    case_types = Counter(case.scenario.case_type for case in cases)
    safety_categories = Counter(case.gold.safety.risk_category for case in cases)
    routes = Counter(case.gold.routing.expected_route for case in cases)

    print("\nDataset Statistics")
    print("==================")
    print(f"Total cases: {len(cases)}")

    print("\nDomains:")
    for name, count in sorted(domains.items()):
        print(f"  {name}: {count}")

    print("\nDifficulties:")
    for name, count in sorted(difficulties.items()):
        print(f"  {name}: {count}")

    print("\nCase types:")
    for name, count in sorted(case_types.items()):
        print(f"  {name}: {count}")

    print("\nSafety:")
    for name, count in sorted(safety_categories.items()):
        print(f"  {name}: {count}")

    print("\nRoutes:")
    for name, count in sorted(routes.items()):
        print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse validation command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Validate a ResiliMind evaluation dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the dataset JSONL file.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH, help="Path to the ResiliMind knowledge graph.")
    return parser.parse_args()


def main() -> None:
    """Load, validate, and report statistics for an evaluation dataset."""
    args = parse_args()

    cases = load_cases(args.dataset)
    valid_node_ids = load_valid_node_ids(args.graph)

    validate_dataset(cases, valid_node_ids=valid_node_ids)

    print(f"Dataset validation passed: {len(cases)} cases")
    print_statistics(cases)


if __name__ == "__main__":
    main()
