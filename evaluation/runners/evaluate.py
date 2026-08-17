from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from collections.abc import Sequence

from evaluation.schemas import EvaluationCase, CasePrediction, CaseEvaluationResult, TurnPrediction
from evaluation.evaluators.runner import EvaluationRunner
from evaluation.reporting.aggregator import EvaluationAggregator
from evaluation.reporting.error_analyzer import ErrorAnalyzer
from evaluation.reporting.failure_writer import FailureCSVWriter
from evaluation.reporting.final_report import FinalReportGenerator
from evaluation.evaluators import SafetyEvaluator, ExtractionEvaluator, AssessmentEvaluator, RoutingEvaluator, ResponseEvaluator
from evaluation.judges.gemini import GeminiJudge

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "cases.jsonl"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "evaluation" / "results" / "predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSONL file.

    Args:
        path: Path to the evaluation dataset.

    Returns:
        List of validated evaluation cases.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty or contains an invalid record.
    """
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(EvaluationCase.model_validate(data))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid evaluation case at line {line_number}: {exc}"
                ) from exc

    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")

    return cases


def load_predictions(path: Path) -> list[CasePrediction]:
    """Load benchmark predictions from a JSONL file.

    Args:
        path: Path to the raw benchmark predictions.

    Returns:
        List of validated CasePrediction instances.

    Raises:
        FileNotFoundError: If the prediction file does not exist.
        ValueError: If the file is empty or contains an invalid record.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions: list[CasePrediction] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                predictions.append(CasePrediction.model_validate(data))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid prediction at line {line_number}: {exc}"
                ) from exc

    if not predictions:
        raise ValueError(f"No predictions found in {path}")

    return predictions


def validate_alignment(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> None:
    """Validate one-to-one alignment between cases and predictions.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.

    Raises:
        ValueError: If IDs are missing, duplicated, or unexpected.
    """
    case_ids = {case.case_id for case in cases}
    prediction_ids = [prediction.case_id for prediction in predictions]
    prediction_id_set = set(prediction_ids)

    if len(prediction_ids) != len(prediction_id_set):
        duplicates = sorted(
            case_id
            for case_id in prediction_id_set
            if prediction_ids.count(case_id) > 1
        )
        raise ValueError(f"Duplicate case IDs found in predictions: {duplicates}")

    missing_predictions = case_ids - prediction_id_set
    unexpected_predictions = prediction_id_set - case_ids

    if missing_predictions:
        raise ValueError(
            f"Missing predictions for cases: {sorted(missing_predictions)}"
        )
    if unexpected_predictions:
        raise ValueError(
            f"Predictions contain unknown case IDs: {sorted(unexpected_predictions)}"
        )


def validate_dataset_versions(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> None:
    """Ensure cases and predictions belong to the same dataset version.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.

    Raises:
        ValueError: If versions do not match.
    """
    case_versions = {case.dataset_version for case in cases}
    prediction_versions = {prediction.dataset_version for prediction in predictions}

    if len(case_versions) != 1:
        raise ValueError(
            f"Multiple dataset versions found in cases: {sorted(case_versions)}"
        )
    if len(prediction_versions) != 1:
        raise ValueError(
            f"Multiple dataset versions found in predictions: {sorted(prediction_versions)}"
        )

    case_version = next(iter(case_versions))
    prediction_version = next(iter(prediction_versions))

    if case_version != prediction_version:
        raise ValueError(
            f"Dataset version mismatch: cases={case_version}, predictions={prediction_version}"
        )

def build_evaluator_runner() -> EvaluationRunner:
    """Instantiate and configure the evaluation pipeline runner with registered evaluators.

    Returns:
        EvaluationRunner configured with all active evaluators.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for LLM-as-a-Judge.")
    model = os.getenv("GEMINI_MODEL","gemini-3.5-flash-lite")

    judge = GeminiJudge(api_key, model)

    return EvaluationRunner(
        evaluators=[
            SafetyEvaluator(),
            ExtractionEvaluator(),
            AssessmentEvaluator(),
            RoutingEvaluator(),
            ResponseEvaluator(judge=judge),
        ]
    )


def derive_route(turn: TurnPrediction) -> str:
    """Derive the actual workflow route from the final state.

    The workflow does not explicitly store a `routing` object,
    therefore evaluation derives the route using the same routing
    semantics implemented by workflow.py.

    Args:
        turn: TurnPrediction containing the final turn's state details.

    Returns:
        str: Derived route name ('emergency_response', 'questioner', or 'advisor').
    """
    if turn.safety_status == "HIGH_RISK":
        return "emergency_response"

    if turn.safety_status == "UNAVAILABLE" or turn.requires_disambiguation or not turn.assessments:
        return "questioner"

    for assessment in turn.assessments:
        confidence = assessment.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < 0.70:
            return "questioner"

    return "advisor"


def build_evaluator_prediction(prediction: CasePrediction) -> dict[str, Any]:
    """Adapt the raw CasePrediction structure to the input contract expected by evaluators.

    Args:
        prediction: CasePrediction object containing recorded conversation turns.

    Returns:
        dict[str, Any]: Structured dictionary formatted for evaluation modules.
    """
    if not prediction.turns:
        return {
            "safety": {"is_high_risk": False},
            "extraction": {"signals": []},
            "assessment": {"assessments": []},
            "routing": {"route": "unknown"},
            "advisor_response": "",
        }

    turn = prediction.turns[-1]
    is_high_risk = turn.safety_status == "HIGH_RISK" or turn.safety_flag

    return {
        "safety": {"is_high_risk": is_high_risk, "status": turn.safety_status},
        "extraction": {"signals": turn.active_signals, "active_nodes": turn.active_nodes},
        "assessment": {"assessments": turn.assessments},
        "routing": {"route": derive_route(turn)},
        "advisor_response": prediction.final_response,
        "user_context": "\n".join(item.user_message for item in prediction.turns),
        "raw": prediction.model_dump(),
    }


def build_prediction_mapping(
    predictions: list[CasePrediction],
) -> dict[str, dict[str, Any]]:
    """Build a case-ID keyed mapping for EvaluationRunner.

    Args:
        predictions: Raw benchmark predictions.

    Returns:
        Mapping from case IDs to normalized evaluator predictions.
    """
    return {
        prediction.case_id: build_evaluator_prediction(prediction)
        for prediction in predictions
    }


def evaluate_dataset(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> list[CaseEvaluationResult]:
    """Run all registered evaluators over the benchmark dataset.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.

    Returns:
        Per-case evaluation results.
    """
    runner = build_evaluator_runner()
    prediction_mapping = build_prediction_mapping(predictions)
    return runner.evaluate_dataset(cases=cases, predictions=prediction_mapping)


def write_case_results(results: list[CaseEvaluationResult], output_path: Path) -> None:
    """Store per-case evaluation results to a JSON Lines file.

    Args:
        results: Sequence of CaseEvaluationResult instances.
        output_path: Destination file path for JSONL output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def write_json(data: Any, output_path: Path) -> None:
    """Write an evaluation artifact to a formatted JSON file.

    Args:
        data: Serializable data object or dictionary.
        output_path: Destination file path for JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def build_execution_summary(
    predictions: Sequence[CasePrediction],
) -> dict[str, int]:
    """Build execution-level success statistics.

    Args:
        predictions: Benchmark predictions.

    Returns:
        Total, successful, and failed execution counts.
    """
    successful = sum(prediction.successful for prediction in predictions)
    return {
        "total": len(predictions),
        "successful": successful,
        "failed": len(predictions) - successful,
    }


def build_final_summary(
    summary: dict[str, Any],
    predictions: Sequence[CasePrediction],
) -> dict[str, Any]:
    """Add benchmark execution statistics to aggregated evaluation metrics.

    Args:
        summary: Dataset-level evaluator metrics.
        predictions: Raw benchmark predictions.

    Returns:
        Combined evaluation summary.
    """
    return {
        **summary,
        "execution": build_execution_summary(predictions),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone evaluation.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate previously generated ResiliMind benchmark predictions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the evaluation dataset (cases.jsonl).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to previously generated benchmark predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone evaluation over existing benchmark predictions."""
    args = parse_args()

    cases = load_cases(args.dataset)
    predictions = load_predictions(args.predictions)

    validate_alignment(cases, predictions)
    validate_dataset_versions(cases, predictions)

    logger.info(
        "Evaluating %d cases using predictions from %s",
        len(cases),
        args.predictions,
    )

    results = evaluate_dataset(cases=cases, predictions=predictions)

    summary = EvaluationAggregator().aggregate(results)
    summary = build_final_summary(summary, predictions)
    failures = ErrorAnalyzer().analyze(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_case_results(results, args.output_dir / "case_results.jsonl")
    write_json(summary, args.output_dir / "summary.json")
    FailureCSVWriter().write(failures, args.output_dir / "failures.csv")
    FinalReportGenerator().generate(
        summary,
        failures,
        args.output_dir / "evaluation_report.json",
    )

    successful_cases = summary["execution"]["successful"]
    failed_cases = summary["execution"]["failed"]

    print()
    print("=" * 60)
    print("ResiliMind Evaluation")
    print("=" * 60)
    print(f"Dataset:     {args.dataset}")
    print(f"Predictions: {args.predictions}")
    print(f"Cases:       {len(cases)}")
    print(f"Successful:  {successful_cases}")
    print(f"Failed:      {failed_cases}")
    print(f"Results:     {args.output_dir / 'case_results.jsonl'}")
    print(f"Summary:     {args.output_dir / 'summary.json'}")
    print(f"Failures:    {args.output_dir / 'failures.csv'}")
    print(f"Report:      {args.output_dir / 'evaluation_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
