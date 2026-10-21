from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any
from collections.abc import Sequence

from dotenv import load_dotenv

# Ensure project root is in sys.path when executed directly as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.schemas import EvaluationCase, CasePrediction, CaseEvaluationResult, TurnPrediction
from evaluation.evaluators.runner import EvaluationRunner
from evaluation.reporting.aggregator import EvaluationAggregator
from evaluation.reporting.error_analyzer import ErrorAnalyzer
from evaluation.reporting.failure_writer import FailureCSVWriter
from evaluation.reporting.final_report import FinalReportGenerator
from evaluation.evaluators import (
    SafetyEvaluator,
    ExtractionEvaluator,
    AssessmentEvaluator,
    RoutingEvaluator,
    ResponseEvaluator,
    ResponseJudge,
)

from evaluation.runners.common import (
    PROJECT_ROOT,
    DEFAULT_DATASET_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_LATEST_LINK,
    setup_run_logging,
    compute_case_fingerprint,
    load_cases,
    load_predictions,
)
from evaluation.runners.adapters import (
    validate_alignment,
    validate_dataset_versions,
    normalize_assessments,
    build_evaluator_prediction,
    build_prediction_mapping,
    get_turn_gold,
    build_turn_prediction,
)
from evaluation.runners.summary import (
    build_execution_summary,
    build_final_summary,
)

logger = logging.getLogger(__name__)

GEMINI_MAX_RETRIES = 5
GEMINI_RETRY_DELAY = 5
GEMINI_REQUEST_DELAY = 5


def build_evaluator_runner(
    max_retries: int,
    retry_delay: float,
    request_delay: float,
    include_response_eval: bool = True,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> EvaluationRunner:
    """Instantiate and configure the evaluation pipeline runner with registered evaluators.

    Args:
        max_retries: Maximum number of retry attempts upon failure.
        retry_delay: Base delay in seconds between retries.
        request_delay: Delay between two requests.
        include_response_eval: Whether to run ResponseEvaluator with LLM-as-a-Judge.
        judge_provider: Target LLM provider for the response judge.
        judge_model: Target model identifier for the response judge.

    Returns:
        EvaluationRunner configured with all active evaluators.
    """
    evaluators = [
        SafetyEvaluator(),
        ExtractionEvaluator(),
        AssessmentEvaluator(),
        RoutingEvaluator(),
    ]

    if include_response_eval:
        resolved_provider = judge_provider or os.getenv("RESILIMIND_LLM_PROVIDER", "gemini")
        resolved_model = judge_model or os.getenv("RESILIMIND_LLM_MODEL")
        resolved_api_key = os.getenv("RESILIMIND_LLM_API_KEY")

        judge = ResponseJudge(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_delay=request_delay,
        )
        evaluators.append(ResponseEvaluator(judge=judge))

    return EvaluationRunner(evaluators=evaluators)


def evaluate_dataset(
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
    max_retries: int,
    retry_delay: float,
    request_delay: float,
    output_path: Path | str | None = None,
    include_response_eval: bool = True,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> list[CaseEvaluationResult]:
    """Run all registered evaluators over the benchmark dataset.

    Args:
        cases: Ground-truth evaluation cases.
        predictions: Benchmark predictions.
        max_retries: Maximum number of retry attempts upon failure.
        retry_delay: Base delay in seconds between retries.
        request_delay: Delay between two requests.
        output_path: Path for saving results.
        include_response_eval: Whether to run ResponseEvaluator.
        judge_provider: Target LLM provider for the response judge.
        judge_model: Target model identifier for the response judge.

    Returns:
        Per-case evaluation results.
    """
    runner = build_evaluator_runner(
        max_retries=max_retries,
        retry_delay=retry_delay,
        request_delay=request_delay,
        include_response_eval=include_response_eval,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    prediction_map = build_prediction_mapping(predictions)
    return runner.evaluate_dataset(cases=cases, predictions=prediction_map, output_path=output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone evaluation.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate previously generated ResiliMind benchmark predictions."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a benchmark run directory (e.g. evaluation/results/runs/... or 'latest'). If specified, automatically sets --predictions and --output-dir.",
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=GEMINI_MAX_RETRIES,
        help="Maximum number of retry attempts upon failure.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=GEMINI_RETRY_DELAY,
        help="Base delay in seconds between retries.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=GEMINI_REQUEST_DELAY,
        help="Delay between two requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level override (default: read from LOG_LEVEL in .env).",
    )
    parser.add_argument(
        "--skip-response-eval",
        action="store_true",
        default=False,
        help="Skip LLM-based response evaluation (ResponseEvaluator).",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=os.getenv("RESILIMIND_LLM_PROVIDER", "gemini"),
        help="LLM provider for response evaluation (default: read from RESILIMIND_LLM_PROVIDER or 'gemini').",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.getenv("RESILIMIND_LLM_MODEL"),
        help="Model name for response evaluation (default: read from RESILIMIND_LLM_MODEL or provider default).",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone evaluation over existing benchmark predictions."""
    args = parse_args()
    load_dotenv()

    if args.run_dir:
        target_dir = args.run_dir
        if str(target_dir).lower() == "latest" or target_dir == Path("latest"):
            target_dir = DEFAULT_LATEST_LINK.resolve()
        elif not target_dir.is_absolute():
            # Check relative to runs dir or current dir
            if (DEFAULT_RUNS_DIR / target_dir).exists():
                target_dir = DEFAULT_RUNS_DIR / target_dir

        if not target_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

        args.predictions = target_dir / "predictions.jsonl"
        args.output_dir = target_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "evaluation.log"
    configured_logger = setup_run_logging(log_file, level=args.log_level)

    cases = load_cases(args.dataset)
    predictions = load_predictions(args.predictions)

    validate_alignment(cases, predictions)
    validate_dataset_versions(cases, predictions)

    logger.info("=" * 60)
    logger.info("Starting ResiliMind Evaluation")
    logger.info("Dataset:     %s", args.dataset)
    logger.info("Predictions: %s", args.predictions)
    logger.info("Output Dir:  %s", args.output_dir)
    logger.info("Log file:    %s", log_file)
    logger.info("Log level:   %s", logging.getLevelName(configured_logger.level))
    logger.info("Judge Prov:  %s", args.judge_provider)
    logger.info("Judge Model: %s", args.judge_model)
    logger.info("=" * 60)

    results_path = args.output_dir / "case_results.jsonl"
    results = evaluate_dataset(
        cases=cases,
        predictions=predictions,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
        output_path=str(results_path),
        include_response_eval=not args.skip_response_eval,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )

    summary = EvaluationAggregator().aggregate(results)
    summary = build_final_summary(summary, predictions)
    failures = ErrorAnalyzer().analyze(results)

    FailureCSVWriter().write(failures, args.output_dir / "failures.csv")
    FinalReportGenerator().generate(summary, failures, args.output_dir / "evaluation_report.json")

    successful_cases = summary["execution"]["successful"]
    failed_cases = summary["execution"]["failed"]

    print()
    print("=" * 60)
    print("ResiliMind Evaluation Complete")
    print("=" * 60)
    print(f"Dataset:     {args.dataset}")
    print(f"Predictions: {args.predictions}")
    print(f"Cases:       {len(cases)}")
    print(f"Successful:  {successful_cases}")
    print(f"Failed:      {failed_cases}")
    print(f"Results:     {args.output_dir / 'case_results.jsonl'}")
    print(f"Failures:    {args.output_dir / 'failures.csv'}")
    print(f"Report:      {args.output_dir / 'evaluation_report.json'}")
    print(f"Log:         {log_file}")
    print("=" * 60)


__all__ = [
    "compute_case_fingerprint",
    "load_cases",
    "load_predictions",
    "validate_alignment",
    "validate_dataset_versions",
    "build_evaluator_runner",
    "normalize_assessments",
    "build_evaluator_prediction",
    "build_prediction_mapping",
    "get_turn_gold",
    "build_turn_prediction",
    "evaluate_dataset",
    "build_execution_summary",
    "build_final_summary",
    "parse_args",
    "main",
    "GEMINI_MAX_RETRIES",
    "GEMINI_RETRY_DELAY",
    "GEMINI_REQUEST_DELAY",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_LATEST_LINK",
    "PROJECT_ROOT",
]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()