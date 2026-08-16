from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from evaluation.schemas import EvaluationCase, CasePrediction, TurnPrediction, CaseEvaluationResult, EvaluationSummary
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
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "evaluation" / "runtime"


def create_run_id() -> str:
    """Create a unique identifier for one benchmark execution."""
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

def load_cases(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSONL dataset file.

    Args:
        path: Path to the dataset JSONL file.

    Returns:
        List of parsed and validated EvaluationCase instances.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or if JSON validation fails.
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
                raise ValueError(f"Invalid case at line {line_number}: {exc}") from exc

    if not cases:
        raise ValueError("Evaluation dataset is empty")
    logger.info("Loaded %d evaluation cases from %s", len(cases), path)
    return cases


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message object into a JSON-compatible dictionary.

    Args:
        message: LangChain message instance.

    Returns:
        Dictionary containing serialized message fields.
    """
    return {
        "type": message.type,
        "content": message.content,
        "additional_kwargs": dict(message.additional_kwargs),
        "response_metadata": dict(message.response_metadata),
    }


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Serialize a list of LangChain workflow messages.

    Args:
        messages: List of LangChain message instances.

    Returns:
        List of serialized message dictionaries.
    """
    return [serialize_message(message) for message in messages]


def build_initial_state(*, user_id: int, user_message: str) -> dict[str, Any]:
    """Build the initial state dictionary for the LangGraph workflow.

    Args:
        user_id: Identifier for the current conversation thread/user.
        user_message: Input text prompt from the user.

    Returns:
        Dictionary representing the initial LangGraph workflow state.
    """
    return {
        "user_id": user_id,
        "user_message": user_message,
        "safety_status": None,
        "safety_flag": False,
        "active_nodes": [],
        "active_signals": [],
        "subgraph_context": "",
        "assessments": [],
        "requires_disambiguation": False,
        "final_response": "",
        "messages": [HumanMessage(content=user_message)],
    }


def extract_turn_prediction(
    *, turn_index: int, user_message: str, final_state: dict[str, Any]
) -> TurnPrediction:
    """Extract workflow state output into a TurnPrediction schema.

    Args:
        turn_index: Zero-based turn index within the conversation.
        user_message: User message for the current turn.
        final_state: Resulting state dictionary after LangGraph execution.

    Returns:
        Populated TurnPrediction instance.
    """
    messages = final_state.get("messages", [])

    return TurnPrediction(
        turn_index=turn_index,
        user_message=user_message,
        safety_status=final_state.get("safety_status"),
        safety_flag=bool(final_state.get("safety_flag", False)),
        active_nodes=list(final_state.get("active_nodes", [])),
        active_signals=list(final_state.get("active_signals", [])),
        subgraph_context=str(final_state.get("subgraph_context", "")),
        assessments=list(final_state.get("assessments", [])),
        requires_disambiguation=bool(final_state.get("requires_disambiguation", False)),
        final_response=str(final_state.get("final_response", "")),
        messages=serialize_messages(messages),
    )


def run_case(
    app: Any,
    case: EvaluationCase,
    *,
    case_number: int,
    total_cases: int,
    run_id: str,
) -> CasePrediction:
    """Execute one evaluation case across all conversation turns through the workflow.

    Args:
        app: Compiled LangGraph workflow application.
        case: Target EvaluationCase instance to execute.
        case_number: Sequence number of the current case.
        total_cases: Total number of cases being executed.
        run_id: if of current running benchmark

    Returns:
        CasePrediction containing executed turn predictions and execution status.
    """
    thread_id = f"evaluation: {run_id} : {case.dataset_version} : {case.case_id}"
    user_id = case_number
    config = {"configurable": {"thread_id": thread_id}}
    turns: list[TurnPrediction] = []

    logger.info("Running case %d/%d: %s", case_number, total_cases, case.case_id)

    try:
        for turn_index, user_message in enumerate(case.input.messages):
            logger.info("Running %s turn %d/%d", case.case_id, turn_index + 1, len(case.input.messages))
            
            state = build_initial_state(user_id=user_id, user_message=user_message)
            final_state = app.invoke(state, config=config)

            turns.append(
                extract_turn_prediction(
                    turn_index=turn_index,
                    user_message=user_message,
                    final_state=final_state,
                )
            )

        return CasePrediction(
            case_id=case.case_id,
            dataset_version=case.dataset_version,
            thread_id=thread_id,
            successful=True,
            turns=turns,
            final_response=turns[-1].final_response if turns else "",
        )

    except Exception as exc:
        logger.exception("Evaluation failed for case %s", case.case_id)
        return CasePrediction(
            case_id=case.case_id,
            dataset_version=case.dataset_version,
            thread_id=thread_id,
            successful=False,
            turns=turns,
            final_response=turns[-1].final_response if turns else "",
            error=str(exc),
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
        "raw": prediction.model_dump(),
    }


def write_predictions(predictions: list[CasePrediction], output_path: Path) -> None:
    """Write workflow predictions to a JSON Lines file.

    Args:
        predictions: List of CasePrediction instances to serialize.
        output_path: Destination file path for JSONL output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction.model_dump(), ensure_ascii=False) + "\n")


def write_json(data: Any, output_path: Path) -> None:
    """Write an evaluation artifact to a formatted JSON file.

    Args:
        data: Serializable data object or dictionary.
        output_path: Destination file path for JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prediction_mapping(predictions: list[CasePrediction]) -> dict[str, dict[str, Any]]:
    """Convert a list of CasePrediction objects into an ID-mapped dictionary for evaluators.

    Args:
        predictions: List of executed CasePrediction instances.

    Returns:
        Mapping from case IDs to serialized prediction dictionaries.
    """
    return {prediction.case_id: build_evaluator_prediction(prediction) for prediction in predictions}


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


def configure_evaluation_environment(runtime_dir: Path) -> None:
    """Configure an isolated benchmark runtime environment directory.

    Args:
        runtime_dir: Target directory path for benchmark database and runtime files.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(runtime_dir.resolve())


def build_evaluator_runner() -> EvaluationRunner:
    """Instantiate and configure the evaluation pipeline runner with registered evaluators.

    Returns:
        EvaluationRunner configured with all active evaluators.
    """
    judge = GeminiJudge(api_key=api_key, model=judge_model)
    return EvaluationRunner(
        evaluators=[
            SafetyEvaluator(),
            ExtractionEvaluator(),
            AssessmentEvaluator(),
            RoutingEvaluator(),
            ResponseEvaluator(judge=judge),
        ]
    )


def run_evaluation_pipeline(*, cases: list[EvaluationCase], predictions: list[CasePrediction], results_dir: Path) -> None:
    """Execute evaluation metrics calculation, aggregation, and artifact persistence.

    Args:
        cases: List of benchmark evaluation cases.
        predictions: List of workflow prediction results.
        results_dir: Path to save results
    """
    logger.info("Starting evaluation metrics...")

    runner = build_evaluator_runner()
    prediction_map = load_prediction_mapping(predictions)
    results = runner.evaluate_dataset(cases, prediction_map)
    summary = EvaluationAggregator().aggregate(results)
    failures = ErrorAnalyzer().analyze(results)

    write_case_results(results, results_dir / "case_results.jsonl")
    write_json(summary, results_dir / "summary.json")
    FailureCSVWriter().write(failures, results_dir / "failures.csv")
    FinalReportGenerator().generate(summary, failures, results_dir / "evaluation_report.json")

    logger.info("Evaluation pipeline completed.")


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments.

    Returns:
        Parsed CLI argument namespace.
    """
    parser = argparse.ArgumentParser(description="Run ResiliMind evaluation benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "predictions.jsonl")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end evaluation pipeline from CLI arguments."""
    args = parse_args()
    run_id = create_run_id()
    logger.info("Starting evaluation run: %s", run_id)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    configure_evaluation_environment(args.runtime_dir)

    from resilimind.core.database import init_db
    init_db()

    from resilimind.core.workflow import build_workflow

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]

    app = build_workflow()
    predictions: list[CasePrediction] = []

    for index, case in enumerate(cases, start=1):
        predictions.append(run_case(app, case, case_number=index, total_cases=len(cases), run_id=run_id))

    write_predictions(predictions, args.output)
    run_evaluation_pipeline(cases=cases, predictions=predictions, results_dir=args.output.parent)

    successful = sum(item.successful for item in predictions)
    failed = len(predictions) - successful

    print()
    print("=" * 60)
    print("ResiliMind Evaluation")
    print("=" * 60)
    print(f"Executed: {len(predictions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Predictions: {args.output}")
    print(f"Case results: {DEFAULT_RESULTS_DIR / 'case_results.jsonl'}")
    print(f"Summary: {DEFAULT_RESULTS_DIR / 'summary.json'}")
    print(f"Failures: {DEFAULT_RESULTS_DIR / 'failures.csv'}")
    print(f"Report: {DEFAULT_RESULTS_DIR / 'evaluation_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
