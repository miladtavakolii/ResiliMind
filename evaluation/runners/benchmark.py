from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import hashlib

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from evaluation.schemas import EvaluationCase, CasePrediction, TurnPrediction

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
        "safety_risk_category": "SAFE",
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
        safety_risk_category=final_state.get("safety_risk_category", "SAFE"),
        route=final_state.get("route"),
        active_nodes=list(final_state.get("active_nodes", [])),
        active_signals=list(final_state.get("active_signals", [])),
        subgraph_context=str(final_state.get("subgraph_context", "")),
        assessments=list(final_state.get("assessments", [])),
        requires_disambiguation=bool(final_state.get("requires_disambiguation", False)),
        final_response=str(final_state.get("final_response", "")),
        messages=serialize_messages(messages),
    )

def build_evaluation_user_id(run_id: str, case_id: str) -> int:
    """Build a deterministic isolated synthetic user ID.

    Args:
        run_id: Identifier for the current evaluation run.
        case_id: Unique benchmark case identifier.

    Returns:
        int: Deterministic pseudo-random integer user ID generated from the hash.
    """
    digest = hashlib.sha256(f"{run_id}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

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
    user_id = build_evaluation_user_id(run_id, case.case_id)
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


def configure_evaluation_environment(runtime_dir: Path) -> None:
    """Configure an isolated benchmark runtime environment directory.

    Args:
        runtime_dir: Target directory path for benchmark database and runtime files.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(runtime_dir.resolve())


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
