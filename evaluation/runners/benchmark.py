from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from evaluation.schemas import (
    CasePrediction,
    EvaluationCase,
    TurnPrediction,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "v1"
    / "cases.jsonl"
)

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

DEFAULT_RUNTIME_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "runtime"
)


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load validated evaluation cases from a JSONL dataset.

    Args:
        path: Path to the evaluation dataset.

    Returns:
        A list of validated evaluation cases.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If any record is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    cases: list[EvaluationCase] = []

    with path.open(
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
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid evaluation case at line "
                    f"{line_number}: {exc}"
                ) from exc

    if not cases:
        raise ValueError(
            f"No evaluation cases found in {path}"
        )

    return cases


def serialize_message(
    message: BaseMessage,
) -> dict[str, Any]:
    """Convert a LangChain message into JSON-serializable data.

    Args:
        message: LangChain message instance.

    Returns:
        A dictionary containing the message type, content, and metadata.
    """
    return {
        "type": message.type,
        "content": message.content,
        "additional_kwargs": dict(
            message.additional_kwargs
        ),
        "response_metadata": dict(
            message.response_metadata
        ),
    }


def serialize_messages(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """Serialize a collection of LangChain messages.

    Args:
        messages: Messages stored in the workflow state.

    Returns:
        JSON-serializable message dictionaries.
    """
    return [
        serialize_message(message)
        for message in messages
    ]


def build_initial_state(
    *,
    user_id: int,
    user_message: str,
) -> dict[str, Any]:
    """Build the initial LangGraph state for an evaluation turn.

    Args:
        user_id: Synthetic user identifier assigned to the benchmark case.
        user_message: Current user message.

    Returns:
        Initial state compatible with the ResiliMind workflow.
    """
    return {
        "user_id": user_id,
        "user_message": user_message,
        "safety_flag": False,
        "active_nodes": [],
        "active_signals": [],
        "subgraph_context": "",
        "assessments": [],
        "requires_disambiguation": False,
        "final_response": "",
        "messages": [
            HumanMessage(
                content=user_message,
            )
        ],
    }


def extract_turn_prediction(
    *,
    turn_index: int,
    user_message: str,
    final_state: dict[str, Any],
) -> TurnPrediction:
    """Extract a serializable raw prediction from workflow state.

    Args:
        turn_index: Zero-based conversation turn index.
        user_message: Input message used for this turn.
        final_state: Final LangGraph state returned by the workflow.

    Returns:
        Serializable raw prediction for the current turn.
    """
    messages = final_state.get(
        "messages",
        [],
    )

    return TurnPrediction(
        turn_index=turn_index,
        user_message=user_message,
        safety_status=final_state.get(
            "safety_status",
        ),
        safety_flag=bool(
            final_state.get(
                "safety_flag",
                False,
            )
        ),
        active_nodes=list(
            final_state.get(
                "active_nodes",
                [],
            )
        ),
        active_signals=list(
            final_state.get(
                "active_signals",
                [],
            )
        ),
        subgraph_context=str(
            final_state.get(
                "subgraph_context",
                "",
            )
        ),
        assessments=list(
            final_state.get(
                "assessments",
                [],
            )
        ),
        requires_disambiguation=bool(
            final_state.get(
                "requires_disambiguation",
                False,
            )
        ),
        final_response=str(
            final_state.get(
                "final_response",
                "",
            )
        ),
        messages=serialize_messages(
            messages,
        ),
    )


def run_case(
    app: Any,
    case: EvaluationCase,
    *,
    case_number: int,
    total_cases: int,
) -> CasePrediction:
    """Execute one evaluation case through the ResiliMind workflow.

    Each evaluation case receives a dedicated LangGraph thread. All turns
    belonging to the same case reuse that thread so that conversational state
    and checkpointed memory can influence later turns.

    Args:
        app: Compiled ResiliMind LangGraph application.
        case: Evaluation case to execute.
        case_number: One-based case number used for logging.
        total_cases: Total number of cases in the current run.

    Returns:
        Raw predictions produced by the workflow.
    """
    thread_id = (
        f"evaluation:"
        f"{case.dataset_version}:"
        f"{case.case_id}"
    )

    user_id = case_number

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    turns: list[TurnPrediction] = []

    logger.info(
        "Running case %d/%d: %s",
        case_number,
        total_cases,
        case.case_id,
    )

    try:
        for turn_index, user_message in enumerate(
            case.input.messages,
        ):
            logger.info(
                "Running case %s turn %d/%d",
                case.case_id,
                turn_index + 1,
                len(case.input.messages),
            )

            initial_state = build_initial_state(
                user_id=user_id,
                user_message=user_message,
            )

            final_state = app.invoke(
                initial_state,
                config=config,
            )

            turn_prediction = extract_turn_prediction(
                turn_index=turn_index,
                user_message=user_message,
                final_state=final_state,
            )

            turns.append(
                turn_prediction,
            )

        final_response = (
            turns[-1].final_response
            if turns
            else ""
        )

        return CasePrediction(
            case_id=case.case_id,
            dataset_version=case.dataset_version,
            thread_id=thread_id,
            successful=True,
            turns=turns,
            final_response=final_response,
        )

    except Exception as exc:
        logger.exception(
            "Evaluation failed for case %s",
            case.case_id,
        )

        return CasePrediction(
            case_id=case.case_id,
            dataset_version=case.dataset_version,
            thread_id=thread_id,
            successful=False,
            turns=turns,
            final_response=(
                turns[-1].final_response
                if turns
                else ""
            ),
            error=str(exc),
        )


def write_predictions(
    predictions: list[CasePrediction],
    output_path: Path,
) -> None:
    """Write raw benchmark predictions to JSONL.

    Args:
        predictions: Raw predictions generated by the benchmark runner.
        output_path: Destination JSONL file.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def configure_evaluation_environment(
    runtime_dir: Path,
) -> None:
    """Configure an isolated runtime directory for benchmark execution.

    The benchmark uses a separate data directory so that its LangGraph
    checkpoint database does not share runtime state with the interactive
    ResiliMind application.

    Args:
        runtime_dir: Directory used for evaluation runtime databases.
    """
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["DATA_DIR"] = str(
        runtime_dir.resolve()
    )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark execution.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the ResiliMind evaluation benchmark "
            "and store raw predictions."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the rendered evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the raw predictions JSONL file. "
            "Defaults to a timestamped file under evaluation/results."
        ),
    )

    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help="Isolated runtime directory for benchmark checkpoints.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N evaluation cases.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete benchmark pipeline and persist raw predictions."""
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero"
        )

    configure_evaluation_environment(
        args.runtime_dir,
    )

    from resilimind.core.database import init_db

    init_db()

    from resilimind.core.workflow import build_workflow

    cases = load_cases(
        args.dataset,
    )

    if args.limit is not None:
        cases = cases[:args.limit]

    if args.output is None:
        output_path = (
            DEFAULT_RESULTS_DIR
            / "predictions.jsonl"
        )
    else:
        output_path = args.output

    app = build_workflow()

    predictions: list[CasePrediction] = []

    for case_number, case in enumerate(
        cases,
        start=1,
    ):
        prediction = run_case(
            app,
            case,
            case_number=case_number,
            total_cases=len(cases),
        )

        predictions.append(
            prediction,
        )

    write_predictions(
        predictions,
        output_path,
    )

    successful = sum(
        prediction.successful
        for prediction in predictions
    )

    failed = len(predictions) - successful

    print(
        f"Executed {len(predictions)} cases."
    )
    print(
        f"Successful: {successful}"
    )
    print(
        f"Failed: {failed}"
    )
    print(
        f"Predictions: {output_path}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    main()
