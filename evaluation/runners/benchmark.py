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
    EvaluationCase,
    CasePrediction,
    TurnPrediction,
    CaseEvaluationResult,
    EvaluationSummary,
)

from evaluation.evaluators.runner import (
    EvaluationRunner,
)

from evaluation.reporting.aggregator import (
    EvaluationAggregator,
)

from evaluation.reporting.error_analyzer import (
    ErrorAnalyzer,
)

from evaluation.reporting.failure_writer import (
    FailureCSVWriter,
)

from evaluation.reporting.final_report import (
    FinalReportGenerator,
)


from evaluation.evaluators import (
    SafetyEvaluator,
    ExtractionEvaluator,
    AssessmentEvaluator,
    RoutingEvaluator,
    ResponseEvaluator,
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



def load_cases(
    path: Path,
) -> list[EvaluationCase]:
    """
    Load evaluation cases from JSONL dataset.

    Args:
        path:
            Dataset JSONL file path.

    Returns:
        Validated evaluation cases.

    Raises:
        FileNotFoundError:
            If dataset does not exist.
        ValueError:
            If dataset contains invalid records.
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

                data = json.loads(
                    line
                )

                cases.append(
                    EvaluationCase.model_validate(
                        data
                    )
                )

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:

                raise ValueError(
                    f"Invalid case at line "
                    f"{line_number}: {exc}"
                ) from exc


    if not cases:
        raise ValueError(
            "Evaluation dataset is empty"
        )


    return cases



def serialize_message(
    message: BaseMessage,
) -> dict[str, Any]:
    """
    Convert LangChain message into JSON format.
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
    """
    Serialize workflow messages.
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
    """
    Build initial LangGraph state.
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
                content=user_message
            )
        ],
    }



def extract_turn_prediction(
    *,
    turn_index: int,
    user_message: str,
    final_state: dict[str, Any],
) -> TurnPrediction:
    """
    Extract workflow output into evaluation schema.
    """

    return TurnPrediction(

        turn_index=turn_index,

        user_message=user_message,

        safety_status=final_state.get(
            "safety_status"
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
            final_state.get(
                "messages",
                [],
            )
        ),
    )

def run_case(
    app: Any,
    case: EvaluationCase,
    *,
    case_number: int,
    total_cases: int,
) -> CasePrediction:
    """
    Execute one evaluation case through ResiliMind workflow.

    Each benchmark case receives an isolated LangGraph thread.

    Args:
        app:
            Compiled LangGraph workflow.

        case:
            Evaluation benchmark case.

        case_number:
            Current case index.

        total_cases:
            Total number of cases.

    Returns:
        Raw workflow prediction.
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
                "Running %s turn %d/%d",
                case.case_id,
                turn_index + 1,
                len(case.input.messages),
            )


            state = build_initial_state(
                user_id=user_id,
                user_message=user_message,
            )


            final_state = app.invoke(
                state,
                config=config,
            )


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
            final_response=(
                turns[-1].final_response
                if turns
                else ""
            ),
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
    """
    Write raw workflow predictions.
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



def write_json(
    data: Any,
    output_path: Path,
) -> None:
    """
    Write JSON evaluation artifact.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )



def load_prediction_mapping(
    predictions: list[CasePrediction],
) -> dict[str, dict[str, Any]]:
    """
    Convert predictions into evaluator input format.
    """

    return {
        prediction.case_id:
            prediction.model_dump()
        for prediction in predictions
    }



def write_case_results(
    results,
    output_path: Path,
) -> None:
    """
    Store per-case evaluation results.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )



def configure_evaluation_environment(
    runtime_dir: Path,
) -> None:
    """
    Configure isolated benchmark runtime.
    """

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    os.environ["DATA_DIR"] = str(
        runtime_dir.resolve()
    )



def build_evaluator_runner() -> EvaluationRunner:
    """
    Create evaluation metric pipeline.
    """

    return EvaluationRunner(
        evaluators=[
            SafetyEvaluator(),
            ExtractionEvaluator(),
            AssessmentEvaluator(),
            RoutingEvaluator(),
            ResponseEvaluator(),
        ]
    )



def run_evaluation_pipeline(
    *,
    cases: list[EvaluationCase],
    predictions: list[CasePrediction],
) -> None:
    """
    Execute metrics, aggregation and reporting.
    """

    logger.info(
        "Starting evaluation metrics..."
    )


    runner = build_evaluator_runner()


    prediction_map = load_prediction_mapping(
        predictions
    )


    results = runner.evaluate_dataset(
        cases,
        prediction_map,
    )


    summary = EvaluationAggregator().aggregate(
        results
    )


    failures = ErrorAnalyzer().analyze(
        results
    )


    write_case_results(
        results,
        DEFAULT_RESULTS_DIR
        /
        "case_results.jsonl",
    )


    write_json(
        summary,
        DEFAULT_RESULTS_DIR
        /
        "summary.json",
    )


    FailureCSVWriter().write(
        failures,
        DEFAULT_RESULTS_DIR
        /
        "failures.csv",
    )


    FinalReportGenerator().generate(
        summary,
        failures,
        DEFAULT_RESULTS_DIR
        /
        "evaluation_report.json",
    )


    logger.info(
        "Evaluation pipeline completed."
    )


def parse_args() -> argparse.Namespace:
    """
    Parse benchmark CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run ResiliMind evaluation benchmark."
        )
    )


    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIR
            /
            "predictions.jsonl"
        ),
    )


    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )


    return parser.parse_args()



def main() -> None:
    """
    Execute complete evaluation pipeline.
    """

    args = parse_args()


    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be positive"
        )


    configure_evaluation_environment(
        args.runtime_dir
    )


    from resilimind.core.database import init_db

    init_db()


    from resilimind.core.workflow import (
        build_workflow,
    )


    cases = load_cases(
        args.dataset
    )


    if args.limit:

        cases = cases[
            :args.limit
        ]



    app = build_workflow()


    predictions: list[CasePrediction] = []


    for index, case in enumerate(
        cases,
        start=1,
    ):

        predictions.append(
            run_case(
                app,
                case,
                case_number=index,
                total_cases=len(cases),
            )
        )



    write_predictions(
        predictions,
        args.output,
    )


    run_evaluation_pipeline(
        cases=cases,
        predictions=predictions,
    )


    successful = sum(
        item.successful
        for item in predictions
    )


    failed = (
        len(predictions)
        -
        successful
    )


    print(
        f"Executed: {len(predictions)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Predictions: {args.output}"
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
