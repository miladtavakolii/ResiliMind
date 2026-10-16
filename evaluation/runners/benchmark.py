from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from datetime import datetime, timezone
import hashlib

# Ensure project root is in sys.path when executed directly as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from evaluation.schemas import EvaluationCase, CasePrediction, TurnPrediction, BenchmarkRunManifest
from evaluation.runners.common import (
    PROJECT_ROOT,
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_LATEST_LINK,
    DEFAULT_RUNTIME_DIR,
    compute_case_fingerprint,
    load_cases,
    write_predictions,
    write_json,
    build_run_manifest,
    is_manifest_compatible,
    create_run_directory,
    update_latest_symlink,
    setup_run_logging,
    get_git_commit,
)

logger = logging.getLogger(__name__)


def create_run_id() -> str:
    """Create a unique identifier for one benchmark execution."""
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


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
        "safety_status": "UNAVAILABLE",
        "safety_flag": False,
        "safety_risk_category": "SAFE",
        "route": "service_unavailable",
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
        run_id: ID of current running benchmark

    Returns:
        CasePrediction containing executed turn predictions and execution status.
    """
    thread_id = f"evaluation: {run_id} : {case.dataset_version} : {case.case_id}"
    user_id = build_evaluation_user_id(run_id, case.case_id)
    config = {"configurable": {"thread_id": thread_id}}
    turns: list[TurnPrediction] = []

    logger.info("Running case %d/%d: %s", case_number, total_cases, case.case_id)
    
    from resilimind.core.database.connection import get_connection
    try:
        with get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)", (user_id, f"bench_{user_id}", "dummy"))
            conn.commit()
    except Exception as e:
        logger.error("Failed to insert dummy user for benchmark: %s", e)

    try:
        for turn_index, user_message in enumerate(case.input.messages):
            logger.info("Running %s turn %d/%d", case.case_id, turn_index + 1, len(case.input.messages))

            if turn_index == 0:
                state = build_initial_state(
                    user_id=user_id,
                    user_message=user_message,
                )
            else:
                state = {
                    "user_id": user_id,
                    "user_message": user_message,
                    "safety_status": "UNAVAILABLE",
                    "safety_flag": False,
                    "safety_risk_category": "SAFE",
                    "route": "service_unavailable",
                    "active_nodes": [],
                    "active_signals": [],
                    "subgraph_context": "",
                    "assessments": [],
                    "requires_disambiguation": False,
                    "final_response": "",
                    "messages": [HumanMessage(content=user_message)],
                }

            final_state = app.invoke(state, config=config)

            from resilimind.services.resilience_service import ResilienceService
            new_assessments = final_state.get("assessments", [])
            if new_assessments:
                ResilienceService.persist_assessments(user_id, new_assessments)

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
            case_fingerprint=compute_case_fingerprint(case),
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
            case_fingerprint=compute_case_fingerprint(case),
            successful=False,
            turns=turns,
            final_response=turns[-1].final_response if turns else "",
            error=str(exc),
        )


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
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to evaluation dataset (cases.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path for predictions.jsonl or run directory. If omitted, a dedicated timestamped run folder is created.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag/label appended to the run directory name (e.g. baseline, safety_patch).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Base directory where runs are stored (default: evaluation/results/runs).",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Directory for isolated benchmark database and runtime files (default: <run_dir>/runtime).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level override (default: read from LOG_LEVEL in .env).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end evaluation pipeline from CLI arguments."""
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    # Resolve run directory, output predictions path, manifest, and log file
    if args.output is None:
        run_dir = create_run_directory(base_dir=args.runs_dir, tag=args.tag)
        output_path = run_dir / "predictions.jsonl"
    elif args.output.suffix == ".jsonl":
        output_path = args.output
        run_dir = output_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = args.output
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "predictions.jsonl"

    runtime_dir = args.runtime_dir if args.runtime_dir is not None else (run_dir / "runtime")
    manifest_path = run_dir / "run_manifest.json"
    log_path = run_dir / "benchmark.log"

    # Configure dual-target logging (console + benchmark.log) with level from .env or CLI
    configured_logger = setup_run_logging(log_path, level=args.log_level)
    update_latest_symlink(run_dir)

    run_id = create_run_id()
    git_hash = get_git_commit()

    logger.info("=" * 60)
    logger.info("Starting ResiliMind Benchmark Run: %s", run_id)
    logger.info("Run directory: %s", run_dir)
    logger.info("Runtime dir:   %s", runtime_dir)
    logger.info("Git commit:    %s", git_hash)
    logger.info("Log file:      %s", log_path)
    logger.info("Log level:     %s", logging.getLevelName(configured_logger.level))
    logger.info("=" * 60)

    configure_evaluation_environment(runtime_dir)

    from resilimind.core.database import init_db
    init_db()

    from resilimind.core.workflow import build_workflow

    current_manifest = build_run_manifest(run_id, dataset_path=args.dataset)

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]

    processed_case_ids = set()
    successful = 0
    failed = 0

    can_resume = False
    if output_path.exists():
        if manifest_path.exists():
            try:
                prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                compatible, reason = is_manifest_compatible(prev_manifest, current_manifest)
                if compatible:
                    can_resume = True
                    logger.info("Found existing results with compatible manifest. Resuming from %s", output_path)
                else:
                    logger.warning(
                        "Cannot resume existing results from %s: %s. Starting fresh run.",
                        output_path,
                        reason,
                    )
            except Exception as exc:
                logger.warning("Error reading manifest %s: %s. Starting fresh run.", manifest_path, exc)
        else:
            logger.warning("No run_manifest.json found alongside existing %s. Starting fresh run.", output_path)

        if can_resume:
            with output_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            processed_case_ids.add(data["case_id"])
                            if data.get("successful"):
                                successful += 1
                            else:
                                failed += 1
                        except json.JSONDecodeError:
                            pass
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_suffix(f".backup_{timestamp}.jsonl")
            try:
                output_path.rename(backup_path)
                logger.info("Backed up incompatible previous results to %s", backup_path)
            except Exception as e:
                logger.warning("Could not backup %s: %s", output_path, e)

    # Persist the current benchmark run manifest
    write_json(current_manifest.model_dump(), manifest_path)

    cases_to_run = [c for c in cases if c.case_id not in processed_case_ids]

    if not cases_to_run:
        logger.info("All cases have already been processed.")
    else:
        logger.info("Executing %d remaining cases...", len(cases_to_run))

    app = build_workflow()

    for index, case in enumerate(cases_to_run, start=1):
        pred = run_case(app, case, case_number=index, total_cases=len(cases_to_run), run_id=run_id)
        
        write_predictions([pred], output_path)

        if pred.successful:
            successful += 1
        else:
            failed += 1

    total_executed = successful + failed

    print()
    print("=" * 60)
    print("ResiliMind Evaluation")
    print("=" * 60)
    print(f"Run ID:      {run_id}")
    print(f"Git Commit:  {git_hash}")
    print(f"Run Dir:     {run_dir}")
    print(f"Runtime Dir: {runtime_dir}")
    print(f"Executed:    {total_executed}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")
    print(f"Predictions: {output_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Log:         {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()