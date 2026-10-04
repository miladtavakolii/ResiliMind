from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from pydantic import ValidationError

from evaluation.schemas import EvaluationCase, CasePrediction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "cases.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR
DEFAULT_PREDICTIONS_PATH = DEFAULT_RESULTS_DIR / "predictions.jsonl"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "evaluation" / "runtime"


def compute_case_fingerprint(case: EvaluationCase) -> str:
    """Compute a deterministic SHA-256 fingerprint for an evaluation case.

    Serializes the case's core components (identifier, dataset version, scenario
    metadata, gold annotations, and input conversation) into a canonical, sorted-key,
    compact JSON representation to generate a unique content hash.

    Args:
        case: EvaluationCase instance whose content will be fingerprinted.

    Returns:
        str: Hexadecimal SHA-256 digest uniquely identifying the case contents.
    """
    payload = {
        "case_id": case.case_id,
        "dataset_version": case.dataset_version,
        "scenario": case.scenario.model_dump(mode="json"),
        "gold": case.gold.model_dump(mode="json"),
        "input": case.input.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    logger.info("Loaded %d evaluation cases from %s", len(cases), path)
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


def write_predictions(predictions: list[CasePrediction], output_path: Path) -> None:
    """Write workflow predictions to a JSON Lines file.

    Args:
        predictions: List of CasePrediction instances to serialize.
        output_path: Destination file path for JSONL output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
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
