from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any
import urllib.request
from collections.abc import Sequence

from pydantic import ValidationError

from evaluation.schemas import EvaluationCase, CasePrediction, BenchmarkRunManifest

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


def get_git_commit() -> str:
    """Retrieve the current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_prompt_hashes() -> dict[str, str]:
    """Compute SHA-256 hashes of all system and evaluation prompt files."""
    hashes: dict[str, str] = {}
    prompt_dirs = [
        PROJECT_ROOT / "src" / "resilimind" / "assets" / "prompts",
        PROJECT_ROOT / "evaluation" / "prompts",
    ]
    for p_dir in prompt_dirs:
        if p_dir.exists():
            for p_file in sorted(p_dir.glob("*.txt")):
                hashes[p_file.name] = compute_file_sha256(p_file)
    return hashes


def get_model_digest(model_name: str, base_url: str = "http://localhost:11434") -> str:
    """Query Ollama API to retrieve exact model digest hash."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name", "")
                if name == model_name or name.startswith(f"{model_name}:") or model_name.startswith(f"{name}:"):
                    return m.get("digest", "")
                if m.get("model") == model_name:
                    return m.get("digest", "")
    except Exception:
        pass
    return ""


def build_run_manifest(
    run_id: str,
    *,
    dataset_path: Path,
    judge_model: str = "gemini-3.5-flash-lite",
) -> BenchmarkRunManifest:
    """Construct a BenchmarkRunManifest from the real active environment.

    Args:
        run_id: Unique benchmark run identifier.
        dataset_path: Path to evaluation dataset.
        judge_model: Identifier of the judge LLM model.

    Returns:
        BenchmarkRunManifest: Populated manifest capturing code, model, dataset, and prompt hashes.
    """
    from resilimind.core.config import settings

    git_hash = get_git_commit()
    graph_path = PROJECT_ROOT / "src" / "resilimind" / "assets" / "final_resilience_graph.json"

    model_name = settings.RESILIMIND_LLM_MODEL
    model_digest = get_model_digest(model_name, settings.OLLAMA_BASE_URL)

    generation_config = {
        "temperature": settings.RESILIMIND_LLM_TEMPERATURE,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "routing_confidence_threshold": settings.RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD,
    }

    return BenchmarkRunManifest(
        run_id=run_id,
        git_commit=git_hash,
        dataset_sha256=compute_file_sha256(dataset_path),
        graph_sha256=compute_file_sha256(graph_path),
        model_name=model_name,
        model_digest=model_digest,
        generation_config=generation_config,
        prompt_hashes=get_prompt_hashes(),
        evaluation_code_commit=git_hash,
        judge_model=judge_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def is_manifest_compatible(
    previous: dict[str, Any] | BenchmarkRunManifest,
    current: BenchmarkRunManifest,
) -> tuple[bool, str]:
    """Check if previous benchmark run manifest is compatible with current configuration.

    Args:
        previous: Previous manifest dictionary or BenchmarkRunManifest instance.
        current: Current BenchmarkRunManifest instance.

    Returns:
        tuple[bool, str]: Tuple of (is_compatible, reason_description).
    """
    prev_dict = previous.model_dump() if isinstance(previous, BenchmarkRunManifest) else previous

    if prev_dict.get("model_name") != current.model_name:
        return False, f"Model name changed: {prev_dict.get('model_name')} != {current.model_name}"

    if current.model_digest and prev_dict.get("model_digest") and prev_dict.get("model_digest") != current.model_digest:
        return False, f"Model digest changed: {prev_dict.get('model_digest')} != {current.model_digest}"

    if prev_dict.get("dataset_sha256") != current.dataset_sha256:
        return False, "Dataset file changed (SHA-256 mismatch)"

    if prev_dict.get("graph_sha256") != current.graph_sha256:
        return False, "Knowledge graph file changed (SHA-256 mismatch)"

    prev_prompts = prev_dict.get("prompt_hashes", {})
    for prompt_name, curr_hash in current.prompt_hashes.items():
        if prompt_name in prev_prompts and prev_prompts[prompt_name] != curr_hash:
            return False, f"Prompt '{prompt_name}' content changed (hash mismatch)"

    return True, "Compatible"
