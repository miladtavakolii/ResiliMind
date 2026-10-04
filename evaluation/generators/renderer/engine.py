from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from google import genai
from google.genai import types

from evaluation.schemas import EvaluationCase
from evaluation.generators.validators import validate_dataset
from .models import ScenarioRenderOutput
from .evidence import (
    clean_messages,
    extract_evidence,
    attach_evidence,
    validate_evidence_markup,
    validate_rendered_output,
)
from .prompts import (
    build_user_prompt,
    build_retry_prompt,
)
from .audit import (
    validate_ambiguous_output,
    audit_rendered_output,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "evaluation" / "prompts" / "scenario_renderer.txt"
DEFAULT_AUDIT_PROMPT_PATH = PROJECT_ROOT / "evaluation" / "prompts" / "scenario_renderer_audit.txt"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "scenarios.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "v1" / "cases.jsonl"
GRAPH_PATH = PROJECT_ROOT / "src" / "resilimind" / "assets" / "final_resilience_graph.json"


class ScenarioRenderer:
    """Render latent evaluation scenarios into natural Persian conversations.

    Attributes:
        VERSION (str): Renderer component version.
        model_name (str): Gemini model identifier.
        temperature (float): Generation temperature.
        prompt_path (Path): Path to the system rendering prompt template.
        max_retries (int): Number of retries on API failure.
        retry_delay (float): Initial backoff delay between retries.
        request_delay (float): Politeness delay before semantic audit and between successive scenario generations.
        client (genai.Client): Initialized Google GenAI SDK client.
        system_prompt (str): Loaded rendering prompt instructions.
    """

    VERSION: str = "1.0.0"

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.4,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        audit_prompt_path: Path = DEFAULT_AUDIT_PROMPT_PATH,
        graph_path: Path = GRAPH_PATH,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_delay: float = 1.0,
    ) -> None:
        """Initialize the scenario renderer.

        Args:
            model_name: Gemini model name for natural language generation.
            api_key: Optional Gemini API key. If omitted, fetched from GEMINI_API_KEY.
            temperature: Sampling temperature for generation diversity.
            prompt_path: Path to the rendering prompt text file.
            audit_prompt_path: Path to the audit prompt text file.
            graph_path: Path to the knowledge graph JSON file.
            max_retries: Maximum retry attempts for generation, audit, and validation failures.
            retry_delay: Base delay in seconds for exponential backoff.
            request_delay: Delay in seconds before semantic audit and between rendered cases.

        Raises:
            ValueError: If delays or retries are negative, or if the API key is unset.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        if request_delay < 0:
            raise ValueError("request_delay must be >= 0")

        self.model_name = model_name
        self.temperature = temperature
        self.prompt_path = prompt_path
        self.audit_prompt_path = audit_prompt_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.graph_path = Path(graph_path)
        self.graph = self._load_graph()
        self.nodes = self._load_nodes()

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY in the environment.")

        self.client = genai.Client(api_key=api_key)
        self.system_prompt = self._load_prompt()
        self.audit_prompt = self._load_audit_prompt()

    def _load_graph(self) -> dict[str, Any]:
        """Load and validate the knowledge graph from disk.

        Returns:
            dict[str, Any]: Parsed knowledge graph containing validated nodes.

        Raises:
            FileNotFoundError: If the knowledge graph file does not exist.
            ValueError: If the JSON payload lacks a valid top-level 'nodes' dictionary.
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {self.graph_path}")

        with self.graph_path.open("r", encoding="utf-8") as file:
            graph = json.load(file)

        if "nodes" not in graph or not isinstance(graph["nodes"], dict):
            raise ValueError("Knowledge graph must contain a valid 'nodes' object")

        return graph

    def _load_audit_prompt(self) -> str:
        """Load the semantic audit prompt from disk."""
        if not self.audit_prompt_path.exists():
            raise FileNotFoundError(
                f"Audit prompt not found: {self.audit_prompt_path}"
            )
        return self.audit_prompt_path.read_text(encoding="utf-8")

    def _load_nodes(self) -> dict[str, dict[str, Any]]:
        """Retrieve node definitions from the loaded knowledge graph.

        Returns:
            dict[str, dict[str, Any]]: Mapping of node IDs to their attributes and metadata.
        """
        return self.graph["nodes"]

    def _load_prompt(self) -> str:
        """Load the rendering system prompt from disk.

        Returns:
            str: System prompt text content.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Rendering prompt not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def _generate(self, *, prompt: str) -> ScenarioRenderOutput:
        """Generate structured scenario content with automatic retries and exponential backoff.

        Args:
            prompt: Structured JSON string representing the latent scenario spec.

        Returns:
            ScenarioRenderOutput: Structured messages and evidence output.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[self.system_prompt, prompt],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=ScenarioRenderOutput,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )

                if not response.text:
                    raise ValueError("Gemini returned an empty response")

                return ScenarioRenderOutput.model_validate_json(response.text)

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                delay = self.retry_delay * (2**attempt)

                logger.warning(
                    "Gemini request failed (attempt %d/%d): %s. Retrying in %.1f seconds...",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Gemini request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def render(self, case: EvaluationCase) -> EvaluationCase:
        """Render one latent scenario into a natural-language conversation.

        Args:
            case: Evaluation case containing the latent scenario specification.

        Returns:
            EvaluationCase: Updated case with rendered messages and attached evidence.
        """
        base_prompt = build_user_prompt(case, self.nodes)
        prompt = base_prompt
        messages: list[str] = []
        evidence: list[dict[str, Any]] = []

        for attempt in range(self.max_retries + 1):
            try:
                rendered = self._generate(prompt=prompt)

                validate_evidence_markup(
                    case=case,
                    output=rendered,
                )

                evidence = extract_evidence(
                    case,
                    rendered,
                )
                messages = clean_messages(rendered.messages)

                validate_ambiguous_output(
                    case=case,
                    messages=messages,
                    nodes=self.nodes,
                )

                validate_rendered_output(
                    case=case,
                    messages=messages,
                    evidence=evidence,
                )

                audit = audit_rendered_output(
                    client=self.client,
                    model_name=self.model_name,
                    audit_prompt=self.audit_prompt,
                    case=case,
                    messages=messages,
                    nodes=self.nodes,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    request_delay=self.request_delay,
                )

                audit_errors = (
                    audit.missing_target_signals
                    + audit.unintended_signal_nodes
                    + audit.polarity_errors
                    + audit.evidence_issues
                    + audit.assessment_errors
                    + audit.safety_errors
                )

                if not audit.valid or audit_errors:
                    raise ValueError(
                        f"{case.case_id}: semantic audit failed: "
                        f"missing={audit.missing_target_signals}, "
                        f"unintended={audit.unintended_signal_nodes}, "
                        f"polarity={audit.polarity_errors}, "
                        f"evidence={audit.evidence_issues}, "
                        f"assessment={audit.assessment_errors}, "
                        f"safety={audit.safety_errors}, "
                        f"explanation={audit.explanation}"
                    )

                break

            except ValueError as exc:
                if attempt >= self.max_retries:
                    raise

                delay = self.retry_delay * (2**attempt)

                logger.warning(
                    "%s: invalid rendering on attempt %d/%d: %s. "
                    "Retrying in %.1f seconds...",
                    case.case_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )

                time.sleep(delay)

                prompt = build_retry_prompt(
                    base_prompt=base_prompt,
                    error=str(exc),
                )

        case.input.messages = messages

        attach_evidence(
            case=case,
            evidence=evidence,
        )

        return case

    def render_dataset(
        self,
        cases: list[EvaluationCase],
        *,
        skip_failures: bool = False,
        output_path: Path | None = None,
        valid_node_ids: set | None = None,
    ) -> list[EvaluationCase]:
        """Render all cases in a dataset sequentially.

        Args:
            cases: Sequence of evaluation cases to render.
            skip_failures: Whether to bypass individual failures instead of raising.
            output_path: Optional path to save incrementally.
            valid_node_ids: Optional set of graph node IDs for individual case validation.

        Returns:
            list[EvaluationCase]: Successfully rendered evaluation cases.
        """
        rendered_cases = []

        for index, case in enumerate(cases, start=1):
            logger.info(
                "Rendering case %d/%d: %s",
                index,
                len(cases),
                case.case_id,
            )

            try:
                rendered_case = self.render(case)

                # Validate single case before saving
                if valid_node_ids is not None:
                    validate_dataset(
                        [rendered_case],
                        valid_node_ids=valid_node_ids,
                        require_rendered=True,
                    )

                # Save instantly if output path is provided
                if output_path:
                    write_cases([rendered_case], output_path)

                rendered_cases.append(rendered_case)

            except Exception:
                logger.exception(
                    "Failed to render case %s",
                    case.case_id,
                )

                if not skip_failures:
                    raise

            if index < len(cases) and self.request_delay > 0:
                logger.info(
                    "Sleeping %.1f seconds before next case...",
                    self.request_delay,
                )
                time.sleep(self.request_delay)

        return rendered_cases


def load_cases(input_path: Path) -> list[EvaluationCase]:
    """Load evaluation scenarios from a JSONL input file.

    Args:
        input_path: Path to the JSONL input file.

    Returns:
        list[EvaluationCase]: Parsed evaluation cases.

    Raises:
        FileNotFoundError: If the input file is missing.
        ValueError: If line deserialization or schema validation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    cases: list[EvaluationCase] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(
                    EvaluationCase.model_validate(
                        json.loads(line)
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation case at line "
                    f"{line_number}: {exc}"
                ) from exc

    return cases


def write_cases(
    cases: list[EvaluationCase],
    output_path: Path,
) -> None:
    """Write rendered evaluation cases to a JSONL file.

    Args:
        cases: Sequence of evaluation cases to write.
        output_path: Destination path for the rendered cases JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        for case in cases:
            file.write(
                json.dumps(
                    case.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )
