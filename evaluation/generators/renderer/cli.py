from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .engine import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_AUDIT_PROMPT_PATH,
    ScenarioRenderer,
    load_cases,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scenario renderer.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Render latent ResiliMind evaluation scenarios into natural-language Persian conversations."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the latent scenarios JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the rendered cases JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ),
        help="Gemini model used for scenario rendering.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the rendering system prompt.",
    )
    parser.add_argument(
        "--audit-prompt",
        type=Path,
        default=DEFAULT_AUDIT_PROMPT_PATH,
        help="Path to the semantic audit prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N cases.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip cases that fail rendering.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries after an API failure.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay before semantic audit and between consecutive cases in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete scenario rendering pipeline."""
    load_dotenv()
    args = parse_args()

    cases = load_cases(args.input)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")

        cases = cases[:args.limit]

    processed_case_ids = set()
    if args.output.exists():
        logger.info("Found existing results. Resuming from %s", args.output)
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "case_id" in data:
                            processed_case_ids.add(data["case_id"])
                    except json.JSONDecodeError:
                        pass
    
    cases_to_run = [c for c in cases if c.case_id not in processed_case_ids]

    if not cases_to_run:
        logger.info("All cases have already been processed.")
        return

    renderer = ScenarioRenderer(
        model_name=args.model,
        temperature=args.temperature,
        prompt_path=args.prompt,
        audit_prompt_path=args.audit_prompt,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
    )

    valid_node_ids = set(renderer.nodes)

    rendered_cases = renderer.render_dataset(
        cases_to_run,
        skip_failures=args.skip_failures,
        output_path=args.output,
        valid_node_ids=valid_node_ids,
    )

    print(f"Rendered {len(rendered_cases)} cases in this run.")
    print(f"Dataset: {args.output}")
    print(f"Model: {args.model}")
