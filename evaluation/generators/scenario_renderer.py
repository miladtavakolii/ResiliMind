"""Backward-compatible entry point for ScenarioRenderer.

This module re-exports components from the modular `evaluation.generators.renderer`
subpackage so existing scripts, evaluation runners, and CLI invocations continue
to function without changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.generators.renderer import (
    ScenarioRenderer,
    RenderedMessage,
    ScenarioRenderOutput,
    ScenarioRenderAudit,
    EVIDENCE_PATTERN,
    EVIDENCE_MARKER_PATTERN,
    normalize_match_text,
    clean_messages,
    extract_evidence,
    attach_evidence,
    validate_evidence_markup,
    validate_rendered_output,
    build_assessment_requirements,
    build_user_prompt,
    build_retry_prompt,
    validate_ambiguous_output,
    audit_rendered_output,
    load_cases,
    write_cases,
    parse_args,
    main,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_AUDIT_PROMPT_PATH,
    GRAPH_PATH,
)

# Backward-compatibility alias
_build_assessment_requirements = build_assessment_requirements

__all__ = [
    "ScenarioRenderer",
    "RenderedMessage",
    "ScenarioRenderOutput",
    "ScenarioRenderAudit",
    "EVIDENCE_PATTERN",
    "EVIDENCE_MARKER_PATTERN",
    "normalize_match_text",
    "clean_messages",
    "extract_evidence",
    "attach_evidence",
    "validate_evidence_markup",
    "validate_rendered_output",
    "TARGET_SCOPE_GUIDANCE",
    "build_assessment_requirements",
    "_build_assessment_requirements",
    "build_user_prompt",
    "build_retry_prompt",
    "validate_ambiguous_output",
    "audit_rendered_output",
    "load_cases",
    "write_cases",
    "parse_args",
    "main",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_AUDIT_PROMPT_PATH",
    "GRAPH_PATH",
]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()