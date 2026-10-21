# ResiliMind Evaluation Guide

This directory contains the evaluation pipeline for ResiliMind. It is separate from the runtime application and supports synthetic scenario generation, Persian conversation rendering, dataset validation, benchmark execution, metric evaluation, response judging, and error reporting.

For application architecture, installation, and current results, see the [main README](../README.md).

> [!IMPORTANT]
> Evaluation datasets, run directories, logs, and reports are excluded from Git by the repository's <code>.gitignore</code>. You must generate or obtain a dataset before running these commands. Generating a new dataset does not recreate the exact historical benchmark inputs, so do not expect identical scores without the original dataset and matching configuration.

## Pipeline overview

~~~mermaid
flowchart TD
    A[Seeded scenario generation] --> B[Persian conversation rendering]
    B --> C[Evidence and dataset validation]
    C --> D[LangGraph benchmark execution]
    D --> E[Saved predictions and run manifest]
    E --> F[Deterministic evaluators]
    E --> G[LLM-as-a-judge response evaluator]
    F --> H[Aggregate report and failure analysis]
    G --> H
~~~

The evaluation pipeline has five main stages:

1. **Scenario generation** creates synthetic, structured cases containing target safety labels, resilience signals, assessment profiles, expected routing, and response criteria.
2. **Conversation rendering** turns scenarios into Persian messages and attaches evidence annotations. Semantic auditing and validation help check consistency between the rendered messages, evidence, graph, and gold labels.
3. **Dataset validation** checks case schemas and graph node references before execution.
4. **Benchmark execution** runs each case through the actual LangGraph workflow, turn by turn, and records predictions, run metadata, and logs.
5. **Evaluation and reporting** computes safety, extraction, assessment, and routing metrics; optionally uses Gemini as an LLM judge for response quality; then writes aggregate results and a failure CSV.

## Requirements

- Python 3.13+
- Dependencies installed with <code>uv sync</code>
- A running Ollama service and the model configured for ResiliMind
- A Gemini API key for Persian scenario rendering and response judging

The main application can run with local Ollama inference without a Gemini API key. Gemini is required only for the evaluation steps that call the Google GenAI API.

Install project dependencies from the repository root:

~~~bash
uv sync
~~~

Add your key to the root <code>.env</code> file (do not commit it):

~~~dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_JUDGE_MODEL=gemini-3.5-flash-lite
~~~

<code>GEMINI_MODEL</code> is used by the conversation renderer unless overridden with <code>--model</code>. Response judging defaults to <code>GEMINI_JUDGE_MODEL</code> or <code>gemini-3.5-flash-lite</code>.

## Dataset paths

The default paths are:

| Artifact | Default path |
| --- | --- |
| Latent scenarios | <code>evaluation/datasets/v1/scenarios.jsonl</code> |
| Rendered dataset | <code>evaluation/datasets/v1/cases.jsonl</code> |
| Benchmark run folders | <code>evaluation/results/runs/</code> |
| Latest-run pointer | <code>evaluation/results/latest</code> |

These directories are ignored by Git. Make sure the dataset path is the same when running the benchmark and evaluating the saved predictions.

## Step 1 — Generate latent scenarios

Run from the repository root:

~~~bash
uv run python -m evaluation.generators.generator --count 100 --seed 42 --output evaluation/datasets/v1/scenarios.jsonl
~~~

Useful options:

- <code>--count</code>: number of cases to generate (default: <code>100</code>).
- <code>--seed</code>: random seed for scenario sampling (default: <code>42</code>).
- <code>--graph</code>: alternate graph JSON path.
- <code>--output</code>: output JSONL path.

The generator builds the scenario and target labels; its output is not yet a rendered Persian conversation dataset.

## Step 2 — Render scenarios into Persian conversations

~~~bash
uv run python -m evaluation.generators.renderer --input evaluation/datasets/v1/scenarios.jsonl --output evaluation/datasets/v1/cases.jsonl --model gemini-2.5-flash
~~~

The renderer uses Gemini to generate Persian user messages and then validates/audits the rendered evidence. Override the model with <code>--model</code>, change input/output paths, or use <code>--limit N</code> for a smaller test. API access and model availability are required for this stage.

Do not overwrite a dataset associated with a completed benchmark if you need to reproduce or audit that run. Store each dataset revision in a separate versioned path and record its checksum.

## Step 3 — Validate the dataset

~~~bash
uv run python -m evaluation.scripts.validate_dataset --dataset evaluation/datasets/v1/cases.jsonl
~~~

Validation checks schema requirements and that referenced node IDs exist in the configured knowledge graph. Fix validation errors before benchmarking.

You can validate against an alternate graph with <code>--graph path/to/graph.json</code>.

## Step 4 — Run the benchmark

~~~bash
uv run python -m evaluation.runners.benchmark --dataset evaluation/datasets/v1/cases.jsonl --tag baseline
~~~

The benchmark runs the workflow over all cases and stores one run under <code>evaluation/results/runs/</code>. Typical artifacts include:

| File | Purpose |
| --- | --- |
| <code>predictions.jsonl</code> | Per-case, per-turn workflow outputs |
| <code>run_manifest.json</code> | Run ID, code/dataset/graph fingerprints, and model metadata |
| <code>benchmark.log</code> | Execution log |

The runner also updates the <code>evaluation/results/latest</code> pointer when supported by the filesystem.

### Useful benchmark options

Run only the first few cases while debugging:

~~~bash
uv run python -m evaluation.runners.benchmark --dataset evaluation/datasets/v1/cases.jsonl --limit 5 --tag smoke-test
~~~

Use a custom output directory:

~~~bash
uv run python -m evaluation.runners.benchmark --dataset evaluation/datasets/v1/cases.jsonl --output evaluation/results/runs/manual-run
~~~

Use <code>--help</code> to see all supported options.

## Step 5 — Evaluate saved predictions

Evaluate the most recent run:

~~~bash
uv run python -m evaluation.runners.evaluate --run-dir latest --dataset evaluation/datasets/v1/cases.jsonl
~~~

This reads stored predictions and evaluates them against the supplied gold annotations. Response evaluation is enabled by default and requires <code>GEMINI_API_KEY</code>.

The run directory receives:

| File | Purpose |
| --- | --- |
| <code>case_results.jsonl</code> | Per-case metrics from each evaluator |
| <code>failures.csv</code> | Failure and metric-mismatch analysis |
| <code>evaluation_report.json</code> | Aggregated benchmark metrics |
| <code>evaluation.log</code> | Evaluation log |

To skip the Gemini-based response judge and run the other evaluators only:

~~~bash
uv run python -m evaluation.runners.evaluate --run-dir latest --dataset evaluation/datasets/v1/cases.jsonl --skip-response-eval
~~~

To evaluate a specific run, pass its directory name or path instead of <code>latest</code>:

~~~bash
uv run python -m evaluation.runners.evaluate --run-dir 2026-10-17_01-30-21_6778c3f --dataset evaluation/datasets/v1/cases.jsonl
~~~

Replace the example run-directory name with the directory created on your machine.

## Metrics and interpretation

The deterministic evaluators cover:

- **Safety:** high-risk detection and risk-category classification.
- **Extraction:** node/signal precision, recall, F1, polarity accuracy, and evidence matching/localization metrics.
- **Assessment:** score errors, coverage, and status classification metrics.
- **Routing:** classification metrics for advisor, questioner, and emergency-response decisions.
- **Response quality:** optional LLM-as-a-judge scores for empathy, relevance, safety, actionability, consistency, and hallucination handling.

An execution marked successful means the case completed without an execution-level exception. It does **not** mean every prediction matched the gold labels. Metric mismatches can occur in successfully executed cases.

Treat the response judge as an automated evaluator, not a replacement for human review. Results from a synthetic dataset should not be generalized to real users without independent validation.

## Reproducibility checklist

Before comparing model or code changes, preserve and record:

- The exact dataset file, its version, and checksum.
- The Git commit used for benchmark execution.
- The graph version/checksum.
- The Ollama model name and, where available, model digest.
- The Gemini renderer and judge models and their relevant settings.
- The benchmark manifest, predictions, evaluation report, failures CSV, and logs.

**Keep dataset provenance consistent.** The <code>dataset_version</code> recorded in each prediction is not a substitute for preserving the actual dataset. Do not compare or report results if the predictions and gold annotations cannot be shown to come from the same dataset revision.

For research reporting, document whether gold labels were human-reviewed, synthetically generated, or modified after inspection. Describe the limitations of the dataset and evaluator, include error analysis, and avoid presenting synthetic-benchmark scores as evidence of clinical validity.

## Troubleshooting

### <code>GEMINI_API_KEY</code> is required

Set <code>GEMINI_API_KEY</code> in the root <code>.env</code> file or export it in the shell before running scenario rendering or response evaluation. The application's default Ollama inference does not require the Gemini key.

### Dataset file not found

The dataset is excluded by <code>.gitignore</code>. Generate it with Steps 1 and 2, or pass the explicit path to a prepared JSONL dataset using <code>--dataset</code>.

### Dataset-version or alignment errors

Ensure <code>--dataset</code> points to the exact same dataset revision used to generate <code>predictions.jsonl</code>. Do not regenerate the file in place between benchmark execution and evaluation. When experimenting with new gold annotations, create a new versioned dataset and rerun the benchmark.

### Some cases succeed but the report contains failures

Check <code>failures.csv</code>, <code>case_results.jsonl</code>, and <code>benchmark.log</code>. A reported evaluator failure often means the prediction differs from a gold annotation; it does not necessarily mean the workflow crashed.

---

[Back to the main README](../README.md)
