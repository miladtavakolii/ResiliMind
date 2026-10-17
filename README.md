# 🌱 ResiliMind

**A Persian-first, knowledge-graph-grounded LLM agent for personalized resilience support**

ResiliMind is a research prototype that turns Persian conversations into structured resilience signals, evidence-aware assessments, and context-grounded responses. Rather than using one LLM call as the entire application, it orchestrates specialized stages with **LangGraph**, validates outputs with **Pydantic**, grounds analysis in a curated **NetworkX knowledge graph**, and stores conversation state and longitudinal resilience history in **SQLite**.

> [!WARNING]
> **Research prototype — not clinically validated.** ResiliMind is not a medical device, diagnostic system, crisis service, or replacement for licensed mental-health care. Its scores and responses are experimental outputs and must not be interpreted as clinical measurements or professional advice.

## At a glance

- **Persian-first** conversational interface with RTL styling.
- **Stateful, multi-stage workflow** with dedicated safety, extraction, retrieval, assessment, clarification, and response stages.
- **Evidence-linked extraction** with source spans and polarity-specific evidence.
- **Knowledge-graph grounding** across 6 resilience domains (13 nodes and 15 directed edges in the bundled graph).
- **Deterministic score aggregation and status thresholds** after LLM-generated rubric values.
- **Longitudinal structured memory** using LangGraph SQLite checkpoints and a separate SQLite resilience timeline.
- **Evaluation toolkit** for synthetic scenario generation, Persian conversation rendering, dataset validation, benchmark execution, metric aggregation, failure analysis, and LLM-as-a-judge response scoring.
- **Local inference by default** through Ollama; Google Gemini is used by optional evaluation dataset-rendering and response-judging steps.

## Architecture

The current workflow is implemented in <code>src/resilimind/core/workflow.py</code> and uses individual agent modules under <code>src/resilimind/core/agents/</code>.

~~~mermaid
flowchart TD
    A[Persian user message] --> B[Safety classifier]
    B -->|HIGH_RISK| C[Emergency response]
    B -->|UNAVAILABLE / invalid| D[Service unavailable]
    B -->|SAFE| E[Extractor]
    E --> F[Knowledge graph retriever]
    F --> G[Assessor]
    G --> H{Disambiguation needed, no assessments, or low confidence?}
    H -->|Yes| I[Questioner]
    H -->|No| J[Advisor]
    C --> K[Final response]
    D --> K
    I --> K
    J --> K
    J -. reads .-> L[Longitudinal resilience history]
    G -. assessments persisted by service layer .-> L
~~~

### Workflow stages

1. **Safety classifier** normalizes Persian text, applies high-recall phrase screening, and performs structured LLM classification. The safety status is tri-state: <code>SAFE</code>, <code>HIGH_RISK</code>, or <code>UNAVAILABLE</code>. The workflow routes high-risk inputs to a dedicated response and fails closed to a service-unavailable response if the safety subsystem is unavailable or returns an invalid status.
2. **Extractor** selects relevant knowledge-graph nodes and resolves them into structured signals. A signal records the node ID, positive/negative/mixed polarity, a source evidence substring and offsets when available, plus polarity-specific evidence where applicable.
3. **Graph retriever** builds context from the selected nodes and their graph information, including semantic boundaries, exclusions, cues, status definitions, interventions, and relationships.
4. **Assessor** validates that assessments correspond to active signals and graph nodes, then generates a structured four-dimension rubric and reasoning. Python derives the total score and status from the rubric.
5. **Conditional router** sends ambiguous inputs, missing assessments, or low-confidence cases to the **Questioner**. Other assessed cases go to the **Advisor**.
6. **Advisor** combines the current message, conversation history, graph context, and the user's historical resilience timeline to generate a personalized response.
7. **Persistence** stores LangGraph conversation checkpoints and writes new resilience assessments to a separate history table through the service layer.

## Knowledge graph

The bundled graph is stored in <code>src/resilimind/assets/final_resilience_graph.json</code> and loaded as a directed NetworkX graph. It currently contains **13 nodes and 15 directed edges** across six domains:

| Domain | Nodes |
| --- | ---: |
| Personal resilience | 3 |
| Political resilience | 2 |
| Economic resilience | 2 |
| Physical resilience | 2 |
| Social resilience | 2 |
| Spiritual / cultural resilience | 2 |
| **Total** | **13** |

Each node provides domain and bilingual labels, a description, semantic boundary, polarity cues, candidate keywords, status-level descriptions, and intervention guidance. Edges represent cross-domain relationships. The graph provides explicit grounding for extraction, assessment, and response generation; it is not an external graph database.

## Assessment and routing

### Discrete evidence rubric

The LLM supplies four constrained rubric values. The current schema uses **discrete allowed values**, not arbitrary scores from 0 to 25:

| Dimension | Allowed values | Higher values indicate |
| --- | --- | --- |
| Severity | 8, 16, 22 | Less severe distress |
| Frequency | 8, 16, 22 | Less frequent / more isolated difficulty |
| Functionality | 6, 12, 18, 24 | Less functional impairment |
| Coping | 8, 16, 24 | Stronger coping |

The application sums the dimensions in Python and derives the status:

| Total score | Status |
| --- | --- |
| 70 and above | <code>GREEN</code> |
| 40–69 | <code>YELLOW</code> |
| Below 40 | <code>RED</code> |

Because the rubric uses discrete values, the attainable total is not a continuous 0–100 scale. The score and status are **heuristic indicators designed for this prototype**, not validated psychometric measures.

### Confidence-aware routing

The current composite confidence implementation combines:

- 50% LLM-reported assessment confidence
- 30% evidence-quality heuristic
- 20% consistency between polarity evidence and graph cues

The default routing threshold is <code>0.70</code> and can be changed with <code>RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD</code>. This composite is a **routing heuristic, not a calibrated probability**.

## Memory and persistence

ResiliMind keeps conversational state separate from longitudinal assessment history:

- **Conversation/checkpoint memory:** LangGraph's SQLite checkpointer stores workflow state and chat messages in <code>data/checkpoints.db</code>.
- **Longitudinal structured memory:** <code>data/resilimind.db</code> stores user records and timestamped resilience logs, including node, category, status, score, confidence, and reasoning.
- **Advisor context:** past assessments for each node are formatted as a chronological timeline and supplied alongside the current graph context.

This implementation supports temporal tracking of structured resilience assessments. It does **not** yet claim to provide a fully consolidated semantic-memory engine that automatically turns all prior dialogue into validated abstract memories.

## Latest evaluation snapshot

The following figures summarize the evaluation artifacts supplied for the **2026-10-17 local benchmark snapshot**: **100 synthetic cases, 120 Persian conversation turns, and 100/100 cases completing execution**. The reported predictions identify the dataset as <code>v1</code>. These values are a snapshot of one configuration, not a claim of generalization to real-world users.

| Component | Metric | Result |
| --- | --- | ---: |
| Binary safety gate | Precision / recall / F1 | **1.000 / 1.000 / 1.000** |
| Risk-category classification | Accuracy / macro-F1 | **99.0% / 93.1%** |
| Signal/node detection | Macro case-level F1 | **92.4%** |
| Signal/node detection | Micro precision / recall / F1 | **86.3% / 93.5% / 89.8%** |
| Polarity classification | Accuracy | **85.1%** |
| Evidence localization | Substring match | **84.7%** |
| Evidence localization | Exact match / exact-span match | **3.3% / 3.3%** |
| Evidence localization | Token F1 | **47.3%** |
| Assessment | Mean MAE / mean RMSE | **2.32 / 4.02 points** |
| Assessment | Coverage | **94.0%** |
| Assessment status | Accuracy / macro-F1 | **80.9% / 81.5%** |
| Conditional routing | Accuracy / macro-F1 | **96.0% / 94.4%** |
| Questioner routing | Recall | **80.0%** |
| Response quality (LLM judge) | Mean overall score | **9.56 / 10** |
| Execution | Successful cases | **100 / 100** |

The response judge's mean dimension scores were empathy **9.64**, relevance **9.59**, safety **9.81**, actionability **9.48**, consistency **9.68**, and hallucination handling **9.63** (out of 10).

### How to interpret these results

- The binary safety metrics represent **10 high-risk and 90 safe cases**. The risk-category classifier made one category error: a violence case was classified as self-harm. A perfect result on this small high-risk subset is not evidence of validated real-world crisis detection.
- Node detection is stronger than evidence localization. Despite an 84.7% substring-match result, exact evidence/span match is only 3.3% and token F1 is 47.3%; the extractor still needs to identify more precise supporting spans.
- Polarity accuracy is 85.1%, and assessment status accuracy is 80.9%. These remain meaningful error sources for downstream personalization and should not be hidden by the higher routing and response-judge averages.
- The response result is produced by an **LLM-as-a-judge**, not independent human raters. Judge scores can be model-dependent and should be interpreted alongside qualitative review.
- The report lists 58 evaluator-detected failure events across 37 cases. These are **metric mismatches**, not benchmark execution crashes; the benchmark itself completed all 100 cases.

### Evaluation methodology

The evaluation package is separate from runtime agents. It includes:

- **Scenario generation:** creates seeded latent synthetic scenarios with target safety labels, active signals, rubric profiles, expected routing, and response criteria.
- **Persian rendering and evidence audit:** renders scenarios into Persian user messages and validates the rendered evidence against the graph and gold annotations.
- **Dataset validation:** checks schema and node references before evaluation.
- **Benchmark runner:** executes each case turn-by-turn through the LangGraph workflow and records predictions, a run manifest, environment metadata, and logs.
- **Deterministic evaluators:** measure safety, extraction, assessment, and routing.
- **LLM-as-a-judge response evaluator:** scores response empathy, relevance, safety, actionability, consistency, and hallucination handling.
- **Reporting and error analysis:** aggregates metrics and writes per-case results, a failure CSV, and a final JSON report.

Evaluation datasets, runtime databases, and result artifacts are currently excluded by <code>.gitignore</code>. Consequently, the latest figures above are documented benchmark results, but the exact <code>v1</code> dataset and run outputs are not distributed in this repository. Reproducing the same numbers requires the matching dataset, code/model configuration, and judge configuration.

## Installation

### Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Sufficient resources for the configured local model

### Run the application

~~~bash
git clone https://github.com/miladtavakolii/ResiliMind.git
cd ResiliMind

uv sync
cp .env.example .env
ollama pull gemma4:e2b
~~~

Make sure the Ollama service is running in another terminal:

~~~bash
ollama serve
~~~

Start the Streamlit app:

~~~bash
uv run streamlit run src/resilimind/app.py
~~~

The default configuration uses the local Ollama model <code>gemma4:e2b</code> at <code>http://localhost:11434</code>. Edit <code>.env</code> to override settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| <code>RESILIMIND_LLM_MODEL</code> | <code>gemma4:e2b</code> | Ollama model name |
| <code>RESILIMIND_LLM_TEMPERATURE</code> | <code>0.6</code> | LLM temperature setting |
| <code>OLLAMA_BASE_URL</code> | <code>http://localhost:11434</code> | Ollama endpoint |
| <code>DATA_DIR</code> | <code>data</code> | Directory for SQLite databases |
| <code>RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD</code> | <code>0.70</code> | Confidence threshold for advisor routing |
| <code>LOG_LEVEL</code> | <code>INFO</code> | Application logging level |
| <code>GEMINI_API_KEY</code> | — | Required for evaluation rendering and response judging, not for default local inference |
| <code>GEMINI_MODEL</code> | Set in <code>.env</code> or renderer CLI | Gemini model for Persian scenario rendering |
| <code>GEMINI_JUDGE_MODEL</code> | <code>gemini-3.5-flash-lite</code> | Gemini model used by response evaluation |

Do not commit real API keys or local user data.

## Running the evaluation pipeline

The default dataset path is <code>evaluation/datasets/v1/cases.jsonl</code>. The dataset and result folders are git-ignored, so create or obtain the dataset before running these commands.

### 1. Generate latent scenarios

~~~bash
uv run python -m evaluation.generators.generator \
  --count 100 \
  --seed 42 \
  --output evaluation/datasets/v1/scenarios.jsonl
~~~

### 2. Render scenarios into Persian conversations

Set <code>GEMINI_API_KEY</code> in <code>.env</code>, then render the scenarios:

~~~bash
uv run python -m evaluation.generators.renderer \
  --input evaluation/datasets/v1/scenarios.jsonl \
  --output evaluation/datasets/v1/cases.jsonl \
  --model gemini-2.5-flash
~~~

### 3. Validate the dataset

~~~bash
uv run python -m evaluation.scripts.validate_dataset \
  --dataset evaluation/datasets/v1/cases.jsonl
~~~

### 4. Run the benchmark

~~~bash
uv run python -m evaluation.runners.benchmark \
  --dataset evaluation/datasets/v1/cases.jsonl \
  --tag baseline
~~~

The runner creates a timestamped directory under <code>evaluation/results/runs/</code> with <code>predictions.jsonl</code>, <code>run_manifest.json</code>, and <code>benchmark.log</code>. Use <code>--limit N</code> to run only the first N cases while debugging.

### 5. Evaluate the saved predictions

Response evaluation is enabled by default and requires <code>GEMINI_API_KEY</code>. It uses the configured judge model:

~~~bash
uv run python -m evaluation.runners.evaluate \
  --run-dir latest \
  --dataset evaluation/datasets/v1/cases.jsonl
~~~

The run directory receives <code>case_results.jsonl</code>, <code>failures.csv</code>, <code>evaluation_report.json</code>, and <code>evaluation.log</code>. To omit LLM-based response judging:

~~~bash
uv run python -m evaluation.runners.evaluate \
  --run-dir latest \
  --dataset evaluation/datasets/v1/cases.jsonl \
  --skip-response-eval
~~~

Always keep the dataset version and source file consistent between benchmark generation and evaluation. Preserve the run manifest and hashes when comparing runs or preparing research results.

## Project structure

~~~text
ResiliMind/
├── src/resilimind/
│   ├── app.py
│   ├── core/
│   │   ├── agents/
│   │   │   ├── safety.py
│   │   │   ├── extractor.py
│   │   │   ├── retriever.py
│   │   │   ├── assessor.py
│   │   │   ├── responders.py
│   │   │   └── text_utils.py
│   │   ├── database/
│   │   ├── config.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── graph/
│   │   ├── ingestion.py
│   │   └── retriever.py
│   ├── llm/
│   ├── schemas/
│   ├── services/
│   │   ├── memory_service.py
│   │   └── resilience_service.py
│   ├── ui/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── dashboard.py
│   │   └── styles.py
│   └── assets/
│       └── final_resilience_graph.json
├── evaluation/
│   ├── generators/
│   │   ├── generator/
│   │   └── renderer/
│   ├── evaluators/
│   │   ├── safety.py
│   │   ├── extraction.py
│   │   ├── assessment.py
│   │   ├── routing.py
│   │   └── response.py
│   ├── judges/
│   ├── metrics/
│   ├── reporting/
│   ├── runners/
│   │   ├── benchmark.py
│   │   └── evaluate.py
│   ├── schemas.py
│   └── scripts/
│       └── validate_dataset.py
├── pyproject.toml
├── uv.lock
└── README.md
~~~

## Technology stack

| Technology | Role |
| --- | --- |
| Python 3.13+ | Application and evaluation runtime |
| LangGraph | Stateful workflow orchestration and SQLite checkpointing |
| LangChain | LLM integration |
| Ollama / Gemma | Default local inference |
| NetworkX | Directed resilience knowledge graph |
| Pydantic / Pydantic Settings | Structured output schemas and configuration |
| SQLite | Authentication, conversation checkpoints, and resilience history |
| Streamlit | Persian conversational UI |
| Plotly | Dashboard visualizations |
| Google GenAI SDK | Optional scenario rendering and response judge |
| uv | Dependency and environment management |

## Research scope and limitations

ResiliMind is an experimental platform for studying knowledge-graph-grounded LLM agents, evidence-aware extraction, conditional routing, and longitudinal structured memory in Persian conversations.

Current limitations include the small synthetic benchmark, limited high-risk sample count, imperfect polarity and status predictions, low exact evidence-span match, model-dependent LLM-judge scores, and the absence of clinical validation. The evaluation results should be treated as a development snapshot rather than evidence of clinical safety or effectiveness. Future research should include independently reviewed annotations, stronger span-level evaluation, baseline and ablation comparisons, repeated runs, and turn-aligned evaluation of longitudinal changes.

## Author

**Milad Tavakoli** · [GitHub](https://github.com/miladtavakolii)

---

## Disclaimer

ResiliMind does not diagnose mental-health conditions, offer professional psychotherapy, or replace licensed psychological or medical care. Do not rely on this experimental system for emergencies or immediate safety decisions. Seek qualified professional or emergency assistance when needed.
