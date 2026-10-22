# 🌱 ResiliMind

**A Persian-first, knowledge-graph-grounded LLM agent for personalized resilience support**

ResiliMind is a research prototype that turns Persian conversations into structured resilience signals, evidence-aware assessments, and context-grounded responses. Instead of treating one LLM call as the entire application, it orchestrates specialized stages with **LangGraph**, validates outputs with **Pydantic**, grounds analysis in a curated **NetworkX knowledge graph**, and stores conversational state and longitudinal resilience history in **SQLite**.

> [!WARNING]
> **Research prototype — not clinically validated.** ResiliMind is not a medical device, diagnostic system, crisis service, or replacement for licensed mental-health care. Its scores and responses are experimental outputs, not clinical measurements or professional advice.

## At a glance

- Persian-first conversational interface with RTL styling.
- Stateful multi-stage workflow with dedicated safety, extraction, retrieval, assessment, clarification, and response stages.
- Evidence-linked signal extraction with source substrings and polarity-specific evidence.
- Knowledge-graph grounding across 6 resilience domains (13 nodes and 15 directed edges in the bundled graph).
- Deterministic score aggregation and status thresholds after LLM-generated rubric values.
- Longitudinal structured memory using LangGraph SQLite checkpoints and a separate SQLite resilience timeline.
- Modular multi-provider LLM support: local inference by default via Ollama, alongside first-class support for Gemini, OpenAI, Anthropic, and custom OpenAI-compatible endpoints across both runtime agents and evaluation.

## Architecture

The current workflow is implemented in <code>src/resilimind/core/workflow.py</code>, with agent nodes separated under <code>src/resilimind/core/agents/</code>.

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

1. **Safety classifier** normalizes Persian text, applies high-recall phrase screening, and performs structured LLM classification. The safety status is <code>SAFE</code>, <code>HIGH_RISK</code>, or <code>UNAVAILABLE</code>. High-risk messages use a dedicated response path; an unavailable or invalid safety result fails closed to a service-unavailable response.
2. **Extractor** identifies relevant knowledge-graph nodes and produces structured signals with node IDs, positive/negative/mixed polarity, source evidence, and polarity-specific evidence where available.
3. **Graph retriever** builds context from selected nodes, including semantic boundaries, exclusions, cues, status definitions, interventions, and graph relationships.
4. **Assessor** validates active nodes and generates a structured four-dimension rubric and reasoning. Python derives the total score and status.
5. **Conditional router** sends ambiguous inputs, missing assessments, or low-confidence cases to the **Questioner**. Other assessed cases go to the **Advisor**.
6. **Advisor** combines the current message, conversation history, graph context, and the user's historical resilience timeline to generate a personalized response.
7. **Persistence** stores LangGraph checkpoints and writes new resilience assessments to a separate SQLite history table through the service layer.

## Knowledge graph

The bundled graph is stored in <code>src/resilimind/assets/final_resilience_graph.json</code> and loaded as a directed NetworkX graph. It currently contains **13 nodes and 15 directed edges** across six domains.

| Domain | Nodes |
| --- | ---: |
| Personal resilience | 3 |
| Political resilience | 2 |
| Economic resilience | 2 |
| Physical resilience | 2 |
| Social resilience | 2 |
| Spiritual / cultural resilience | 2 |
| **Total** | **13** |

Each node contains bilingual labels, domain information, a description, semantic boundaries, polarity cues, candidate keywords, status-level descriptions, and intervention guidance. Edges represent cross-domain relationships. The graph provides explicit grounding without requiring an external graph database.

## Assessment and routing

### Discrete evidence rubric

The LLM supplies four constrained rubric values. The current schema uses **discrete allowed values**, not arbitrary scores from 0 to 25.

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

Because the rubric uses discrete values, the attainable total is not a continuous 0–100 scale. Scores and statuses are heuristic indicators designed for this prototype, not validated psychometric measures.

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
- **Advisor context:** prior assessments for each node are formatted as a chronological timeline and supplied alongside the current graph context.

This implementation supports temporal tracking of structured resilience assessments. It does **not** yet claim to provide a fully consolidated semantic-memory engine that automatically turns all prior dialogue into validated abstract memories.

## Latest evaluation snapshot

The following figures summarize the evaluation artifacts supplied for the **2026-10-17 local benchmark snapshot**: **100 synthetic cases and 100/100 cases completing execution**. The reported predictions identify the dataset as <code>v1</code>. These results describe one configuration and should not be interpreted as evidence of generalization to real-world users.

| Component | Metric | Result |
| --- | --- | ---: |
| Binary safety gate | Precision / recall / F1 | **100% / 100% / 100%** |
| Risk-category classification | Accuracy / macro-F1 | **99.0% / 93.1%** |
| Signal/node detection | Macro case-level F1 | **92.4%** |
| Signal/node detection | Micro precision / recall / F1 | **86.3% / 93.5% / 89.8%** |
| Polarity classification | Accuracy | **85.1%** |
| Evidence localization | Exact match / exact-span match | **3.3% / 3.3%** |
| Assessment status | Accuracy / macro-F1 | **80.9% / 81.5%** |
| Conditional routing | Accuracy / macro-F1 | **96.0% / 94.4%** |
| Response quality (LLM judge) | Mean overall score | **9.56 / 10** |
| Execution | Successful cases | **100 / 100** |

### How to interpret these results

- The safety metrics represent **10 high-risk and 90 safe cases**. The risk-category classifier made one category error: a violence case was classified as self-harm. A perfect binary result on this small high-risk subset is not evidence of validated real-world crisis detection.
- Node detection is stronger than evidence localization. Exact evidence/span match is only 3.3%, despite a higher substring-match score; precise evidence localization remains a major limitation.
- Polarity accuracy is 85.1%, and assessment status accuracy is 80.9%. These remain sources of downstream error.
- The response result comes from an **LLM-as-a-judge**, not independent human raters; scores can be model-dependent.
- The report lists 58 evaluator-detected failure events across 37 cases. These are metric mismatches, not benchmark execution crashes; the benchmark itself completed all 100 cases.

The evaluation datasets and result artifacts are excluded by <code>.gitignore</code>. The exact historical dataset and run outputs are not distributed in this repository, so reproducing the same scores requires the matching dataset and model/judge configuration.

For the evaluation workflow, command-line examples, artifact descriptions, and reproducibility guidance, see the **[Evaluation Guide](evaluation/README.md)**.

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

The default configuration uses the local Ollama model <code>gemma4:e2b</code> at <code>http://localhost:11434</code>. Edit the root <code>.env</code> file to override settings.

### Core Application Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| <code>RESILIMIND_LLM_PROVIDER</code> | <code>ollama</code> | Active LLM provider (<code>ollama</code>, <code>gemini</code>, <code>openai</code>, <code>anthropic</code>) |
| <code>RESILIMIND_LLM_MODEL</code> | <code>gemma4:e2b</code> | Target model identifier |
| <code>RESILIMIND_LLM_API_KEY</code> | _None_ | API key for cloud providers (Gemini, OpenAI, Anthropic) |
| <code>RESILIMIND_LLM_BASE_URL</code> | <code>http://localhost:11434</code> | Unified endpoint URL (for Ollama, vLLM, or custom proxies) |
| <code>RESILIMIND_LLM_TEMPERATURE</code> | <code>0.6</code> | Sampling temperature for conversational responses |
| <code>DATA_DIR</code> | <code>data</code> | Directory for SQLite databases and runtime storage |
| <code>RESILIMIND_ROUTING_CONFIDENCE_THRESHOLD</code> | <code>0.70</code> | Confidence threshold for advisor routing |
| <code>LOG_LEVEL</code> | <code>INFO</code> | Application logging level |

### Evaluation & Benchmark Settings (Optional)

Evaluation components (such as scenario rendering and response judging) can be configured independently from the core agent to allow evaluating local models against high-capability cloud judges:

| Variable | Default | Purpose |
| --- | --- | --- |
| <code>RESILIMIND_EVAL_PROVIDER</code> | _Falls back to core provider_ | Provider for evaluation judge and scenario renderer |
| <code>RESILIMIND_EVAL_MODEL</code> | _Falls back to core model_ | Model identifier for evaluation |
| <code>RESILIMIND_EVAL_API_KEY</code> | _Falls back to core key_ | Dedicated API key for evaluation provider |
| <code>RESILIMIND_EVAL_BASE_URL</code> | _Falls back to core URL_ | Dedicated base URL for evaluation provider |

See the [Evaluation Guide](evaluation/README.md) for full setup instructions and CLI arguments. Do not commit API keys or local user data.

## Project structure

~~~text
ResiliMind/
├── src/resilimind/
│   ├── app.py
│   ├── core/
│   │   ├── agents/
│   │   ├── database/
│   │   ├── config.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── graph/
│   ├── llm/
│   │   ├── engine.py
│   │   ├── prompts.py
│   │   └── providers/
│   │       ├── base.py
│   │       ├── factory.py
│   │       ├── ollama.py
│   │       ├── gemini.py
│   │       ├── openai.py
│   │       └── anthropic.py
│   ├── schemas/
│   ├── services/
│   ├── ui/
│   └── assets/
├── evaluation/
│   ├── README.md
│   ├── generators/
│   │   ├── generator/
│   │   └── renderer/
│   ├── evaluators/
│   │   ├── base.py
│   │   ├── judge.py
│   │   └── response.py
│   ├── metrics/
│   ├── reporting/
│   ├── runners/
│   ├── schemas.py
│   └── scripts/
├── pyproject.toml
├── uv.lock
└── README.md
~~~

## Technology stack

| Technology | Role |
| --- | --- |
| Python 3.13+ | Application and evaluation runtime |
| LangGraph | Stateful workflow orchestration and SQLite checkpointing |
| LangChain | LLM abstraction & provider integrations (Ollama, Gemini, OpenAI, Anthropic) |
| Multi-Provider LLM Layer | Local inference (Ollama) or Cloud APIs (Gemini, OpenAI, Anthropic) |
| NetworkX | Directed resilience knowledge graph |
| Pydantic / Pydantic Settings | Structured output schemas and centralized configuration |
| SQLite | Authentication, conversation checkpoints, and resilience history |
| Streamlit | Persian conversational UI |
| Plotly | Dashboard visualizations |
| uv | Fast Python dependency and environment management |

## Research scope and limitations

ResiliMind is an experimental platform for studying knowledge-graph-grounded LLM agents, evidence-aware extraction, conditional routing, and longitudinal structured memory in Persian conversations.

Current limitations include the small synthetic benchmark, limited high-risk sample count, imperfect polarity and status predictions, low exact evidence-span match, model-dependent LLM-judge scores, and the absence of clinical validation. Results should be treated as a development snapshot rather than evidence of clinical safety or effectiveness. Future work should include independently reviewed annotations, stronger span-level evaluation, baseline and ablation comparisons, repeated runs, and turn-aligned evaluation of longitudinal changes.

## Author

**Milad Tavakoli** · [GitHub](https://github.com/miladtavakolii)

---

## Disclaimer

ResiliMind does not diagnose mental-health conditions, offer professional psychotherapy, or replace licensed psychological or medical care. Do not rely on this experimental system for emergencies or immediate safety decisions. Seek qualified professional or emergency assistance when needed.
