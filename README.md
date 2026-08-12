# 🌱 ResiliMind: AI-Powered Psychological Resilience Platform

**ResiliMind** is an advanced, multi-agent conversational AI framework designed to evaluate, track, and enhance psychological resilience. By combining Large Language Models (LLMs) with a structured Knowledge Graph, the platform provides tailored psychological interventions and tracks user progress across six core life domains.

---

## ✨ Key Features

* **Multi-Agent State Machine:** Built on `LangGraph`, the conversational flow is routed across specialized nodes (Extractor, Retriever, Assessor, Questioner, and Advisor) to ensure precise evaluation and prevent LLM hallucination.
* **Knowledge Graph Integration (RAG):** Utilizes `NetworkX` to ground the AI's responses in established psychological resilience domains, ensuring evidence-based assessments.
* **Dual-Memory Architecture:**
* *Short-term Memory:* Contextual thread tracking for seamless conversational flow.
* *Long-term Memory:* Persistent SQLite database logging real-time resilience metrics over time.


* **Interactive Visualizations:** Features dynamic Radar Charts (via Plotly) to visualize resilience capacities across six dimensions in real-time.
* **Modern Glassmorphic UI:** A responsive, RTL-supported Streamlit interface optimized for Persian language users.

---

## 🧠 The 6 Core Resilience Domains

ResiliMind evaluates user inputs against a multi-dimensional graph encompassing the following domains:

1. **Personal Resilience:** Self-Efficacy, Hardiness, and Emotion Regulation.
2. **Political-Social Resilience:** Cognitive Immunity and Political Agency.
3. **Economic Resilience:** Financial Adaptation and Career Hardiness.
4. **Physical Resilience:** Somatic Responses, Sleep, and Nutrition.
5. **Social Resilience:** Family Cohesion and Peer Support Networks.
6. **Spiritual/Cultural Resilience:** Meaning-Making and Cultural Connectedness.

---

## 🏗️ Architecture & Workflow

The platform operates on a cyclical multi-agent pipeline:

1. **Extractor Agent:** Parses the user's input to identify active psychological triggers.
2. **Graph Retriever:** Fetches relevant definitions, cues, and interventions from the `NetworkX` Knowledge Graph.
3. **Assessor Agent:** Quantifies the user's state (Score 0-100) and categorizes it into `GREEN`, `YELLOW`, or `RED` zones with a calculated confidence score.
4. **Questioner Agent:** Intervenes with targeted disambiguation questions if the confidence score is low or the input is vague.
5. **Advisor Agent:** Generates empathetic, actionable psychological advice grounded in both the real-time graph context and the user's historical resilience profile.

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.10 or higher
* Git
* uv

### Step-by-Step Guide

1. **Clone the repository:**
```bash
git clone https://github.com/miladtavakolii/ResiliMind.git
cd ResiliMind
```


2. **Create virtual environment and install dependencies using uv:**
```bash
uv sync
```

3. **Run the Application:**
```bash
streamlit run src/resilimind/app.py
```
