# Conversational HCI Model Using LLMs

A research-grade pipeline for evaluating conversational Human-Computer Interaction (HCI) models powered by Large Language Models. The system generates, evaluates, and benchmarks dialog responses across rule-based and OpenAI-driven modes, with a built-in Flask dashboard for interactive reporting and artifact management.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Pipeline](#evaluation-pipeline)
- [API Reference](#api-reference)
- [License](#license)

---

## Overview

This project implements an end-to-end conversational HCI evaluation framework that:

- Runs multi-turn dialog simulations using configurable modes (rule-based heuristics or OpenAI GPT)
- Computes token-level overlap metrics against gold-standard responses
- Generates task-level and conversation-level summary reports
- Provides a web-based dashboard for pipeline execution and artifact download
- Exports evaluation artifacts as CSV, JSON, Markdown, and PDF

---

## Architecture

```
+---------------------+
|   Flask Dashboard   |  <-- Browser UI (port 8000)
|   (app.py)          |
+---------------------+
         |
         v
+---------------------+     +----------------------+
| Pipeline Runner     | --> | Report Package       |
| src/run_pipeline    |     | src/generate_report  |
+---------------------+     +----------------------+
         |                            |
         v                            v
+---------------------+     +----------------------+
| Dialog Engine       |     | reports/             |
| (rule_based/openai) |     |  - CSV summaries     |
+---------------------+     |  - JSON data pack    |
         |                  |  - Markdown report    |
         v                  +----------------------+
+---------------------+
| Evaluation Module   |
| (token overlap,     |
|  task metrics)      |
+---------------------+
```

### Data Flow

1. **Initialization** -- User selects mode (rule-based / OpenAI) and max dialog turns via the dashboard.
2. **Pipeline Execution** -- `src/run_pipeline` generates responses across conversations and tasks.
3. **Evaluation** -- Token overlap against gold-standard responses is computed at task and conversation granularity.
4. **Report Generation** -- `src/generate_report_package` produces structured CSV, JSON, and Markdown artifacts.
5. **Visualization** -- The Flask dashboard renders metric cards, data tables, and download links.

---

## Technology Stack

| Component            | Technology                     |
|----------------------|--------------------------------|
| Runtime              | Python 3.10+                   |
| Web Framework        | Flask 3.1                      |
| Data Processing      | pandas 2.2, NumPy 2.2         |
| ML Evaluation        | scikit-learn 1.6               |
| LLM Integration      | OpenAI SDK 1.67                |
| Report Generation    | ReportLab 4.4 (PDF)            |
| Environment Config   | python-dotenv 1.0              |
| Progress Tracking    | tqdm 4.67                      |

---

## Project Structure

```
Conversational-HCI-Model-LLM/
|
|-- app.py                     # Flask dashboard (entry point)
|-- requirements.txt           # Python dependencies
|-- .env.example               # Environment variable template
|
|-- src/
|   |-- run_pipeline.py        # Dialog generation pipeline
|   |-- generate_report_package.py  # Report artifact builder
|   |-- evaluator.py           # Token overlap and metrics engine
|   +-- dialog_engine.py       # Rule-based and OpenAI response generators
|
|-- data/
|   +-- (dataset files)        # Input conversation datasets
|
|-- prompts/
|   +-- (prompt templates)     # System and user prompt templates
|
+-- reports/
    |-- full_data_for_report.csv
    |-- task_level_summary.csv
    |-- conversation_level_summary.csv
    |-- report_data_pack.json
    |-- final_report.md
    |-- generated_responses.csv
    +-- summary_metrics.json
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/karthik-idikuda/Conversational-HCI-Model-LLM.git
cd Conversational-HCI-Model-LLM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your OpenAI API key to .env (required for OpenAI mode)
```

---

## Usage

### Start the Dashboard

```bash
python app.py
```

Open `http://127.0.0.1:8000` in your browser.

### Run Pipeline via CLI

```bash
# Rule-based mode
python -m src.run_pipeline --mode rule_based --max-turns 500

# OpenAI mode
python -m src.run_pipeline --mode openai --max-turns 500

# Generate report artifacts
python -m src.generate_report_package
```

---

## Evaluation Pipeline

The evaluation engine computes the following metrics:

| Metric                    | Description                                           |
|---------------------------|-------------------------------------------------------|
| Token Overlap vs Gold     | Jaccard similarity between generated and reference tokens |
| Task-Level Summary        | Aggregated metrics per dialog task category            |
| Conversation-Level Summary| Per-conversation accuracy and response quality         |
| Dataset Overview          | Row count, conversation count, task count, avg overlap |

---

## API Reference

| Endpoint              | Method | Description                              |
|-----------------------|--------|------------------------------------------|
| `/`                   | GET    | Dashboard with metrics and reports       |
| `/run`                | POST   | Execute pipeline (mode, max_turns)       |
| `/download/<filename>`| GET    | Download report artifact                 |

---

## License

This project is released for educational and research purposes.
