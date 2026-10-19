<div align="center">

<img src="./static/image/weaveragent_logo.svg" alt="WeaverAgent Logo" width="75%"/>

**A Microsoft-GraphRAG-Powered Academic Knowledge Graph Engine**
</br>
<em>基于 Microsoft GraphRAG 的学术知识图谱分析引擎</em>

[English](./README-EN.md) | [中文文档](./README.md) | [Full Worklog](./reports/work.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![GraphRAG](https://img.shields.io/badge/Engine-Microsoft%20GraphRAG-purple)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Vue 3](https://img.shields.io/badge/Vue-3-4FC08D)

</div>

## ⚡ Overview

**WeaverAgent** is an academic knowledge-graph workbench. Drop in a batch of PDF / Markdown papers and the system will:

1. **Auto-design an ontology** — the LLM reads the papers and decides which entity types matter (Paper / Method / Dataset / Metric / Innovation / Task / Baseline / Author)
2. **Build a GraphRAG index** — chunking, entity/relation extraction, Leiden community hierarchy, LLM-written community reports
3. **Answer with a ReACT agent** — multi-tool retrieval with every fact automatically tagged by **source paper**
4. **Generate provenance-grounded analysis reports** — no hallucinated citations

> **In one sentence**: upload PDFs, ask questions, get answers that always tell you which paper each claim came from.

### Key Features

- 🧠 **Auto-designed ontology** — no pre-defined schema, the LLM figures out what to extract
- 🏗️ **Full local GraphRAG** — Microsoft's GraphRAG runs entirely on your machine; all outputs are portable Parquet files
- 📎 **Source-paper traceability** — every fact ships with `【Source: <paper title>】`, no fabrication
- 🔄 **Multi-tool ReACT** — `quick_search` / `panorama_search` / `insight_forge` / `deep_entity_query` picked by intent
- 🛡️ **Anti-"stalling" guard** — detects and retries when the LLM promises to search but forgets to emit a tool call
- 🌐 **Multi-provider LLM** — GPT-5 / GPT-4o / GLM / Qwen / Kimi via any OpenAI-SDK-compatible endpoint
- 💰 **Token & cost reporting** — built-in scripts summarise how much each graph cost to build

## 🔄 Workflow

Five-step UI wizard (`Step1GraphBuild.vue` … `Step5Interaction.vue`):

1. **Graph Build** — Upload PDFs → LLM generates ontology → GraphRAG indexing → Parquet + LanceDB
2. **Environment Setup** — Inspect entities/relations → graph statistics → tune analysis parameters
3. **Graph Analysis** — Node-type distribution → hub detection → technical-pathway extraction
4. **Report Generation** — ReportAgent runs multi-round ReACT → section-by-section report → export Markdown/PDF
5. **Deep Interaction** — Chat in natural language → agent picks tools → answers cite paper sources

Full architecture diagrams in [`reports/work.md`](./reports/work.md).

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose | Check |
|------|---------|---------|-------|
| **Node.js** | ≥ 18 | frontend runtime | `node -v` |
| **Python** | 3.11 – 3.12 | backend runtime | `python --version` |
| **uv** | latest | Python package manager | `uv --version` |

### 1. Configure environment variables

```bash
cp .env.example .env
# edit .env, fill in your LLM API key
```

#### Required

```env
# Any OpenAI-SDK-compatible LLM endpoint
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-5-mini
```

#### Recommended: independent Embedding config

```env
EMBEDDING_API_KEY=sk-xxx                         # falls back to LLM_API_KEY if unset
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-large      # 3072-dim, recommended
```

#### Optional

```env
GRAPHRAG_DATA_DIR=/path/to/graphrag_data         # default: ../graphrag_data

REPORT_AGENT_MAX_TOOL_CALLS=5                    # per-turn tool-call budget
REPORT_AGENT_MAX_REFLECTION_ROUNDS=2
REPORT_AGENT_TEMPERATURE=0.5
```

#### Model selection cheat-sheet

| Scenario | Recommended | Rationale |
|---|---|---|
| Large-batch indexing (≥ 30 papers) | `gpt-4o-mini` + `text-embedding-3-small` | ~¥0.08 per paper |
| Small batch, highest quality | `gpt-5` + `text-embedding-3-large` (3072d) | Richer community reports, better recall |
| Chinese compliance / cost-sensitive | `glm-5-turbo` + `embedding-3` | Data stays in China (`thinking=disabled`) |
| Very long documents | `kimi-k2-turbo-preview` | 128K context |
| **Avoid** for bulk indexing | `gpt-4o` | 16.7× the price of mini, no proportional gain |

Full cost comparison (LaTeX-scanned builds + GPT-5 **10-paper** and **100-paper** case studies) in [`reports/work.md §5`](./reports/work.md#5-graphrag-多模型构建成本对比).

### 2. Install dependencies

```bash
# one-shot (recommended)
npm run setup:all

# or step by step
npm run setup          # Node parts (root + frontend)
npm run setup:backend  # Python parts (uv sync, auto-creates .venv)
```

> ⚠️ The repo vendors the Microsoft GraphRAG monorepo in `graphrag/packages/` and installs it as an editable package. If the backend reports `ModuleNotFoundError: graphrag`, `start.sh` will auto-repair the editable install.

### 3. Start services

```bash
# launch both frontend and backend
npm run dev

# or use the shell helpers (background + auto-restart)
./start.sh
./stop.sh
```

**URLs**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5001

Start individually:

```bash
npm run backend   # backend only
npm run frontend  # frontend only
```

### 4. Docker

```bash
cp .env.example .env
docker compose up -d
```

Maps ports `3000` (frontend) / `5001` (backend), auto-loads root `.env`.

## 🏗️ Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Vue 3 + Vue Router 4 + Vite + D3.js (graph visualisation) |
| Backend | Flask 3 + Flask-CORS |
| **Graph engine** | **Microsoft GraphRAG** (vendored monorepo, runs locally) |
| Vector store | LanceDB (GraphRAG native) + optional ChromaDB (paragraph RAG) |
| LLM | OpenAI-SDK-compatible (GPT-5 reasoning / GLM / Qwen / Kimi / GPT-4o …) |
| Embedding | `text-embedding-3-large` (3072d) / `text-embedding-3-small` / `embedding-3` / … |
| Storage | **Local filesystem** (Parquet + LanceDB + JSON), shippable via `tar` |
| File parsing | PyMuPDF (PDF) + charset-normalizer |
| Optional graph DB | Neo4j (reserved for future graph algorithms) |

### Data flow

```
PDF/MD → TextProcessor → OntologyGenerator (LLM)
                            ↓
                       GraphBuilderService (GraphRAG)
                            ↓
            ┌───────────────┴───────────────┐
            ▼                                ▼
   output/*.parquet                     lancedb/
   (nodes, edges, communities)       (entity embeddings)
            │                                │
            └─────────┬──────────────────────┘
                      ▼
            GraphRAGToolsService
            (quick / panorama / insight_forge / deep_entity)
                      ▼
               ReportAgent (ReACT)
                      ▼
          answer + 📚 Top-5 references + 🧭 source paper
```

## 🔑 Feature Highlights

### Source-paper traceability (added 2026-04)

Every retrieved fact is automatically tagged with the paper title it came from:

```
TURBOQUANT → RABITQ: Consistently outperforms RabitQ in recall ratio across experiments
【Source: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate】
```

In the final answer, the LLM is forced to move `【Source: …】` into each reference's `*(source: …)*` slot — and **explicitly forbidden** from making up titles.

**How it works**: `entity.text_unit_ids → text_units.document_id → documents.text → first markdown H1` = real title. See `backend/app/services/graphrag_tools.py::_build_paper_source_map`.

### Multi-tool ReACT strategy

```python
# Automatically picked from the question
"What is X?"                →  quick_search
"Compare A vs B"            →  quick_search(A) + quick_search(B) + quick_search("A B outperforms")
"Give me a landscape view"  →  panorama_search       (calls global_search + community reports)
"Deep-dive on X"            →  insight_forge         (calls local_search + vector retrieval)
"List every Dataset"        →  deep_entity_query(entity_type="Dataset")
```

For comparison questions, the prompt **enforces at least 3 distinct tool calls** to avoid the agent settling for only 2.

### Anti-stalling

Detects 10+ stalling phrases ("please wait", "let me search", "我将检索", "稍等" …). If the agent uses one but forgot to emit a `<tool_call>`, it is re-prompted with a hard "either emit a tool call now, or give the complete final answer".

### Multi-provider LLM wrapper

`llm_client.py` auto-handles provider quirks:

| Family | Auto-applied |
|---|---|
| GPT-5 | `max_completion_tokens` + `reasoning_effort=low` (chat) / `minimal` (indexing) |
| GLM reasoning | `thinking.type=disabled` |
| Qwen3 | `enable_thinking=false` |
| Kimi | Concurrency throttled to 3 RPS |

## 📦 Data Sharing

Everything is a local Parquet file, so sharing is just `tar`:

```bash
# Recommended: full graph, ~10–20 MB
tar --exclude='cache' --exclude='reporting' \
    -czf share.tgz graphrag_data/weaveragent_<id>/

# Minimal: browsable + keyword-searchable only, ~2 MB
tar -czf minimal.tgz \
    graphrag_data/weaveragent_<id>/output/ \
    graphrag_data/weaveragent_<id>/{meta,ontology}.json
```

The receiver unpacks into their own `graphrag_data/` and is ready to query. Full details in [`reports/work.md §6`](./reports/work.md#6-数据存储与共享).

## 📊 Cost Reporting

Scan all builds and generate a report:

```bash
# Markdown
python reports/gen_markdown.py

# PDF (requires xelatex)
python reports/gen_latex.py && cd reports && xelatex graphrag_cost_report.tex
```

Per-graph breakdown:

```bash
python backend/scripts/token_usage.py /path/to/graphrag_data/weaveragent_xxxxx
```

## 📖 Documentation

| Doc | Content |
|---|---|
| [reports/work.md](./reports/work.md) | Full worklog: architecture, cost report (incl. 100-paper benchmark), roadmap |
| [reports/graphrag_cost_report.pdf](./reports/graphrag_cost_report.pdf) | Authoritative cost comparison across 7 LLMs (PDF) |
| [.env.example](./.env.example) | Environment variable template |
| [start.sh](./start.sh) / [stop.sh](./stop.sh) | One-click start / stop scripts |

## 📄 Acknowledgements

**Standing on the shoulders of giants**:
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — graph construction & retrieval core
- [MiroFish](https://github.com/mirofish) — frontend interaction reference

## 📄 License

AGPL-3.0 — free to use, modify, and redistribute; derivative network services must release their source.
