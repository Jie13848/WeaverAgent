<div align="center">

<img src="./static/image/weaveragent_logo.svg" alt="WeaverAgent Logo" width="75%"/>

**基于 Microsoft GraphRAG 的学术知识图谱分析引擎**
</br>
<em>A GraphRAG-Powered Academic Knowledge Graph Analysis Engine</em>

[English](./README-EN.md) | [中文文档](./README.md) | [完整工作日志](./reports/work.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![GraphRAG](https://img.shields.io/badge/Engine-Microsoft%20GraphRAG-purple)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Vue 3](https://img.shields.io/badge/Vue-3-4FC08D)

</div>

## ⚡ 项目概述

**WeaverAgent** 是一款面向学术研究的知识图谱工作台。上传一批 PDF / Markdown 论文，系统会自动：

1. **本体自动设计** —— LLM 根据论文内容生成领域本体（Paper / Method / Dataset / Metric / Innovation / Task / Baseline / Author）
2. **GraphRAG 构建** —— 切块、抽实体关系、Leiden 社区分层、LLM 生成社区报告
3. **ReACT 问答** —— Agent 自主选择工具、多轮检索，答案里**精确标注每条引用来自哪篇论文**
4. **深度分析报告** —— 自动生成带溯源的技术路径分析

> 一句话：**你只需要上传 PDF + 用自然语言提问，系统负责把学术文献变成可交互、可溯源的知识图谱。**

### 核心能力

- 🧠 **自动本体设计** —— 无需预先定义 schema，LLM 读一遍论文自己决定要抽什么
- 🏗️ **完整 GraphRAG** —— 本地运行 Microsoft GraphRAG（无云依赖），所有产物都是可分享的 Parquet 文件
- 📎 **论文来源回溯** —— 每条事实自动附 `【来源: 论文标题】`，杜绝幻觉
- 🔄 **ReACT 多工具** —— `quick_search` / `panorama_search` / `insight_forge` / `deep_entity_query`，按问题类型自动选
- 🛡️ **反"打白条"机制** —— 检测到 LLM 敷衍就强制重推
- 🌐 **多供应商 LLM** —— GPT-5 / GPT-4o / GLM / Qwen / Kimi，只要兼容 OpenAI SDK 都能接
- 💰 **Token 成本统计** —— 内置脚本，扫一眼就知道这图谱花了多少钱

## 🔄 工作流程

5 步向导式 UI（对应 `Step1GraphBuild.vue` … `Step5Interaction.vue`）：

1. **图谱构建** —— 上传 PDF → LLM 生成 ontology → GraphRAG 索引 → Parquet + LanceDB
2. **环境搭建** —— 可视化检查实体/关系 → 统计图谱规模 → 配置分析参数
3. **图谱分析** —— 节点类型分布 → 枢纽节点识别 → 技术路径抽取
4. **报告生成** —— ReportAgent 多轮 ReACT 检索 → 分章节生成 → 导出 Markdown/PDF
5. **深度互动** —— 自然语言对话 → Agent 自主调工具 → 带论文来源的答案

详细系统架构见 [`reports/work.md`](./reports/work.md)。

## 🚀 快速开始

### 前置要求

| 工具 | 版本 | 说明 | 安装检查 |
|------|------|------|---------|
| **Node.js** | ≥ 18 | 前端运行环境 | `node -v` |
| **Python** | 3.11 – 3.12 | 后端运行环境 | `python --version` |
| **uv** | latest | Python 包管理器（推荐） | `uv --version` |

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 LLM API key
```

#### 必填

```env
# 任意 OpenAI SDK 兼容的 LLM API
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-5-mini
```

#### 推荐：Embedding 独立配置（可与 LLM 不同供应商）

```env
EMBEDDING_API_KEY=sk-xxx               # 不填则回退到 LLM_API_KEY
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-large   # 3072 维，推荐
```

#### 可选

```env
# 图谱数据目录（默认 ../graphrag_data）
GRAPHRAG_DATA_DIR=/path/to/graphrag_data

# ReportAgent 行为
REPORT_AGENT_MAX_TOOL_CALLS=5         # 单轮对话最多调几次工具
REPORT_AGENT_MAX_REFLECTION_ROUNDS=2
REPORT_AGENT_TEMPERATURE=0.5
```

#### LLM 选型速查

| 场景 | 推荐模型 | 理由 |
|---|---|---|
| 大批量索引 (>100 篇) | `gpt-4o-mini` / `qwen3.5-plus` | 每篇 < ¥0.01 |
| 小批量 + 高质量 | `gpt-5` + `text-embedding-3-large` | 社区报告更细，召回更准 |
| 国产合规 | `glm-5-turbo` / `qwen-plus` | 数据留在国内 |
| 长文档 | `kimi-k2-turbo-preview` | 128K 上下文 |

完整成本对比（含 **2026-04-22 百篇**基准库 token / 费用 / 墙钟）见 [`reports/work.md §5`](./reports/work.md#5-graphrag-多模型构建成本对比)。

### 2. 安装依赖

```bash
# 一键装（推荐）
npm run setup:all

# 或分步
npm run setup          # Node 部分（根目录 + frontend）
npm run setup:backend  # Python 部分（uv sync，自动创建 .venv）
```

> ⚠️ 本项目内嵌了 Microsoft GraphRAG 的 monorepo（在 `graphrag/packages/` 下），会作为 editable 包安装到后端 venv。如果启动时报 `ModuleNotFoundError: graphrag`，脚本会自动修复（见 `start.sh`）。

### 3. 启动服务

```bash
# 同时启前后端
npm run dev

# 或用 shell 脚本（后台运行 + 自动重启）
./start.sh
./stop.sh
```

**服务地址**：
- 前端：http://localhost:3000
- 后端 API：http://localhost:5001

单独启动：
```bash
npm run backend   # 仅后端
npm run frontend  # 仅前端
```

### 4. Docker 部署

```bash
cp .env.example .env
docker compose up -d
```

端口：`3000`（前端）/ `5001`（后端），自动读取根目录 `.env`。

## 🏗️ 技术架构

| 层级 | 技术栈 |
|------|--------|
| 前端 | Vue 3 + Vue Router 4 + Vite + D3.js（图谱可视化） |
| 后端 | Flask 3 + Flask-CORS |
| **图谱引擎** | **Microsoft GraphRAG**（内嵌 monorepo，本地运行） |
| 向量库 | LanceDB（GraphRAG 原生）+ ChromaDB（可选 RAG 段落检索） |
| LLM | OpenAI SDK 兼容（GPT-5 reasoning / GLM / Qwen / Kimi / GPT-4o 等） |
| Embedding | `text-embedding-3-large` (3072d) / `text-embedding-3-small` / `text-embedding-v3` / `embedding-3` |
| 数据存储 | **纯本地文件**（Parquet + LanceDB + JSON），可 tar 打包分享 |
| 文件处理 | PyMuPDF（PDF）+ charset-normalizer |
| 可选图数据库 | Neo4j（未来用于高级图算法） |

### 数据流

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
          答案 + 📚 Top-5 参考 + 🧭 论文来源
```

## 🔑 亮点功能

### 论文来源回溯（2026-04 新增）

每条检索出的事实都自动带上出处论文标题：

```
TURBOQUANT → RABITQ: Consistently outperforms RabitQ in recall ratio across experiments
【来源: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate】
```

LLM 在最终回答时被强制把 `【来源: ...】` 搬到 `*（来源：...）*`，不得编造。

**实现**：`entity.text_unit_ids → text_units.document_id → documents.text` 提取首个 markdown H1 作为真实标题。详见 `backend/app/services/graphrag_tools.py::_build_paper_source_map`。

### ReACT Agent 的多工具策略

```python
# 自动根据问题选择
"X 是什么?"          →  quick_search
"对比 A 和 B"        →  quick_search(A) + quick_search(B) + quick_search(A+B+outperforms)
"领域全景"            →  panorama_search (调 global_search + 社区报告)
"深度分析 X"          →  insight_forge (调 local_search + 向量检索)
"列出所有 Dataset"    →  deep_entity_query(entity_type="Dataset")
```

### 反打白条

检测 LLM 输出里"请稍等 / 我将检索 / let me search"等 10+ 种话术，若命中且未到末轮，强制再推一轮并明令"要么发工具调用，要么直接给答案"。

### 多模型统一封装

`llm_client.py` 根据模型名自动处理差异：

| 家族 | 自动处理 |
|---|---|
| GPT-5 | `max_completion_tokens` + `reasoning_effort=low`（对话）/ `minimal`（索引） |
| GLM 思考版 | `thinking.type=disabled` 关闭推理 |
| Qwen3 | `enable_thinking=false` |
| Kimi | 降低并发（limit 严格） |

## 📦 数据共享

所有图谱都是本地 Parquet 文件，支持三种粒度的共享：

```bash
# 方案 B（推荐）：完整图谱，~10-20 MB
tar --exclude='cache' --exclude='reporting' \
    -czf share.tgz graphrag_data/weaveragent_<id>/

# 方案 A（最小）：只要能用关键词检索 + 浏览，~2 MB
tar -czf minimal.tgz graphrag_data/weaveragent_<id>/output/ \
    graphrag_data/weaveragent_<id>/{meta,ontology}.json
```

接收方解压到自己的 `graphrag_data/` 即可使用，详见 [`reports/work.md §6`](./reports/work.md#6-数据存储与共享)。

## 📊 Token 成本速查

扫描当前所有构建，生成成本报告：

```bash
# Markdown
python reports/gen_markdown.py

# PDF (需 xelatex)
python reports/gen_latex.py && cd reports && xelatex graphrag_cost_report.tex
```

单个图谱的详细消耗：
```bash
python backend/scripts/token_usage.py /path/to/graphrag_data/weaveragent_xxxxx
```

## 📖 文档

| 文档 | 内容 |
|---|---|
| [reports/work.md](./reports/work.md) | 完整工作日志、架构、成本报告（含百篇基准）、路线图 |
| [reports/graphrag_cost_report.pdf](./reports/graphrag_cost_report.pdf) | 多模型构建成本对比（PDF） |
| [.env.example](./.env.example) | 环境变量模板 |
| [start.sh](./start.sh) / [stop.sh](./stop.sh) | 一键启动/停止脚本 |

## 📄 开源致谢

**WeaverAgent 站在巨人的肩膀上**：
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) —— 图谱构建与检索核心
- [MiroFish](https://github.com/mirofish) —— 前端交互设计参考

## 📄 许可证

AGPL-3.0 —— 允许自由使用、修改、分发；若作为网络服务提供，必须开源衍生代码。
