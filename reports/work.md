# WeaverAgent 工作日志与系统报告

> **项目代号**：WeaverAgent　|　**定位**：基于 Microsoft GraphRAG 的学术论文知识图谱构建与分析引擎
> **最后更新**：2026-04-22　|　**当前主用模型**：GPT-5 + `text-embedding-3-large` (3072d)
> **许可证**：AGPL-3.0

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 关键功能演进](#3-关键功能演进)
- [4. 支持的 LLM / Embedding 模型](#4-支持的-llm--embedding-模型)
- [5. GraphRAG 多模型构建成本对比](#5-graphrag-多模型构建成本对比)
- [6. 数据存储与共享](#6-数据存储与共享)
- [7. 开发过程中的关键问题与解决方案](#7-开发过程中的关键问题与解决方案)
- [8. 路线图](#8-路线图)

---

## 1. 项目概述

**WeaverAgent** 是一个面向学术研究的知识图谱工作台。你上传一批 PDF / Markdown 论文，系统会：

1. 用 LLM 自动生成论文领域本体（Paper / Method / Dataset / Metric / Innovation / Task / Baseline / Author 等）
2. 用 Microsoft GraphRAG 把论文切块、抽实体和关系、构建社区报告
3. 通过 ReACT Agent（可多轮调用工具）回答问题，并在答案里精确标注**每条引用来自哪篇论文**
4. 生成带溯源的技术路径分析报告

**与上游 GraphRAG 的差异**：本项目把 GraphRAG 包裹成一个完整的 Web 产品，新增了
- 前端可视化（Vue 3 + D3.js）
- 多供应商 LLM 适配（OpenAI / GPT-5 reasoning / 智谱 GLM / 阿里 Qwen / Moonshot Kimi）
- 按查询意图的多策略检索（`quick_search` / `panorama_search` / `insight_forge` / `deep_entity_query`）
- 论文来源回溯（每条事实自动附 `【来源: XXX】`）
- Token/金额成本统计脚本

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3 + Vite)                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │
│  │ Step1   │ │ Step2   │ │ Step3   │ │ Step4   │ │ Step5       │   │
│  │ 图谱构建 │ │ 环境搭建 │ │ 图谱分析 │ │ 报告生成 │ │ 深度问答    │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                       HTTP / Server-Sent Events
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                         Backend (Flask 3)                            │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐     │
│  │ graph API  │  │ report API │  │ rag API (文档全文检索)     │     │
│  └─────┬──────┘  └─────┬──────┘  └───────────┬────────────────┘     │
│        │               │                      │                      │
│  ┌─────▼───────────────▼──────────────────────▼───────────────┐     │
│  │                      Service Layer                          │     │
│  │                                                             │     │
│  │  OntologyGenerator  ──►  TextProcessor  ──►  GraphBuilder  │     │
│  │     (LLM)                 (PDF→MD)            (GraphRAG)   │     │
│  │                                                             │     │
│  │             ┌───────────────────────────────┐              │     │
│  │             │     GraphRAGToolsService       │              │     │
│  │             │  ┌──────────┐  ┌────────────┐ │              │     │
│  │             │  │ parquet  │  │ lancedb    │ │              │     │
│  │             │  │ 关键词   │  │ 向量检索   │ │              │     │
│  │             │  └──────────┘  └────────────┘ │              │     │
│  │             └───────────────┬────────────────┘              │     │
│  │                             │                                │     │
│  │                  ┌──────────▼──────────┐                    │     │
│  │                  │   ReportAgent        │                    │     │
│  │                  │   (ReACT loop)       │                    │     │
│  │                  │  tools_description + │                    │     │
│  │                  │  sandbagging detect  │                    │     │
│  │                  └──────────────────────┘                    │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │               LLMClient (OpenAI SDK compatible)             │     │
│  │   GPT-5 / GPT-4o / GLM / Qwen / Kimi  (统一封装)           │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                   Storage (本地文件系统，无云依赖)                   │
│                                                                      │
│  graphrag_data/<graph_id>/                                          │
│  ├── input/              原始 markdown（每篇论文一个 .txt）          │
│  ├── settings.yaml       GraphRAG per-graph 配置                    │
│  ├── ontology.json       LLM 生成的本体（实体类型列表）              │
│  ├── meta.json           图谱元信息（创建时间等）                    │
│  ├── output/             ★ 核心产物（Parquet）                      │
│  │   ├── entities.parquet         节点（含 text_unit_ids）           │
│  │   ├── relationships.parquet    边（含 text_unit_ids）             │
│  │   ├── text_units.parquet       文本切块 → document_id             │
│  │   ├── documents.parquet        文档（title + raw text）           │
│  │   ├── communities.parquet      Leiden 社区分层                    │
│  │   └── community_reports.parquet 每个社区的 LLM 总结               │
│  ├── lancedb/             向量索引（entity description embeddings） │
│  ├── cache/               LLM 调用缓存（可复现构建，~80-95% 存储） │
│  └── reporting/           构建日志                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键路径：**检索 → 答案**

```
用户问题
  │
  ▼
ReportAgent.chat()  [最多 5 轮 ReACT 迭代]
  │
  ├── round 1: LLM 决定调什么工具
  │     │
  │     ▼
  │   GraphRAGToolsService
  │     ├─ quick_search(query, limit)    # 纯 Parquet 关键词匹配（最快）
  │     ├─ panorama_search(query)         # + global_search 社区报告
  │     ├─ insight_forge(query)           # + local_search 向量感知
  │     └─ deep_entity_query(type, q)     # 按实体类型深度查询
  │     │
  │     ▼
  │   返回 SearchResult
  │     facts = [
  │       "TURBOQUANT → RABITQ: Consistently outperforms... 【来源: TurboQuant: ...】",
  │       ...
  │     ]
  │
  ├── round 2: 对比查询时强制再调 2-3 次（提示词约束）
  │
  └── final round: 按"Top-5 参考条目 + 💡 回答"格式输出
        每条引用后面都带 *（来源：论文全标题）*
```

---

## 3. 关键功能演进

### 3.1 Zep Cloud → Microsoft GraphRAG（重大架构迁移）

**动机**：Zep 每月免费额度有限、网络依赖、云端数据合规性问题。

**迁移后的好处**：
- 全本地运行（除 LLM API 外无任何云依赖）
- 数据可复制、可共享、可版本控制（见 §6）
- 支持完整 GraphRAG 能力栈：local/global/drift/basic 搜索、Leiden 社区分层、社区报告
- 接口完全兼容：`ZepToolsService` → `GraphRAGToolsService` 作为别名，上层调用无改动

**相关文件**：
- `backend/app/services/graphrag_builder.py` — 构建流水线，负责生成 `settings.yaml`、per-graph `.env`、调用 `graphrag.api.index.build_index`
- `backend/app/services/graphrag_tools.py` — 检索服务
- `graphrag/packages/*` — Microsoft GraphRAG monorepo，以 editable 方式安装

### 3.2 多供应商 LLM 适配（统一 OpenAI SDK 格式）

核心文件：`backend/app/utils/llm_client.py`

根据模型名自动选择 API 参数：

| 模型家族 | 特殊处理 |
|---|---|
| **GPT-5 系列** | 不支持自定义 temperature；用 `max_completion_tokens`；设 `reasoning_effort=low`（对话）/`minimal`（索引） |
| **GLM 思考模型** | `extra_body.thinking.type=disabled` 关闭推理（否则每次多烧 10-20× token） |
| **Qwen3 系列** | `extra_body.enable_thinking=false`（否则 JSON 模式报错） |
| **Kimi** | 非思考模型 → 原生参数；限流严格 → concurrent_requests=3 |
| **其他 OpenAI 兼容** | 原生参数 |

### 3.3 论文来源回溯（2026-04-21 新增）

每一条检索到的实体/关系自动带上出处论文标题。

**实现路径**（`graphrag_tools.py` 中的 `_build_paper_source_map`）：
```
entity.text_unit_ids  ──►  text_units.document_id  ──►  documents.text
                                                         └─► 提取首个 markdown H1 作为论文真实标题
```

示例输出：
```
TURBOQUANT → RABITQ: Consistently outperforms RabitQ in recall ratio across experiments
【来源: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate】
```

LLM 被提示必须把 `【来源: ...】` 搬运到最终答复的 `*（来源：...）*` 字段，**不得编造**。

### 3.4 ReACT Agent 的"打白条"检测

**问题**：LLM 有时会写"好的，我来搜一下…"然后停下来不发 `<tool_call>`，导致 Agent 提前返回空白。

**解决**（`report_agent.py` 的 `chat()`）：
- 正则匹配"请稍等 / 我将检索 / let me search / please hold on" 等 10+ 种典型话术
- 若命中且不是最后一轮，**强制重推一轮**，系统消息明令必须立刻给工具调用或完整答案
- 配合 Prompt 里加粗的"严禁打白条"条款，基本杜绝此问题

### 3.5 对比/比较问题的专用策略

**问题**：问 `compare A vs B` 时 Agent 只调 2 次工具就收工，导致漏掉图里明明存在的对比边（例如 TurboQuant → RaBitQ 的 `outperforms` 边）。

**解决**：在 `CHAT_SYSTEM_PROMPT_TEMPLATE` 加了【对比/比较问题的专用策略】，强制调用至少 3 次：
1. `quick_search("A")` — 搜 A
2. `quick_search("B")` — 搜 B
3. `quick_search("A B outperforms baseline comparison")` — 直接找对比边

同时把 `MAX_TOOL_CALLS_PER_CHAT` 从 3 → 4，`max_iterations` 从 4 → 5。

### 3.6 Embedding 维度升级（1536d → 3072d）

切到 `text-embedding-3-large` 完整 3072 维后：
- 向量检索召回率更高，对"同义词"查询（如 TurboQuant / TQ / turbo quant）更友好
- LanceDB 在维度变化时自动重建索引（`graphrag_builder.py` 里有兼容代码）
- 成本影响极小（embedding 占总账 < 5%）

---

## 4. 支持的 LLM / Embedding 模型

### 4.1 LLM

| 模型 | 供应商 | 适用场景 | 调用特点 |
|---|---|---|---|
| `gpt-5` / `gpt-5-mini` / `gpt-5-nano` | OpenAI | 高质量图谱构建 + 深度问答 | reasoning 模型；`reasoning_effort` 可调 |
| `gpt-4o` | OpenAI | 最高质量报告（成本 40× gpt-4o-mini） | 原生 |
| `gpt-4o-mini` | OpenAI | 性价比之王（800 篇约 ¥1.5） | 原生 |
| `qwen-plus` / `qwen3.5-plus` | 阿里 DashScope | 国产快速构建 | 需关闭 thinking |
| `glm-5-turbo` / `glm-4.5` | 智谱 AI | 国产合规 | 需关闭 thinking |
| `kimi-k2-turbo-preview` | Moonshot Kimi | 长上下文 | 限流严格 |

### 4.2 Embedding

| 模型 | 维度 | 供应商 | 成本/1K tokens |
|---|---:|---|---|
| `text-embedding-3-large` | **3072** | OpenAI | $0.00013 |
| `text-embedding-3-small` | 1536 | OpenAI | $0.00002 |
| `text-embedding-v3` | 1024 | 阿里 DashScope | ¥0.0007 |
| `embedding-3` | 2048 | 智谱 AI | ¥0.0005 |

---

## 5. GraphRAG 多模型构建成本对比

> 说明：数据以 [`reports/graphrag_cost_report.tex`](./graphrag_cost_report.tex) 为权威来源（由 `gen_latex.py` 自动扫描 `graphrag_data/` 生成），外加手工维护的 OpenAI 构建：**2026-04-22** 百篇 Demo（`344ec09767c64996`，见 §5.2）、**2026-04-21** 十篇（`9904a9d92b62`，见 §5.3）。汇率 $1 ≈ ¥6.82。  
> **注**：百篇图谱目录当前在 `WRDS/graphrag_data/` 下；若要让 LaTeX/PDF 报告自动含此行，需保证 `gen_latex.py` 的扫描路径覆盖该目录后重新编译 `graphrag_cost_report.tex`。

**执行摘要**

- 以 [`graphrag_cost_report.tex`](./graphrag_cost_report.tex) 为准：成功产出缓存的构建 **11** 次，文档 **64** 篇，token **12,502,328**，成本 **¥106.45**
- **2026-04-22 百篇基准库**（`weaveragent_344ec09767c64996`，`benchmark_100_md`，GPT-5 + `text-embedding-3-large`）：文档 **100** 篇，token **13,810,442**，墙钟 **5352 s（1h29m12s）**，成本 **$38.3315 ≈ ¥261.42**（`backend/scripts/token_usage.py` 扫 `cache/`，汇率 ¥6.82/USD）
- **在 LaTeX 口径上叠加本次百篇**：文档 **164** 篇，token **26,312,770**，成本约 **¥367.87**（106.45 + 261.42）

### 5.1 批量构建总览（文档 ≥ 5 篇）

> 仅保留具有统计意义的批量构建（单篇/2 篇样本已过滤），并 **追加 2026-04-22 百篇基准库** 一行。历史批量见 [`reports/graphrag_cost_report.tex`](./graphrag_cost_report.tex)。

| # | 构建 ID | LLM 模型 | Embedding | 文档 | 输入 tok | 输出 tok | 耗时(s) | 成本(¥) |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `7d9de8b6f876` | gpt-5-mini | text-embedding-3-small | 10 | 563,094 | 207,528 | — | ¥3.77 |
| 2 | `bae3c7c26ca1` | qwen3.5-plus | — | 5 | 1,531,023 | 333,209 | 987 | ¥7.06 |
| 3 | `1bc04e752fb9` | glm-5-turbo | embedding-3 | 5 | 170,523 | 34,068 | 1378 | ¥1.58 |
| 4 | `47ddbb80f5f5` | glm-5-turbo | embedding-3 | 10 | 407,506 | 92,544 | 3022 | ¥4.05 |
| 5 | `fc867e05c308` | kimi-k2-turbo-preview | — | 5 | 1,088,807 | 95,239 | 3060 | ¥5.88 |
| 6 | `0bb60bf501f1` | gpt-4o-mini | text-embedding-3-small | 5 | 1,105,386 | 65,803 | 446 | ¥1.38 |
| 7 | `d8c85f6284c4` | gpt-4o-mini | text-embedding-3-small | 10 | 481,410 | 86,478 | — | ¥0.83 |
| 8 | `9904a9d92b62` | gpt-5 (2025-08-07) | text-embedding-3-large (3072d) | 10 | 929,099 | 236,368 | 385 | ¥22.50 |
| **9** | **`344ec09767c64996`** | **gpt-5-2025-08-07** | **text-embedding-3-large (3072d)** | **100** | **10,947,935** | **2,862,507** | **5352** | **¥261.42** |
| **合计** | **9** 次 | — | — | **160** | **17,224,783** | **4,013,744** | — | **¥308.47** |

### 5.2 Benchmark 百篇库构建详情（`weaveragent_344ec09767c64996`）

**背景**：`benchmark_100_md/` 全量 **100** 篇论文 Markdown（约 **6.0M** 字符），用于公开 Demo 的冻结语料。

**基本信息**

| 项 | 值 |
|---|---|
| 开始时间 | 2026-04-22 17:00:21 |
| 完成时间 | 2026-04-22 18:29:33 |
| 墙钟耗时 | **5352 s**（**1 小时 29 分 12 秒**） |
| 文档数 | 100 |
| LLM 模型 | `gpt-5-2025-08-07`（cache 记录名；计价同 `gpt-5`） |
| Embedding | `text-embedding-3-large`（3072 维） |
| GraphRAG | 10 个 workflow 全部成功 |
| 图谱规模 | **6705** 节点，**7574** 边 |

**Token 分布（按阶段，`token_usage.py` 扫 `cache/`）**

| 阶段 | 调用次数 | 输入 | 输出 | 总计 | 占比 |
|---|---:|---:|---:|---:|---:|
| community_reporting | 1,123 | 3,696,106 | 1,680,883 | 5,376,989 | 38.9% |
| extract_graph | 1,402 | 3,266,968 | 820,014 | 4,086,982 | 29.6% |
| text_embedding | 850 | 3,552,198 | 0 | 3,552,198 | 25.7% |
| summarize_descriptions | 2,160 | 432,663 | 361,610 | 794,273 | 5.7% |
| **合计** | **5,535** | **10,947,935** | **2,862,507** | **13,810,442** | 100% |

**成本明细（与 `token_usage.py` 内置价一致）**

| 模型 | 调用 | 输入成本 | 输出成本 | 小计 |
|---|---:|---:|---:|---:|
| `gpt-5-2025-08-07` | 4,685 | $9.2447 | $28.6251 | **$37.8697** |
| `text-embedding-3-large` | 850 | $0.4618 | $0 | **$0.4618** |
| **合计** | **5,535** | $9.7065 | $28.6251 | **$38.3315 ≈ ¥261.42** |

**磁盘占用（本机构建产物）**

| 子目录 | 大小 |
|---|---:|
| `cache/` | ~1.2 GB |
| `lancedb/` | ~111 MB |
| `output/` | ~21 MB |
| `input/` | ~6 MB |

**关键观察**：
- 百篇 × GPT-5 × 大社区规模 → `community_reporting` 调用 **1123** 次，与十篇档（100 次）呈数量级差异。
- 墙钟 **~89 min** 完成全流程，适合「夜间挂机一次出 Demo 包」；费用约 **¥261** / 次（以官方价估算），公网 Demo 需配合限流以免问答阶段 API 被刷爆。

### 5.3 小规模 GPT-5 + 3072d 构建详情（`weaveragent_9904a9d92b62`）

**基本信息**

| 项 | 值 |
|---|---|
| 完成时间 | 2026-04-21 00:12 |
| 文档数 | 10（arxiv 向量量化/LLM 压缩主题论文） |
| LLM 模型 | `gpt-5` (snapshot `2025-08-07`) |
| Embedding 模型 | `text-embedding-3-large` (3072 维完整) |
| 总耗时 | 385 秒 ≈ 6.4 分钟 |

**Token 分布（按阶段）**

| 阶段 | 调用次数 | 输入 | 输出 | 总计 | 占比 |
|---|---:|---:|---:|---:|---:|
| community_reporting（社区报告） | 100 | 318,514 | 139,136 | 457,650 | **39%** |
| extract_graph（抽实体/关系） | 119 | 277,470 | 65,815 | 343,285 | 29% |
| text_embedding（向量化） | 69 | 292,885 | 0 | 292,885 | 25% |
| summarize_descriptions（摘要） | 200 | 40,230 | 31,417 | 71,647 | 6% |
| **合计** | **488** | **929,099** | **236,368** | **1,165,467** | 100% |

**成本明细**

| 模型 | 调用 | 输入成本 | 输出成本 | 小计 |
|---|---:|---:|---:|---:|
| `gpt-5` (2025-08-07 snapshot) | 419 | $0.80 | $2.36 | **$3.16** |
| `text-embedding-3-large` | 69 | $0.04 | $0 | **$0.04** |
| **合计** | **488** | $0.84 | $2.36 | **$3.20 ≈ ¥22.50** |

**关键观察**：
- community_reporting 吃掉 39%，因为 GPT-5 对每个社区要写一段结构化 JSON 报告（10 篇论文 → 100 个社区报告）
- text_embedding 只有 0 输出 token 但占 25% 输入 —— 是 3072d 向量的代价（比 small 贵 6.5×）
- 10 篇论文 ¥22.50，单篇 ¥2.25，**比 `gpt-4o-mini` 贵 ~15×**，换来更高质量的社区分析和向量召回

### 5.4 单位文档成本对比（≥ 5 篇批量）

| 构建 | LLM 模型 | 文档数 | 总成本(¥) | 每文档(¥) |
|---|---|---:|---:|---:|
| `d8c85f6284c4` | gpt-4o-mini | 10 | ¥0.83 | **¥0.083** ★ 最便宜 |
| `0bb60bf501f1` | gpt-4o-mini | 5 | ¥1.38 | ¥0.276 |
| `1bc04e752fb9` | glm-5-turbo | 5 | ¥1.58 | ¥0.316 |
| `7d9de8b6f876` | gpt-5-mini | 10 | ¥3.77 | ¥0.377 |
| `47ddbb80f5f5` | glm-5-turbo | 10 | ¥4.05 | ¥0.405 |
| `fc867e05c308` | kimi-k2-turbo-preview | 5 | ¥5.88 | ¥1.176 |
| `bae3c7c26ca1` | qwen3.5-plus | 5 | ¥7.06 | ¥1.412 |
| **`9904a9d92b62`** | **gpt-5 + 3072d** | **10** | **¥22.50** | **¥2.250** |
| **`344ec09767c64996`** | **gpt-5 + 3072d** | **100** | **¥261.42** | **¥2.614** |
| `55f5bb2516d2` | gpt-4o | 2 | ¥55.47 | ¥27.735 ⚠️ 最贵（仅 2 篇样本，不具代表性） |

**结论**：

- 在 ≥5 篇批量场景下，最便宜的是 `gpt-4o-mini`（¥0.083/文档），小样本 `gpt-4o` 行仅作警示。
- **同一套 GPT-5 + 3072d**：十篇档单篇 **¥2.25**；百篇档因社区/切块规模上升，单篇 **¥2.61**，仍处在同一数量级（规模经济有限，主要被 `community_reporting` 拖高）。
- `qwen3.5-plus` 虽然单价是人民币计价，但实际输入/输出 token 消耗比预期高 3-4×，所以单篇成本并不便宜。
- `glm-5-turbo` 在思考模型关闭（`thinking.type=disabled`）后，单篇 ¥0.3 左右，国产里算中低档。
- **百篇 Demo**：一次构建约 **¥261**、**~1.5h**；若只做公网只读体验，应预构建后分发 `output/`+`lancedb/`，避免访客触发重建。

### 5.5 选型建议

- **≥30 篇索引** → 默认仍推荐 `gpt-4o-mini` + `text-embedding-3-small`（单篇 ¥0.1 量级）；**若必须 GPT-5 语义质量**，接受 **~¥2.6/篇** 与 **1–2h/百篇** 墙钟成本（见 §5.2）。
- **小批量 + 最高质量** → `gpt-5` + `text-embedding-3-large`（3072d），社区报告详尽
- **成本敏感 + 国产合规** → `glm-5-turbo` + `embedding-3`（需关闭思考）
- **长文档 (>100 页/篇)** → `kimi-k2-turbo-preview`，128K 上下文
- **避免** → `gpt-4o` 做批量索引（单价是 mini 的 16.7×，收益不成比例）
- **Embedding 升级收益递减** → 1536d → 3072d 改善约 5-10%，只有相似概念多时明显

---

## 6. 数据存储与共享

### 6.1 存储完全本地化

所有数据保存在磁盘目录：
```
graphrag_data/<graph_id>/
```

单个图谱典型体积（**十篇级** GPT-5 + 3072d，如 `9904a9d92b62`）：
| 子目录 | 大小 | 说明 |
|---|---:|---|
| `cache/` | ~96 MB | LLM 调用缓存（可复现构建） |
| `lancedb/` | ~9 MB | 向量索引（3072d × 实体数） |
| `output/` | ~1.8 MB | 6 个 Parquet 文件（**核心产物**） |
| `input/` | ~500 KB | 原始 markdown 文本 |
| 其他 | < 300 KB | settings.yaml / ontology.json / 日志 |

**百篇级**（`344ec09767c64996`，同配置）见 §5.2：`cache/` **~1.2 GB**，`lancedb/` **~111 MB**，`output/` **~21 MB**，`input/` **~6 MB**。

### 6.2 三种共享方案

| 方案 | 打包内容 | 体积 | 接收方能做什么 |
|---|---|---:|---|
| **A. 只发 Parquet** | `output/` + `meta.json` + `ontology.json` | ~2 MB | ✅ 关键词检索 / 浏览图谱<br>❌ 不能向量检索 |
| **B. 完整图谱** | A + `lancedb/` + `settings.yaml` + `input/` | 10-20 MB | ✅ 全部检索功能<br>⚠️ 需匹配 embedding 模型维度 |
| **C. 含缓存** | B + `cache/` | 十篇约 ~100 MB；百篇 GPT-5 约 **~1.3 GB** | ✅ 方案 B 全部<br>✅ 可零成本重跑构建 |

**推荐打包命令**（方案 B）：
```bash
tar --exclude='cache' --exclude='reporting' \
    -czf weaveragent_share.tgz weaveragent_<graph_id>/
```

### 6.3 分享注意事项

- 图谱目录**不包含** `.env`，不会泄露 API key
- `cache/` 可能含 prompt / 中间 LLM 响应，涉密场景下打包前删除
- embedding 维度必须和接收方 `.env` 里 `EMBEDDING_MODEL_NAME` 对齐，否则 LanceDB 报 dimension mismatch

---

## 7. 开发过程中的关键问题与解决方案

记录这些是为了给后续贡献者参考，避免重复踩坑。

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'graphrag'` 反复出现 | Anaconda 的 `site.py` 不处理 venv 里 uv 写入的 `.pth` editable 文件 | `backend/run.py` 里硬编码把 `graphrag/packages/*` 插入 `sys.path`；`start.sh` 加兜底重装逻辑 |
| 2 | Agent 返回 "无响应" | LLM 输出格式不符合 `<tool_call>` 解析期望 | 放宽 `_parse_tool_calls` 正则，容忍多种分隔符 |
| 3 | GPT-5 API 报 `temperature must be 1` / 返回空 | GPT-5 是 reasoning 模型，禁自定义 temperature；`max_tokens` 改为 `max_completion_tokens`；reasoning tokens 会吃光预算 | `llm_client.py` 识别 `gpt-5*` 前缀，走专属分支，首次空返回时自动翻倍 `max_completion_tokens` 重试 |
| 4 | Agent "打白条"（说要搜但不发工具调用） | LLM 意图对齐不严 | prompt 明令禁止 + 正则检测 10+ 种话术 + 强制重推 |
| 5 | 对比查询只返回 2 条 | Agent 搜索策略单一，只搜了 A 和 B，没有主动找对比边 | prompt 里新增【对比/比较问题的专用策略】强制 3 次调用 |
| 6 | 切换 embedding 维度后 LanceDB 报错 | 历史向量是 1536d，新 embedding 是 3072d | `graphrag_builder.py` 检测维度变化自动重建 lancedb 表 |
| 7 | LaTeX PDF 表格溢出页面 | 长模型名 + 窄列宽 | 改用 `tabularx` + `minipage` + `shorten_model()` 缩写名字 |

---

## 8. 路线图

### 近期 (1 个月内)
- [ ] 把论文 `source_papers` 暴露到图谱可视化前端（节点 tooltip 里直接显示出处）
- [ ] 支持 drift_search（GraphRAG 最新的混合检索模式）
- [ ] 前端增加"一键导出共享包"按钮（自动调用 §6.2 方案 B）
- [ ] Agent 日志里增加 per-iteration token 统计

### 中期 (3 个月内)
- [ ] 多图谱融合检索（跨领域问答）
- [ ] 本地 LLM 支持（llama.cpp / Ollama 适配）
- [ ] 图谱持久化到对象存储（S3 / OSS / MinIO）
- [ ] Neo4j 作为可选图数据库后端（目前只读 Parquet）

### 长期
- [ ] 主动学习：根据用户反馈迭代 ontology
- [ ] Web 端可视化图谱编辑（增删实体/关系）
- [ ] 论文-代码-数据集联合知识图谱（对接 Papers with Code）

---

## 附录 A. 核心代码文件索引

| 文件 | 职责 |
|---|---|
| `backend/run.py` | Flask 启动入口，注入 graphrag 包路径 |
| `backend/app/config.py` | 从 `.env` 加载配置（LLM/Embedding/GraphRAG 数据目录） |
| `backend/app/services/graphrag_builder.py` | GraphRAG 构建流水线 + `settings.yaml` 生成 |
| `backend/app/services/graphrag_tools.py` | 检索工具（`quick_search` / `panorama_search` / `insight_forge` / `deep_entity_query`） + 论文来源映射 |
| `backend/app/services/report_agent.py` | ReACT Agent（含 sandbagging 检测、对比策略） |
| `backend/app/services/ontology_generator.py` | LLM 自动设计本体 |
| `backend/app/utils/llm_client.py` | OpenAI SDK 封装（多模型适配） |
| `backend/app/prompts/extract_graph.txt` | GraphRAG 抽实体 prompt（严格过滤规则） |
| `backend/scripts/token_usage.py` | Token/金额统计脚本 |
| `frontend/src/views/MainView.vue` | 5 步向导主界面 |
| `frontend/src/components/Step{1..5}*.vue` | 每步的 UI |

## 附录 B. 定价参考（2026-04 快照）

> 下列单价严格对应 [`reports/graphrag_cost_report.tex`](./graphrag_cost_report.tex) 的 §4 定价表。单位：per 1K tokens。

| 模型 | 供应商 | 输入 | 输出 | 货币 |
|---|---|---:|---:|---|
| `gpt-5` (2025-08-07) | OpenAI | $0.00125 | $0.01 | USD |
| `gpt-5-mini` | OpenAI | $0.00025 | $0.002 | USD |
| `gpt-5-nano` | OpenAI | $0.00005 | $0.0004 | USD |
| `gpt-4o` | OpenAI | $0.0025 | $0.01 | USD |
| `gpt-4o-mini` | OpenAI | $0.00015 | $0.0006 | USD |
| `glm-5-turbo` | 智谱 AI | ¥0.005 | **¥0.022** | CNY |
| `qwen3.5-plus` | 阿里 DashScope | **¥0.002** | **¥0.012** | CNY |
| `kimi-k2-turbo-preview` | Moonshot Kimi | **¥0.004** | **¥0.016** | **CNY** |
| `text-embedding-3-large` | OpenAI | $0.00013 | $0 | USD |
| `text-embedding-3-small` | OpenAI | $0.00002 | $0 | USD |
| `embedding-3` | 智谱 AI | ¥0.0005 | ¥0 | CNY |

汇率 $1 ≈ ¥6.82（LaTeX 报告快照）。`backend/scripts/token_usage.py` 内置的定价表可能略有差异，以本表为准。

---

*本报告由 `reports/gen_markdown.py` 自动扫描 `graphrag_data/` 目录生成，并附加手工维护的架构说明、路线图，以及 **2026-04-22 百篇基准库**（`weaveragent_344ec09767c64996`）的 token/费用/耗时（来源：`backend/logs/2026-04-22.log` + `token_usage.py`）。定价为公开官方价，实际账单以供应商后台为准。*
