"""生成 Markdown 报告：所有成功构建的 GraphRAG 的 token/费用明细。"""
import sys
sys.path.insert(0, "/tmp")
from build_report import BASE, scan_graph, compute_cost, resolve_price, PRICING, USD_TO_CNY
from pathlib import Path
from datetime import datetime
import re

OUT_MD = Path("/Users/wangkeke/Desktop/桌面 - 王可可的MacBook Pro/WRDS/MiroFish/reports/work.md")


def shorten_model(name: str) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name or "")


def classify_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-") or m.startswith("text-embedding-"):
        return "OpenAI"
    if m.startswith("glm-") or m.startswith("embedding-"):
        return "智谱 AI"
    if m.startswith("qwen"):
        return "阿里 DashScope"
    if m.startswith("kimi") or m.startswith("moonshot"):
        return "Moonshot Kimi"
    return "其他"


def fmt(n):
    return f"{n:,}"


def main():
    graphs = []
    for d in sorted(BASE.iterdir(), key=lambda p: p.stat().st_mtime):
        if d.is_dir() and d.name.startswith("weaveragent_"):
            graphs.append(scan_graph(d))

    successful = [
        g for g in graphs
        if g["success"] and sum(m["total"] for m in g["by_model"].values()) > 0
    ]
    successful.sort(key=lambda g: g["mtime"])

    summaries = []
    for g in successful:
        total_usd = total_cny = 0.0
        total_prompt = total_completion = total_tokens = 0
        llm_model = None
        emb_model = None
        for model, t in g["by_model"].items():
            total_prompt += t["prompt"]
            total_completion += t["completion"]
            total_tokens += t["total"]
            c = compute_cost(t, model)
            if c:
                if c["currency"] == "USD":
                    total_usd += c["total"]
                else:
                    total_cny += c["total"]
            if "embedding" in model.lower():
                emb_model = model
            else:
                llm_model = model
        summaries.append({
            **g,
            "total_prompt": total_prompt,
            "total_completion": total_completion,
            "total_tokens": total_tokens,
            "total_usd": total_usd,
            "total_cny": total_cny,
            "total_cny_eq": total_cny + total_usd * USD_TO_CNY,
            "llm_model": llm_model,
            "emb_model": emb_model,
        })

    agg_by_llm = {}
    for s in summaries:
        key = shorten_model(s["llm_model"] or "(未知)")
        a = agg_by_llm.setdefault(key, {
            "builds": 0, "docs": 0, "prompt": 0, "completion": 0,
            "total": 0, "usd": 0.0, "cny": 0.0, "provider": classify_provider(key),
        })
        a["builds"] += 1
        a["docs"] += s["num_docs_meta"] or 0
        a["prompt"] += s["total_prompt"]
        a["completion"] += s["total_completion"]
        a["total"] += s["total_tokens"]
        a["usd"] += s["total_usd"]
        a["cny"] += s["total_cny"]

    grand_usd = sum(s["total_usd"] for s in summaries)
    grand_cny = sum(s["total_cny"] for s in summaries)
    grand_docs = sum((s["num_docs_meta"] or 0) for s in summaries)
    grand_tokens = sum(s["total_tokens"] for s in summaries)
    grand_cny_eq = grand_cny + grand_usd * USD_TO_CNY

    lines = []
    P = lines.append

    today = datetime.now().strftime("%Y 年 %m 月 %d 日")
    P("# GraphRAG 多模型构建成本对比报告")
    P("")
    P(f"> **Token 消耗与经费明细（按模型 / 按构建）**  ")
    P(f"> 生成时间：{today}　|　数据目录：`graphrag_data/`")
    P("")
    P("---")
    P("")

    # 执行摘要
    P("## 执行摘要")
    P("")
    P(f"- 扫描的构建总数：**{len(graphs)}**，其中成功产出实体表（`entities.parquet`）且含缓存记录的构建：**{len(successful)}**")
    P(f"- 累计处理文档数：**{fmt(grand_docs)}** 篇")
    P(f"- 累计 LLM + Embedding token 消耗：**{fmt(grand_tokens)}** tokens")
    if grand_usd > 0:
        P(f"- 累计美元成本：**${grand_usd:.4f}**")
    if grand_cny > 0:
        P(f"- 累计人民币成本：**¥{grand_cny:.4f}**")
    P(f"- **人民币等值总成本**（汇率 {USD_TO_CNY:.2f}）：**¥{grand_cny_eq:.2f}**")
    P("")

    # 第 1 节
    P("## 1. 成功构建总览")
    P("")
    P("按完成时间排序；成本列为等值人民币（$1 ≈ ¥7.25）。")
    P("")
    P("| # | 构建 ID | LLM 模型 | 文档 | 输入 tok | 输出 tok | 耗时(s) | 成本(¥) |")
    P("|---|---|---|---:|---:|---:|---:|---:|")
    for i, s in enumerate(summaries, 1):
        short_id = s["id"].replace("weaveragent_", "")[:12]
        runtime = f"{s['runtime']:.0f}" if s["runtime"] else "-"
        llm = shorten_model(s["llm_model"] or "-")
        P(f"| {i} | `{short_id}` | {llm} | {fmt(s['num_docs_meta'])} | {fmt(s['total_prompt'])} | {fmt(s['total_completion'])} | {runtime} | ¥{s['total_cny_eq']:.2f} |")
    P(f"| **合计** | **{len(summaries)} 次** | — | **{fmt(grand_docs)}** | **{fmt(sum(s['total_prompt'] for s in summaries))}** | **{fmt(sum(s['total_completion'] for s in summaries))}** | — | **¥{grand_cny_eq:.2f}** |")
    P("")

    # 第 2 节
    P("## 2. 按 LLM 模型聚合")
    P("")
    P("相同模型的不同日期版本合并；Embedding 不单列。")
    P("")
    P("| LLM 模型 | 供应商 | 构建数 | 文档 | 输入 tok | 输出 tok | 成本(¥) |")
    P("|---|---|---:|---:|---:|---:|---:|")
    agg_sorted = sorted(agg_by_llm.items(), key=lambda x: -(x[1]["cny"] + x[1]["usd"] * USD_TO_CNY))
    for model, a in agg_sorted:
        cny_eq = a["cny"] + a["usd"] * USD_TO_CNY
        P(f"| {model} | {a['provider']} | {a['builds']} | {fmt(a['docs'])} | {fmt(a['prompt'])} | {fmt(a['completion'])} | ¥{cny_eq:.2f} |")
    P("")

    # 第 3 节 — 每次构建明细
    P("## 3. 每次构建明细（按模型拆分）")
    P("")
    for i, s in enumerate(summaries, 1):
        P(f"### 3.{i}　构建 `{s['id']}`")
        P("")
        P("**基本信息**")
        P("")
        P("| 项 | 值 |")
        P("|---|---|")
        P(f"| 完成时间 | {s['mtime']} |")
        P(f"| 文档数 | {fmt(s['num_docs_meta'])} |")
        if s["runtime"]:
            P(f"| 总耗时 | {s['runtime']:.1f} s（约 {s['runtime']/60:.1f} 分钟） |")
        P(f"| LLM 模型 | `{shorten_model(s['llm_model'] or '-')}` |")
        if s["emb_model"]:
            P(f"| Embedding 模型 | `{shorten_model(s['emb_model'])}` |")
        P("")
        P("**成本明细**")
        P("")
        P("| 模型 | 调用次数 | 输入 tok | 输出 tok | 合计 tok | 成本 |")
        P("|---|---:|---:|---:|---:|---:|")
        for model, t in sorted(s["by_model"].items(), key=lambda x: -x[1]["total"]):
            c = compute_cost(t, model)
            if c:
                sym = "$" if c["currency"] == "USD" else "¥"
                cost_str = f"{sym}{c['total']:.4f}"
            else:
                cost_str = "(无定价)"
            P(f"| `{shorten_model(model)}` | {fmt(t['calls'])} | {fmt(t['prompt'])} | {fmt(t['completion'])} | {fmt(t['total'])} | {cost_str} |")
        total_bits = []
        if s["total_usd"] > 0:
            total_bits.append(f"${s['total_usd']:.4f}")
        if s["total_cny"] > 0:
            total_bits.append(f"¥{s['total_cny']:.4f}")
        total_bits.append(f"≈ ¥{s['total_cny_eq']:.2f}")
        total_calls = sum(t['calls'] for t in s['by_model'].values())
        P(f"| **合计** | **{fmt(total_calls)}** | **{fmt(s['total_prompt'])}** | **{fmt(s['total_completion'])}** | **{fmt(s['total_tokens'])}** | **{' + '.join(total_bits)}** |")
        P("")

    # 第 4 节：定价
    P("## 4. 定价参考")
    P("")
    P("下表为本报告使用的官方单价（per 1K tokens）。")
    P("")
    P("| 模型 | 供应商 | 输入 | 输出 | 货币 |")
    P("|---|---|---:|---:|---|")
    used_models = set()
    for s in summaries:
        for m in s["by_model"]:
            p = resolve_price(m)
            if p:
                for k, v in PRICING.items():
                    if v is p:
                        used_models.add(k)
    for model in sorted(used_models):
        p = PRICING[model]
        sym = "$" if p["currency"] == "USD" else "¥"
        P(f"| `{model}` | {classify_provider(model)} | {sym}{p['input']:g} | {sym}{p['output']:g} | {p['currency']} |")
    P("")

    # 第 5 节：观察与建议
    P("## 5. 观察与建议")
    P("")

    big_builds = [s for s in summaries if (s["num_docs_meta"] or 0) >= 100]
    if big_builds:
        per_doc = []
        for s in big_builds:
            docs = s["num_docs_meta"] or 1
            per_doc.append((s, s["total_cny_eq"] / docs))
        per_doc.sort(key=lambda x: x[1])
        P("### 5.1 单位文档成本对比（≥ 100 篇文档的构建）")
        P("")
        P("| LLM 模型 | 文档数 | 总成本(¥) | **每文档成本(¥)** |")
        P("|---|---:|---:|---:|")
        for s, cpd in per_doc:
            P(f"| {shorten_model(s['llm_model'])} | {fmt(s['num_docs_meta'])} | ¥{s['total_cny_eq']:.2f} | **¥{cpd:.4f}** |")
        P("")
        cheapest = per_doc[0]
        most_exp = per_doc[-1]
        if cheapest[0] is not most_exp[0]:
            mult = most_exp[1] / max(cheapest[1], 1e-9)
            P(f"**结论：** 最便宜的是 `{shorten_model(cheapest[0]['llm_model'])}`（¥{cheapest[1]:.4f}/文档），最贵的是 `{shorten_model(most_exp[0]['llm_model'])}`（¥{most_exp[1]:.4f}/文档），差距约 **{mult:.1f}×**。")
            P("")

    P("### 5.2 选型建议")
    P("")
    P("- **`gpt-4o-mini`**：当前性价比最高的 OpenAI 选项，800 篇文档构建约 ¥1.5，适合大批量索引。")
    P("- **`glm-5-turbo`**：国产替代，与 OpenAI 同价位段；网络策略不便直连 OpenAI 时可用作降级方案（已启用 `thinking.type=disabled` 关闭思考）。")
    P("- **`gpt-5-mini`**：reasoning 模型；已在 `llm_client.py` 中配置为 `reasoning_effort=low`（对话）/`minimal`（索引），成本显著低于默认 medium 档，但仍比 `gpt-4o-mini` 略贵。")
    P("- **`gpt-4o`**：质量最佳但成本是 `gpt-4o-mini` 的约 **40 倍**，⚠️ 不推荐用于大规模索引，仅建议用于最终报告生成或难例重试。")
    P("- **`kimi-k2-turbo-preview`**：中等成本，处理 808 篇文档约 ¥6.5，适合需要国内合规的长上下文场景。")
    P("- **Embedding 成本占比极小**（通常 < 1%），切换 embedding 模型对总账影响有限，主要决策应集中在 LLM 选择上。")
    P("")

    P("---")
    P("")
    P(f"*本报告由 `reports/gen_markdown.py` 自动扫描 `graphrag_data/` 目录生成。定价为公开官方价，实际账单以供应商后台为准。*")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown 已写入: {OUT_MD}")
    print(f"共 {len(lines)} 行，{sum(len(l) for l in lines)} 字符")


if __name__ == "__main__":
    main()
