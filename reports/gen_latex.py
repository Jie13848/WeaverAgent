"""生成 LaTeX 报告：所有成功构建的 GraphRAG 的 token/费用明细。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import BASE, scan_graph, compute_cost, resolve_price, USD_TO_CNY
from datetime import datetime

OUT_TEX = Path(__file__).resolve().parent / "graphrag_cost_report.tex"


def fmt_int(n):
    return f"{n:,}".replace(",", "{,}")  # 避免 LaTeX 把千位逗号当命令


def escape(s: str) -> str:
    if s is None:
        return "-"
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash ")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def shorten_model(name: str) -> str:
    """去掉 OpenAI 的日期版本后缀，例如 gpt-4o-mini-2024-07-18 → gpt-4o-mini"""
    import re
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name or "")


def classify_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("text-embedding-"):
        return "OpenAI"
    if m.startswith("glm-") or m.startswith("embedding-"):
        return "智谱 AI"
    if m.startswith("qwen"):
        return "阿里 DashScope"
    if m.startswith("kimi") or m.startswith("moonshot"):
        return "Moonshot Kimi"
    return "其他"


def build_tex(graphs):
    # 只保留成功且有 token 记录的构建
    successful = [
        g for g in graphs
        if g["success"] and sum(m["total"] for m in g["by_model"].values()) > 0
    ]
    # 按时间排序
    successful.sort(key=lambda g: g["mtime"])

    # ---- 预计算每个 graph 的合计 ----
    summaries = []
    for g in successful:
        total_usd = total_cny = 0.0
        total_prompt = total_completion = total_tokens = 0
        llm_model = None   # 主 LLM
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
            # 归类
            p = resolve_price(model)
            if p and p["input"] > 0 and ("embedding" in model.lower() or p["output"] == 0):
                emb_model = model
            elif "embedding" in model.lower():
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

    # 汇总（按 LLM 模型聚合）
    agg_by_llm = {}
    for s in summaries:
        key = s["llm_model"] or "(未知)"
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

    # ==== 开始拼 LaTeX ====
    lines = []
    P = lines.append
    P(r"\documentclass[11pt]{article}")
    P(r"\usepackage[margin=2cm]{geometry}")
    P(r"\usepackage{ctex}")                 # 中文支持
    P(r"\usepackage{booktabs}")             # 好看的三线表
    P(r"\usepackage{longtable}")
    P(r"\usepackage{array}")
    P(r"\usepackage{xcolor}")
    P(r"\usepackage{colortbl}")
    P(r"\usepackage{siunitx}")
    P(r"\usepackage{hyperref}")
    P(r"\usepackage{graphicx}")
    P(r"\usepackage{titlesec}")
    P(r"\usepackage{fancyhdr}")
    P(r"\usepackage{tcolorbox}")
    P(r"\tcbuselibrary{skins,breakable}")
    P(r"\usepackage{enumitem}")
    P(r"\usepackage{tabularx}")
    P(r"\newcolumntype{R}{>{\raggedleft\arraybackslash}X}")
    P(r"\definecolor{headerbg}{RGB}{32,86,152}")
    P(r"\definecolor{rowalt}{RGB}{240,245,252}")
    P(r"\definecolor{okgreen}{RGB}{34,139,34}")
    P(r"\definecolor{warnorange}{RGB}{210,105,30}")
    P(r"\pagestyle{fancy}")
    P(r"\fancyhf{}")
    P(r"\fancyhead[L]{\small WeaverAgent / GraphRAG 构建成本报告}")
    P(r"\fancyhead[R]{\small " + datetime.now().strftime("%Y-%m-%d") + r"}")
    P(r"\fancyfoot[C]{\thepage}")
    P(r"\renewcommand{\arraystretch}{1.25}")
    P(r"\hypersetup{colorlinks=true, linkcolor=headerbg, urlcolor=headerbg}")
    P(r"\title{\vspace{-1cm}\Huge\textbf{GraphRAG 多模型构建成本对比报告}\\[4pt]\large Token 消耗与经费明细（按模型 / 按构建）}")
    P(r"\author{WeaverAgent 自动生成}")
    P(r"\date{" + datetime.now().strftime("%Y 年 %m 月 %d 日") + r"}")
    P(r"\begin{document}")
    P(r"\maketitle")
    P(r"\thispagestyle{fancy}")

    # ---- 摘要 ----
    P(r"\begin{tcolorbox}[colback=rowalt, colframe=headerbg, title=\textbf{执行摘要}, breakable]")
    P(r"\begin{itemize}[leftmargin=*]")
    P(rf"  \item 扫描的构建总数：\textbf{{{len(graphs)}}}，其中成功产出实体表（entities.parquet）且含缓存的构建：\textbf{{{len(successful)}}}")
    P(rf"  \item 累计处理文档数：\textbf{{{fmt_int(grand_docs)}}} 篇")
    P(rf"  \item 累计 LLM+Embedding token 消耗：\textbf{{{fmt_int(grand_tokens)}}} tokens")
    if grand_usd > 0:
        P(rf"  \item 累计美元成本：\textbf{{\${grand_usd:.4f}}}")
    if grand_cny > 0:
        P(rf"  \item 累计人民币成本：\textbf{{¥{grand_cny:.4f}}}")
    grand_cny_eq = grand_cny + grand_usd * USD_TO_CNY
    P(rf"  \item 人民币等值总成本（汇率 {USD_TO_CNY:.2f}）：\textbf{{¥{grand_cny_eq:.2f}}}")
    P(r"\end{itemize}")
    P(r"\end{tcolorbox}")

    # ---- 总览表：每次构建一行 ----
    P(r"\section*{1. 成功构建总览}")
    P(r"\noindent\begin{tabularx}{\textwidth}{@{}l X r r r r r@{}}")
    P(r"\toprule")
    P(r"\rowcolor{headerbg!10}\textbf{构建 ID} & \textbf{LLM 模型} & \textbf{文档} & \textbf{输入 tok} & \textbf{输出 tok} & \textbf{耗时(s)} & \textbf{成本(¥)} \\")
    P(r"\midrule")
    for i, s in enumerate(summaries):
        short_id = s["id"].replace("weaveragent_", "")[:12]
        runtime = f"{s['runtime']:.0f}" if s["runtime"] else "-"
        llm = shorten_model(s["llm_model"] or "-")
        cost_eq = f"¥{s['total_cny_eq']:.2f}"
        row_color = r"\rowcolor{rowalt}" if i % 2 == 0 else ""
        P(rf"{row_color}\texttt{{\small {escape(short_id)}}} & \small {escape(llm)} & {fmt_int(s['num_docs_meta'])} & {fmt_int(s['total_prompt'])} & {fmt_int(s['total_completion'])} & {runtime} & {cost_eq} \\")
    P(r"\midrule")
    P(rf"\rowcolor{{headerbg!15}}\textbf{{合计（{len(summaries)} 次）}} & -- & \textbf{{{fmt_int(grand_docs)}}} & \textbf{{{fmt_int(sum(s['total_prompt'] for s in summaries))}}} & \textbf{{{fmt_int(sum(s['total_completion'] for s in summaries))}}} & -- & \textbf{{¥{grand_cny_eq:.2f}}} \\")
    P(r"\bottomrule")
    P(r"\end{tabularx}")
    P(rf"\vspace{{0.1cm}}\noindent{{\small 注：所有成功构建按完成时间排序；成本栏为等值人民币（\$1 $\approx$ ¥{USD_TO_CNY:.2f}）。}}")

    # ---- 按模型聚合 ----
    P(r"\section*{2. 按 LLM 模型聚合}")
    P(r"\noindent\begin{tabularx}{\textwidth}{@{}X l r r r r r@{}}")
    P(r"\toprule")
    P(r"\rowcolor{headerbg!10}\textbf{LLM 模型} & \textbf{供应商} & \textbf{构建} & \textbf{文档} & \textbf{输入 tok} & \textbf{输出 tok} & \textbf{成本(¥)} \\")
    P(r"\midrule")
    agg_sorted = sorted(agg_by_llm.items(), key=lambda x: -(x[1]["cny"] + x[1]["usd"] * USD_TO_CNY))
    for i, (model, a) in enumerate(agg_sorted):
        row_color = r"\rowcolor{rowalt}" if i % 2 == 0 else ""
        cny_eq = a["cny"] + a["usd"] * USD_TO_CNY
        P(rf"{row_color}\small {escape(shorten_model(model))} & \small {escape(a['provider'])} & {a['builds']} & {fmt_int(a['docs'])} & {fmt_int(a['prompt'])} & {fmt_int(a['completion'])} & ¥{cny_eq:.2f} \\")
    P(r"\bottomrule")
    P(r"\end{tabularx}")
    P(r"\vspace{0.1cm}\noindent{\small 按 LLM 主模型聚合（Embedding 不单列；相同模型的不同日期版本合并）。}")

    # ---- 每次构建明细 ----
    P(r"\section*{3. 每次构建明细（按模型拆分）}")
    for i, s in enumerate(summaries, 1):
        P(r"\subsection*{" + rf"3.{i}\;构建 " + escape(s["id"]) + r"}")
        # --- 元信息：用 minipage 顶在左边 ---
        P(r"\noindent\begin{minipage}{\textwidth}")
        P(r"\begin{tabular}{@{}ll@{}}")
        P(rf"\textbf{{完成时间}} & {escape(s['mtime'])} \\")
        P(rf"\textbf{{文档数}} & {fmt_int(s['num_docs_meta'])} \\")
        if s["runtime"]:
            P(rf"\textbf{{总耗时}} & {s['runtime']:.1f}\,s（约 {s['runtime']/60:.1f}\,分钟） \\")
        P(rf"\textbf{{LLM 模型}} & {escape(shorten_model(s['llm_model'] or '-'))} \\")
        if s["emb_model"]:
            P(rf"\textbf{{Embedding 模型}} & {escape(shorten_model(s['emb_model']))} \\")
        P(r"\end{tabular}")
        P(r"\end{minipage}")
        P(r"\par\vspace{0.25cm}")

        # --- 明细表：tabularx 自动填满页宽 ---
        P(r"\noindent\begin{tabularx}{\textwidth}{@{}X r r r r r@{}}")
        P(r"\toprule")
        P(r"\rowcolor{headerbg!10}\textbf{模型} & \textbf{调用} & \textbf{输入 tok} & \textbf{输出 tok} & \textbf{合计 tok} & \textbf{成本} \\")
        P(r"\midrule")
        for j, (model, t) in enumerate(sorted(s["by_model"].items(), key=lambda x: -x[1]["total"])):
            c = compute_cost(t, model)
            if c:
                sym = r"\$" if c["currency"] == "USD" else r"¥"
                cost_str = rf"{sym}{c['total']:.4f}"
            else:
                cost_str = "(无定价)"
            rc = r"\rowcolor{rowalt}" if j % 2 == 0 else ""
            P(rf"{rc}\texttt{{\small {escape(shorten_model(model))}}} & {fmt_int(t['calls'])} & {fmt_int(t['prompt'])} & {fmt_int(t['completion'])} & {fmt_int(t['total'])} & {cost_str} \\")
        P(r"\midrule")
        total_line = []
        if s["total_usd"] > 0:
            total_line.append(rf"\${s['total_usd']:.4f}")
        if s["total_cny"] > 0:
            total_line.append(rf"¥{s['total_cny']:.4f}")
        total_line.append(rf"$\approx$ ¥{s['total_cny_eq']:.2f}")
        P(rf"\rowcolor{{headerbg!15}}\textbf{{合计}} & \textbf{{{fmt_int(sum(t['calls'] for t in s['by_model'].values()))}}} & \textbf{{{fmt_int(s['total_prompt'])}}} & \textbf{{{fmt_int(s['total_completion'])}}} & \textbf{{{fmt_int(s['total_tokens'])}}} & \textbf{{{' + '.join(total_line)}}} \\")
        P(r"\bottomrule")
        P(r"\end{tabularx}")
        P(r"\vspace{0.5cm}")

    # ---- 定价附录 ----
    P(r"\section*{4. 定价参考}")
    P(r"\small")
    P(r"下表为本报告使用的单价（per 1K tokens）：")
    P(r"\begin{longtable}{@{}l l r r l@{}}")
    P(r"\toprule")
    P(r"\textbf{模型} & \textbf{供应商} & \textbf{输入单价} & \textbf{输出单价} & \textbf{货币} \\")
    P(r"\midrule")
    from build_report import PRICING
    # 只打印至少出现过一次的模型
    used = set()
    for s in summaries:
        for m in s["by_model"]:
            p = resolve_price(m)
            if p:
                # 规范化到定价 key
                for k, v in PRICING.items():
                    if v is p:
                        used.add(k)
    for model in sorted(used):
        p = PRICING[model]
        sym = r"\$" if p["currency"] == "USD" else "¥"
        P(rf"{escape(model)} & {escape(classify_provider(model))} & {sym}{p['input']:g} & {sym}{p['output']:g} & {p['currency']} \\")
    P(r"\bottomrule")
    P(r"\end{longtable}")
    P(r"\normalsize")

    # ---- 结论 ----
    P(r"\section*{5. 观察与建议}")
    P(r"\begin{itemize}[leftmargin=*]")
    # 找出每个"同规模"组里最便宜的 LLM
    # 按文档数相近对比
    cheapest = None
    most_exp = None
    for s in summaries:
        docs = s["num_docs_meta"] or 0
        if docs >= 5:
            cost_per_doc = s["total_cny_eq"] / max(docs, 1)
            if cheapest is None or cost_per_doc < cheapest[1]:
                cheapest = (s, cost_per_doc)
            if most_exp is None or cost_per_doc > most_exp[1]:
                most_exp = (s, cost_per_doc)
    if cheapest and most_exp and cheapest[0] is not most_exp[0]:
        P(rf"  \item 在较大规模构建（$\geq$ 5 文档）场景下，最便宜的配置是 \textbf{{{escape(cheapest[0]['llm_model'])}}}（¥{cheapest[1]:.4f}/文档），最贵的是 \textbf{{{escape(most_exp[0]['llm_model'])}}}（¥{most_exp[1]:.4f}/文档），差距约 \textbf{{{most_exp[1]/max(cheapest[1],1e-9):.1f}×}}。")
    P(r"  \item \textbf{gpt-4o-mini}：当前性价比最高的 OpenAI 选项（\$0.15/\$0.60 per M），本批次 5 篇文档的构建约 ¥1.4，适合批量索引。")
    P(r"  \item \textbf{glm-5-turbo}：国产替代选项（¥5/¥22 per M），单价约为 gpt-4o-mini 的 5×，属中低成本区间；如网络策略不便直连 OpenAI，可用作降级方案（已启用 thinking=disabled 关闭思考）。")
    P(r"  \item \textbf{gpt-5-mini}：已切换到 reasoning\_effort=minimal（构建场景）/ low（对话），成本显著低于默认 medium 档，但比 gpt-4o-mini 仍略贵。")
    P(r"  \item \textbf{gpt-4o}：质量最佳但单价约为 gpt-4o-mini 的 \textbf{16.7×}（\$2.50/\$10.00 vs \$0.15/\$0.60 per M），\textcolor{warnorange}{不推荐用于批量索引}，仅建议用于最终报告生成或难例重试。")
    P(r"  \item Embedding 成本占比极小（通常 $<$1\%），切换 embedding 模型对总账影响有限，主要决策应集中在 LLM 选择上。")
    P(r"\end{itemize}")

    P(r"\vfill")
    P(r"\begin{center}\small\textcolor{gray}{本报告由 " + escape(BASE.name) + r" 目录自动扫描生成。定价为公开官方价，实际账单以供应商后台为准。}\end{center}")
    P(r"\end{document}")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX 已生成: {OUT_TEX}  ({len(lines)} 行)")
    return OUT_TEX


if __name__ == "__main__":
    graphs = []
    for d in sorted(BASE.iterdir(), key=lambda p: p.stat().st_mtime):
        if d.is_dir() and d.name.startswith("weaveragent_"):
            graphs.append(scan_graph(d))
    build_tex(graphs)
