"""
统计 GraphRAG 图谱的 token 消耗和费用

用法:
  python token_usage.py <graph_dir>
  python token_usage.py /path/to/graphrag_data/weaveragent_xxxxx

定价（阿里百炼 DashScope 官方价）:
  qwen-plus / qwen3.5-plus: 输入 ¥0.0008/1K, 输出 ¥0.002/1K
  qwen-max:                 输入 ¥0.02/1K,   输出 ¥0.06/1K
  qwen-turbo:               输入 ¥0.0003/1K, 输出 ¥0.0006/1K
  text-embedding-v3:        ¥0.0007/1K
  gpt-4o-mini (OpenAI):     输入 $0.15/1M,  输出 $0.60/1M
  text-embedding-3-small:   $0.02/1M
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict


PRICING = {
    # DashScope 价格（人民币元 / 1K tokens）
    "qwen3.5-plus":          {"input": 0.0008, "output": 0.002,  "currency": "CNY"},
    "qwen3-plus":            {"input": 0.0008, "output": 0.002,  "currency": "CNY"},
    "qwen-plus":             {"input": 0.0008, "output": 0.002,  "currency": "CNY"},
    "qwen-max":              {"input": 0.02,   "output": 0.06,   "currency": "CNY"},
    "qwen-turbo":            {"input": 0.0003, "output": 0.0006, "currency": "CNY"},
    "text-embedding-v3":     {"input": 0.0007, "output": 0.0,    "currency": "CNY"},
    # Moonshot Kimi 国际版（美元 / 1K tokens）— 需自行核对
    "kimi-k2-turbo-preview": {"input": 0.0006, "output": 0.0025, "currency": "USD"},
    "kimi-k2-0905-preview":  {"input": 0.0006, "output": 0.0025, "currency": "USD"},
    "kimi-k2.5":             {"input": 0.0015, "output": 0.0025, "currency": "USD"},
    "moonshot-v1-32k":       {"input": 0.012,  "output": 0.012,  "currency": "CNY"},
    "moonshot-v1-8k":        {"input": 0.012,  "output": 0.012,  "currency": "CNY"},
    # 智谱 AI 价格（人民币元 / 1K tokens）— 已对齐官方/反推 2026 实测价
    # 反推依据: 用户后台显示 ¥1.6 vs cache 中 166K input + 34K output
    # → input ≈ ¥0.005, output ≈ ¥0.015（与官方 $1.2/$4 per million × 6.5 汇率一致）
    "glm-5-turbo":           {"input": 0.005,  "output": 0.015,  "currency": "CNY"},
    "glm-4.5":               {"input": 0.005,  "output": 0.015,  "currency": "CNY"},
    "glm-4-plus":            {"input": 0.05,   "output": 0.05,   "currency": "CNY"},
    "glm-4-air":             {"input": 0.0005, "output": 0.0005, "currency": "CNY"},
    "glm-4-airx":            {"input": 0.01,   "output": 0.01,   "currency": "CNY"},
    "glm-4-flash":           {"input": 0.0,    "output": 0.0,    "currency": "CNY"},
    "glm-4-flashx":          {"input": 0.0001, "output": 0.0001, "currency": "CNY"},
    "embedding-2":           {"input": 0.0005, "output": 0.0,    "currency": "CNY"},
    "embedding-3":           {"input": 0.0005, "output": 0.0,    "currency": "CNY"},
    # OpenAI 价格（美元 / 1K tokens）
    "gpt-4o":                {"input": 0.0025, "output": 0.01,   "currency": "USD"},
    "gpt-4o-mini":           {"input": 0.00015,"output": 0.0006, "currency": "USD"},
    # GPT-5 系列：reasoning_tokens 计入 output，按 output 单价收费
    "gpt-5":                 {"input": 0.00125,"output": 0.01,   "currency": "USD"},
    "gpt-5-mini":            {"input": 0.00025,"output": 0.002,  "currency": "USD"},
    "gpt-5-nano":            {"input": 0.00005,"output": 0.0004, "currency": "USD"},
    "text-embedding-3-small":{"input": 0.00002,"output": 0.0,    "currency": "USD"},
    "text-embedding-3-large":{"input": 0.00013,"output": 0.0,    "currency": "USD"},
    "text-embedding-ada-002":{"input": 0.0001, "output": 0.0,    "currency": "USD"},
}


def walk_cache_files(cache_dir: Path):
    for root, _, files in os.walk(cache_dir):
        for f in files:
            yield Path(root) / f


def parse_cache(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    result = data.get("result", {})
    response = result.get("response") or {}
    usage = response.get("usage") or {}
    model = response.get("model") or "unknown"

    return {
        "model": model,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "folder": file_path.parent.name,
    }


def main(graph_dir: str):
    graph_path = Path(graph_dir).resolve()
    cache_dir = graph_path / "cache"

    if not cache_dir.exists():
        print(f"❌ cache 目录不存在: {cache_dir}")
        sys.exit(1)

    print(f"📊 统计图谱: {graph_path.name}")
    print(f"📁 缓存目录: {cache_dir}")
    print()

    by_stage = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    by_model = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    grand_total = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    for f in walk_cache_files(cache_dir):
        rec = parse_cache(f)
        if rec is None or rec["total_tokens"] == 0:
            continue

        stage = rec["folder"]
        model = rec["model"]

        for d in (by_stage[stage], by_model[model], grand_total):
            d["calls"] += 1
            d["prompt"] += rec["prompt_tokens"]
            d["completion"] += rec["completion_tokens"]
            d["total"] += rec["total_tokens"]

    print("=" * 78)
    print(f"{'阶段':<30}{'调用次数':>10}{'输入tokens':>14}{'输出tokens':>14}{'总tokens':>10}")
    print("-" * 78)
    for stage, v in sorted(by_stage.items(), key=lambda x: -x[1]["total"]):
        print(f"{stage:<30}{v['calls']:>10,}{v['prompt']:>14,}{v['completion']:>14,}{v['total']:>10,}")
    print("-" * 78)
    print(f"{'合计':<30}{grand_total['calls']:>10,}{grand_total['prompt']:>14,}{grand_total['completion']:>14,}{grand_total['total']:>10,}")
    print("=" * 78)

    print()
    print("💰 费用估算（按模型分组）:")
    print("=" * 78)
    total_cny = 0.0
    total_usd = 0.0

    def _resolve_price(model_name: str):
        if model_name in PRICING:
            return PRICING[model_name]
        # 去掉日期后缀（如 gpt-4o-mini-2024-07-18 → gpt-4o-mini）
        parts = model_name.split('-')
        while parts:
            candidate = '-'.join(parts)
            if candidate in PRICING:
                return PRICING[candidate]
            parts.pop()
        return None

    for model, v in sorted(by_model.items(), key=lambda x: -x[1]["total"]):
        price = _resolve_price(model)
        if not price:
            print(f"  ⚠ 未知模型定价: {model}（{v['total']:,} tokens，无法估算）")
            continue

        cost_input = v["prompt"] / 1000.0 * price["input"]
        cost_output = v["completion"] / 1000.0 * price["output"]
        cost = cost_input + cost_output
        currency = price["currency"]
        symbol = "¥" if currency == "CNY" else "$"

        if currency == "CNY":
            total_cny += cost
        else:
            total_usd += cost

        print(f"  {model}")
        print(f"    {v['calls']:,} 次调用 │ 输入 {v['prompt']:,} tokens ({symbol}{cost_input:.4f}) │ 输出 {v['completion']:,} tokens ({symbol}{cost_output:.4f})")
        print(f"    小计: {symbol}{cost:.4f}")
        print()

    print("=" * 78)
    if total_cny > 0:
        print(f"  💴 合计（人民币）: ¥{total_cny:.4f}")
    if total_usd > 0:
        print(f"  💵 合计（美元）:   ${total_usd:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        base = "/Users/wangkeke/Desktop/WRDS/graphrag_data"
        subdirs = sorted(Path(base).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not subdirs:
            print("❌ 未指定图谱目录")
            sys.exit(1)
        print(f"ℹ 使用最新图谱: {subdirs[0].name}\n")
        main(str(subdirs[0]))
    else:
        main(sys.argv[1])
