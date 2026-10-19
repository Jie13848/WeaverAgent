#!/usr/bin/env python3
"""Convert benchmark_100_raw/*.json paper dumps to Markdown for GraphRAG input."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _escape_md_header(s: str) -> str:
    """Avoid accidental ATX heading when section title starts with #."""
    s = s.strip()
    if s.startswith("#"):
        return "\\" + s
    return s


def _authors_md(authors: list) -> str:
    if not authors:
        return ""
    lines = ["## Authors", ""]
    for a in authors:
        if isinstance(a, str) and a.strip():
            lines.append(f"- {a.strip()}")
    lines.append("")
    return "\n".join(lines)


def json_to_markdown(data: dict, source_stem: str) -> str:
    title = (data.get("parsed_title") or data.get("metadata_title") or source_stem).strip()
    corpus_id = data.get("corpus_id")
    arxiv = None
    ext = data.get("parsed_external_ids") or data.get("metadata_externalids") or {}
    if isinstance(ext, dict):
        arxiv = ext.get("arxiv") or ext.get("ArXiv")

    parts: list[str] = []
    parts.append(f"# {title}")
    parts.append("")

    meta_bits = []
    if corpus_id is not None:
        meta_bits.append(f"corpus_id: {corpus_id}")
    if arxiv:
        meta_bits.append(f"arxiv: {arxiv}")
    if meta_bits:
        parts.append("```meta")
        parts.extend(meta_bits)
        parts.append("```")
        parts.append("")

    authors: list[str] = []
    for a in data.get("parsed_authors") or []:
        if isinstance(a, str) and a.strip():
            authors.append(a.strip())
    if not authors and data.get("metadata_authors"):
        for item in data["metadata_authors"]:
            if isinstance(item, dict) and item.get("name"):
                n = str(item["name"]).strip()
                if n:
                    authors.append(n)
    parts.append(_authors_md(authors))

    abstract = (data.get("abstract") or "").strip()
    if abstract:
        parts.append("## Abstract")
        parts.append("")
        parts.append(abstract)
        parts.append("")

    for sec in data.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        st = (sec.get("title") or "").strip()
        content = (sec.get("content") or "").strip()
        if not st and not content:
            continue
        # JSON often duplicates abstract as a section after top-level `abstract`
        if abstract and st.lower() == "abstract":
            continue
        if st:
            parts.append(f"## {_escape_md_header(st)}")
            parts.append("")
        if content:
            parts.append(content)
            parts.append("")

    figures = data.get("figures") or []
    if figures:
        parts.append("## Figures (text descriptions)")
        parts.append("")
        for i, fig in enumerate(figures, 1):
            if not isinstance(fig, dict):
                continue
            cap = (fig.get("caption") or "").strip()
            body = (fig.get("content") or "").strip()
            if not cap and not body:
                continue
            parts.append(f"### Figure {i}")
            parts.append("")
            if cap:
                parts.append(cap)
                parts.append("")
            if body:
                parts.append(body)
                parts.append("")

    refs = data.get("references") or []
    if refs:
        parts.append("## References")
        parts.append("")
        for i, ref in enumerate(refs, 1):
            if isinstance(ref, dict):
                text = (ref.get("text") or "").strip()
                doi = (ref.get("doi") or "").strip()
            else:
                text = str(ref).strip()
                doi = ""
            if not text:
                continue
            line = f"{i}. {text}"
            if doi:
                line += f" (DOI: {doi})"
            parts.append(line)
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    raw_dir = Path(__file__).resolve().parent
    out_dir = raw_dir.parent / "benchmark_100_md"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_dir.glob("bench_*.json"))
    if not json_files:
        print(f"No bench_*.json under {raw_dir}", file=sys.stderr)
        return 1

    for jpath in json_files:
        stem = jpath.stem
        with jpath.open(encoding="utf-8") as f:
            data = json.load(f)
        md = json_to_markdown(data, stem)
        out_path = out_dir / f"{stem}.md"
        out_path.write_text(md, encoding="utf-8")
        print(out_path)

    print(f"Wrote {len(json_files)} files -> {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
