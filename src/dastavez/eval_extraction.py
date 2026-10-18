"""Extraction evaluation: pypdf baseline vs layout-aware text layer vs routed OCR cascade."""

from __future__ import annotations

import json
from collections import Counter

from .annotation import STRATA, load_sample, reference_texts
from .ingest import load_pages
from .metrics import bootstrap_ci, cer, mean, similarity

SYSTEMS = {"pypdf": "pypdf_text", "layout": "layout_text", "cascade": "final_text"}


def run_extraction_eval(settings, reference: str = "auto") -> dict:
    ref_kind, refs = reference_texts(settings, reference)
    if not refs:
        raise RuntimeError(
            "no reference transcriptions found (gold/final/*.txt or gold/silver/*.txt); see docs/ANNOTATION_GUIDE.md"
        )
    pages = {(p["doc_id"], p["page"]): p for p in load_pages(settings.data_dir)}
    rows = []
    for item in load_sample(settings):
        sid = item["sample_id"]
        if sid not in refs:
            continue
        rec = pages[(item["doc_id"], item["page"])]
        row = {"sample_id": sid, "stratum": item["stratum"], "route": rec["route"], "reason": rec["reason"]}
        for name, field in SYSTEMS.items():
            hyp = rec.get(field) or ""
            row[f"{name}_similarity"] = round(similarity(hyp, refs[sid]), 4)
            row[f"{name}_cer"] = round(cer(hyp, refs[sid]), 4)
        rows.append(row)

    summary = {}
    for s in (*STRATA, "all"):
        rs = [r for r in rows if s == "all" or r["stratum"] == s]
        entry = {"n": len(rs), "routes": dict(Counter(r["route"] for r in rs))}
        for name in SYSTEMS:
            sims = [r[f"{name}_similarity"] for r in rs]
            ci = bootstrap_ci(sims)
            entry[name] = {
                "mean_similarity": None if not sims else round(mean(sims), 4),
                "similarity_ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
                "mean_cer": None if not rs else round(mean(r[f"{name}_cer"] for r in rs), 4),
            }
        summary[s] = entry

    lines = [
        f"Reference: **{ref_kind}**"
        + ("  (PROVISIONAL: not human double-annotated gold)" if ref_kind == "silver" else ""),
        "",
        "| Stratum | n | pypdf sim | layout sim | cascade sim | pypdf CER | cascade CER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in (*STRATA, "all"):
        e = summary[s]
        if not e["n"]:
            continue

        def f(x):
            return "-" if x is None else f"{x:.3f}"

        lines.append(
            f"| {s} | {e['n']} | {f(e['pypdf']['mean_similarity'])} | {f(e['layout']['mean_similarity'])} | "
            f"{f(e['cascade']['mean_similarity'])} | {f(e['pypdf']['mean_cer'])} | {f(e['cascade']['mean_cer'])} |"
        )
    table = "\n".join(lines)
    out = {"reference": ref_kind, "summary": summary, "per_page": rows, "summary_table": table}
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / f"extraction_eval_{ref_kind}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (settings.results_dir / f"extraction_eval_{ref_kind}.md").write_text(table + "\n", encoding="utf-8")
    return out
