"""Double-annotation tooling for the extraction gold set.

Layout under ``gold/``::

    sample.json            fixed-seed stratified sample of pages (the "what to transcribe" list)
    pages/<id>.png         rendered page images (regenerable, git-ignored)
    annotator_a/<id>.txt   annotator A's independent transcription
    annotator_b/<id>.txt   annotator B's independent transcription
    adjudication/<id>.md   diff of A vs B for pages where they disagree
    final/<id>.txt         adjudicated gold (written automatically only when A == B after normalisation)
    silver/<id>.txt        PROVISIONAL single-pass references (not human gold), used only until final/ exists
"""

from __future__ import annotations

import csv
import datetime as dt
import difflib
import json
import random
from collections import defaultdict
from pathlib import Path

from .ingest import load_pages
from .metrics import agreement, bootstrap_ci, mean
from .textutil import normalize_for_eval

STRATA = ("english_digital", "hindi_digital", "scanned")
MIN_PAGE_CHARS = 80


def sample_id(doc_id: str, page: int) -> str:
    return f"{doc_id}__p{page:03d}"


def _page_len(p: dict) -> int:
    return max(
        len("".join((p.get(k) or "").split()))
        for k in ("final_text", "pypdf_text", "layout_text", "ocr_text")
    )


def sample_pages(settings, per_stratum: int = 10, seed: int = 20261016, force: bool = False) -> dict:
    gold = settings.gold_dir
    path = gold / "sample.json"
    if path.exists() and not force:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"status": "exists (not resampled)", "path": str(path), "n": len(data["pages"])}
    pages = load_pages(settings.data_dir)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        if _page_len(p) >= MIN_PAGE_CHARS:  # blank/near-blank pages have no meaningful reference
            by_stratum[p["stratum"]].append(p)
    rng = random.Random(seed)
    chosen = []
    for s in STRATA:
        pool = sorted(by_stratum.get(s, []), key=lambda p: (p["doc_id"], p["page"]))
        picks = rng.sample(pool, min(per_stratum, len(pool)))
        for p in sorted(picks, key=lambda p: (p["doc_id"], p["page"])):
            chosen.append(
                {
                    "sample_id": sample_id(p["doc_id"], p["page"]),
                    "doc_id": p["doc_id"],
                    "page": p["page"],
                    "stratum": s,
                    "language": p["language"],
                }
            )
    data = {
        "seed": seed,
        "per_stratum": per_stratum,
        "eligible_pages": {s: len(by_stratum.get(s, [])) for s in STRATA},
        "min_page_chars": MIN_PAGE_CHARS,
        "created": dt.date.today().isoformat(),
        "pages": chosen,
    }
    gold.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "written", "path": str(path), "n": len(chosen), "eligible": data["eligible_pages"]}


def load_sample(settings) -> list[dict]:
    path = settings.gold_dir / "sample.json"
    if not path.exists():
        raise FileNotFoundError("gold/sample.json missing; run `dastavez gold sample`")
    return json.loads(path.read_text(encoding="utf-8"))["pages"]


def prepare(settings, dpi: int = 150) -> dict:
    """Render page images and create empty transcription files for both annotators."""
    from .ocr import render_page

    gold = settings.gold_dir
    rows = []
    for item in load_sample(settings):
        sid = item["sample_id"]
        img = gold / "pages" / f"{sid}.png"
        if not img.exists():
            img.parent.mkdir(parents=True, exist_ok=True)
            render_page(settings.pdf_dir / f"{item['doc_id']}.pdf", item["page"], dpi).save(img)
        for ann in ("annotator_a", "annotator_b"):
            f = gold / ann / f"{sid}.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                f.write_text("", encoding="utf-8")
        rows.append([sid, item["stratum"], item["doc_id"], item["page"], f"pages/{sid}.png"])
    with open(gold / "checklist.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["sample_id", "stratum", "doc_id", "page", "image", "annotator_a_done", "annotator_b_done"]
        )
        for r in rows:
            w.writerow([*r, "", ""])
    return {"pages": len(rows), "images": str(gold / "pages"), "checklist": str(gold / "checklist.csv")}


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def agreement_report(settings, write: bool = True) -> dict:
    gold = settings.gold_dir
    per_page = []
    for item in load_sample(settings):
        sid = item["sample_id"]
        a, b = _read(gold / "annotator_a" / f"{sid}.txt"), _read(gold / "annotator_b" / f"{sid}.txt")
        if a is None or b is None:
            continue
        per_page.append({"sample_id": sid, "stratum": item["stratum"], **agreement(a, b)})
    summary = {}
    for s in (*STRATA, "all"):
        rows = [r for r in per_page if s == "all" or r["stratum"] == s]
        sims = [r["similarity"] for r in rows]
        summary[s] = {
            "n": len(rows),
            "mean_similarity": mean(sims),
            "mean_cer": mean(r["mean_cer"] for r in rows),
            "similarity_ci95": bootstrap_ci(sims),
        }
    report = {
        "pages_double_annotated": len(per_page),
        "pages_in_sample": len(load_sample(settings)),
        "summary": summary,
        "low_agreement": sorted(
            (r for r in per_page if r["similarity"] < 0.95), key=lambda r: r["similarity"]
        ),
        "per_page": per_page,
    }
    if write:
        settings.results_dir.mkdir(parents=True, exist_ok=True)
        (settings.results_dir / "annotation_agreement.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {k: v for k, v in report.items() if k != "per_page"}


def word_diff(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(" ".join(aw[i1:i2]))
        else:
            out.append(f"[-A: {' '.join(aw[i1:i2])} -][+B: {' '.join(bw[j1:j2])} +]")
    return " ".join(out)


def adjudicate(settings, threshold: float = 0.98) -> dict:
    """Auto-accept pages where A == B after normalisation; write diffs for everything else.

    Pages with similarity >= threshold are listed as "minor" so the adjudicator can clear them
    quickly, but they are still NOT accepted automatically: a human writes final/<id>.txt.
    """
    gold = settings.gold_dir
    status = {
        "accepted_identical": [],
        "needs_adjudication_minor": [],
        "needs_adjudication": [],
        "final_exists": [],
        "missing_annotations": [],
    }
    for item in load_sample(settings):
        sid = item["sample_id"]
        final = gold / "final" / f"{sid}.txt"
        if final.exists() and final.read_text(encoding="utf-8").strip():
            status["final_exists"].append(sid)
            continue
        a, b = _read(gold / "annotator_a" / f"{sid}.txt"), _read(gold / "annotator_b" / f"{sid}.txt")
        if a is None or b is None:
            status["missing_annotations"].append(sid)
            continue
        if normalize_for_eval(a) == normalize_for_eval(b):
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_text(a, encoding="utf-8")
            status["accepted_identical"].append(sid)
            continue
        sim = agreement(a, b)["similarity"]
        key = "needs_adjudication_minor" if sim >= threshold else "needs_adjudication"
        status[key].append(sid)
        out = gold / "adjudication" / f"{sid}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"# {sid}  (similarity A vs B = {sim:.4f})\n\n"
            f"Resolve each `[-A: ... -][+B: ... +]` against the page image `pages/{sid}.png`,\n"
            f"then save the corrected full text as `final/{sid}.txt`.\n\n## Word diff\n\n{word_diff(a, b)}\n",
            encoding="utf-8",
        )
    return {k: (len(v) if k != "needs_adjudication" else v) for k, v in status.items()}


def reference_texts(settings, kind: str = "auto") -> tuple[str, dict[str, str]]:
    """Return (reference_kind, {sample_id: text}). 'auto' prefers adjudicated gold over silver."""
    gold = settings.gold_dir
    ids = [i["sample_id"] for i in load_sample(settings)]
    finals = {i: t for i in ids if (t := _read(gold / "final" / f"{i}.txt")) is not None}
    silver = {i: t for i in ids if (t := _read(gold / "silver" / f"{i}.txt")) is not None}
    if kind == "gold" or (kind == "auto" and finals):
        return "gold", finals
    return "silver", silver
