"""Retrieval and answer evaluation over eval/questions.jsonl.

Compares lexical-only, dense-only, hybrid (RRF) and hybrid + rerank on the same index.
The "not found" gate threshold is calibrated on a fixed 20% dev split (by question-id hash)
and reported on the remaining 80% test split, so the headline number is not tuned on itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

from .answer import Answerer
from .metrics import citation_precision, hit_at_k, mean, recall_at_k, reciprocal_rank

KS = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]
    for q in qs:
        q["gold_pages"] = [tuple(p) for p in q.get("gold_pages", [])]
    return qs


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 5 == 0 else "test"


def calibrate(values: list[tuple[float, bool]]) -> float:
    """Pick the gate threshold maximising balanced accuracy on (gate_value, is_answerable) pairs.

    Candidate cuts are the midpoints between consecutive observed values (plus "answer
    everything"). Among cuts with equal balanced accuracy the one with the widest gap to its
    neighbours wins (max-margin), so the threshold does not sit on the edge of either class.
    """
    if not values:
        return 0.0
    pos = [v for v, a in values if a]
    neg = [v for v, a in values if not a]
    vals = sorted({v for v, _ in values})
    cuts = [(vals[0], 0.0)] + [((a + b) / 2, b - a) for a, b in pairwise(vals)]
    best_key, best_t = (-1.0, -1.0), 0.0
    for t, margin in cuts:
        tpr = (sum(v >= t for v in pos) / len(pos)) if pos else 1.0
        tnr = (sum(v < t for v in neg) / len(neg)) if neg else 1.0
        key = ((tpr + tnr) / 2, margin)
        if key > best_key:
            best_key, best_t = key, t
    return float(best_t)


def run_retrieval_eval(
    settings,
    k: int = 5,
    modes=("lexical", "dense", "hybrid", "hybrid_rerank"),
    text_source: str = "cascade",
    answerer: Answerer | None = None,
    questions_path: Path | None = None,
    write: bool = True,
) -> dict:
    questions = load_questions(questions_path or settings.home / "eval" / "questions.jsonl")
    ans = answerer or Answerer.from_settings(settings, text_source)
    depth = max(max(KS), k)

    # pass 1: retrieval + gate values
    runs: dict[str, list[dict]] = {}
    timings = {}
    for mode in modes:
        t0 = time.time()
        rows = []
        for q in questions:
            hits = ans.retriever.search(q["question"], mode=mode, k=depth)
            gate = ans.gate(q["question"], hits[:k], mode)
            rows.append({"q": q, "hits": hits, "gate": gate})
        runs[mode] = rows
        timings[mode] = round((time.time() - t0) / max(1, len(questions)), 3)

    # calibrate gates on the dev split
    gates = {}
    for mode, rows in runs.items():
        kind = rows[0]["gate"]["kind"] if rows else "lexical"
        dev = [
            (r["gate"]["value"], bool(r["q"]["answerable"])) for r in rows if split_of(r["q"]["id"]) == "dev"
        ]
        gates.setdefault(kind, {})[mode] = calibrate(dev)
    # one lexical gate shared by non-rerank modes: take the hybrid mode's calibration when available
    final_gates = {}
    if "cross_encoder" in gates:
        final_gates["cross_encoder"] = gates["cross_encoder"].get("hybrid_rerank")
    if "lexical" in gates:
        lg = gates["lexical"]
        final_gates["lexical"] = lg.get("hybrid", next(iter(lg.values())))
    ans.gates.update({kk: v for kk, v in final_gates.items() if v is not None})

    # pass 2: answers + metrics
    per_mode = {}
    per_question = []
    for mode, rows in runs.items():
        rec = {kk: [] for kk in KS}
        mrr, hit5, cprec, answered, false_nf, support, nf_test, nf_all = [], [], [], [], [], [], [], []
        by_lang: dict[str, list[float]] = {}
        for r in rows:
            q, hits = r["q"], r["hits"]
            pages = [(h.chunk.doc_id, h.chunk.page) for h in hits]
            gold = set(q["gold_pages"])
            result = ans.ask(q["question"], mode=mode, k=k, hits=hits)
            n_checked = len(result.sentences) + len(result.dropped)
            if q["answerable"]:
                for kk in KS:
                    rec[kk].append(recall_at_k(pages, gold, kk))
                mrr.append(reciprocal_rank(pages, gold))
                hit5.append(hit_at_k(pages, gold, k))
                by_lang.setdefault(q.get("language", "en"), []).append(recall_at_k(pages, gold, k))
                answered.append(0.0 if result.not_found else 1.0)
                false_nf.append(1.0 if result.not_found else 0.0)
                if not result.not_found:
                    cprec.append(citation_precision(result.citations, gold))
                if n_checked:
                    support.append(len(result.sentences) / n_checked)
            else:
                nf_all.append(1.0 if result.not_found else 0.0)
                if split_of(q["id"]) == "test":
                    nf_test.append(1.0 if result.not_found else 0.0)
            per_question.append(
                {
                    "mode": mode,
                    "id": q["id"],
                    "split": split_of(q["id"]),
                    "answerable": q["answerable"],
                    "gold_pages": [list(p) for p in gold],
                    "retrieved_pages": [list(p) for p in pages[:k]],
                    "rr": reciprocal_rank(pages, gold) if gold else None,
                    "gate": r["gate"],
                    "not_found": result.not_found,
                    "citations": [list(c) for c in result.citations],
                    "answer": result.answer,
                }
            )
        per_mode[mode] = {
            **{f"recall@{kk}": mean(rec[kk]) for kk in KS},
            f"hit@{k}": mean(hit5),
            "mrr@10": mean(mrr),
            "citation_precision": mean(cprec),
            "answered_rate": mean(answered),
            "false_not_found_rate": mean(false_nf),
            "verifier_support_rate": mean(support),
            "not_found_accuracy_test": mean(nf_test),
            "not_found_accuracy_all": mean(nf_all),
            f"recall@{k}_by_language": {lang: mean(v) for lang, v in sorted(by_lang.items())},
            "seconds_per_query": timings[mode],
        }

    n_ans = sum(1 for q in questions if q["answerable"])
    n_un = len(questions) - n_ans

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    lines = [
        f"Questions: {len(questions)} ({n_ans} answerable, {n_un} unanswerable); text source: {text_source}; "
        f"embedder: {getattr(ans.retriever.embedder, 'name', '?')}; reranker: {getattr(ans.retriever.reranker, 'name', None)}; "
        f"generator: {ans.generator.name if ans.generator else 'extractive'}",
        "",
        f"| Mode | R@1 | R@3 | R@{k} | R@10 | MRR@10 | Cit. precision | Answered | Not-found acc (test) | Not-found acc (all) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, m in per_mode.items():
        lines.append(
            f"| {mode} | {f(m['recall@1'])} | {f(m['recall@3'])} | {f(m[f'recall@{k}'])} | {f(m['recall@10'])} | "
            f"{f(m['mrr@10'])} | {f(m['citation_precision'])} | {f(m['answered_rate'])} | "
            f"{f(m['not_found_accuracy_test'])} | {f(m['not_found_accuracy_all'])} |"
        )
    table = "\n".join(lines)
    out = {
        "text_source": text_source,
        "k": k,
        "n_questions": len(questions),
        "n_answerable": n_ans,
        "n_unanswerable": n_un,
        "gates_calibrated_on_dev": final_gates,
        "per_mode": per_mode,
        "per_question": per_question,
        "summary_table": table,
    }
    if write:
        settings.results_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if text_source == "cascade" else f"_{text_source}"
        (settings.results_dir / f"retrieval_eval{suffix}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (settings.results_dir / f"retrieval_eval{suffix}.md").write_text(table + "\n", encoding="utf-8")
        if text_source == "cascade":
            (settings.results_dir / "calibration.json").write_text(
                json.dumps({"gates": final_gates, "split": "dev = sha1(id) % 5 == 0"}, indent=2) + "\n",
                encoding="utf-8",
            )
    return out
