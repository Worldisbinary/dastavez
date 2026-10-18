"""Command-line interface.

dastavez corpus download|verify     fetch PDFs listed in corpus/manifest.json, verify SHA-256
dastavez ingest                     parse + route + OCR every page; build the retrieval index
dastavez index                      (re)build the retrieval index from parsed pages
dastavez gold sample|prepare|agree|adjudicate   double-annotation tooling
dastavez eval-extraction            pypdf vs layout vs cascade against gold (or silver)
dastavez eval-retrieval             lexical vs dense vs hybrid vs hybrid+rerank
dastavez ask "question"             citation-first answer
dastavez serve                      FastAPI service (/ask) + minimal web page
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings


def _print(obj) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) if not isinstance(obj, str) else obj
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.flush()


def _docs(settings):
    from .corpus import load_manifest

    return load_manifest(settings.corpus_dir / "manifest.json")


def cmd_corpus(args) -> int:
    from .corpus import download, import_local, verify

    s = get_settings()
    docs = _docs(s)
    if args.action == "download":
        report = download(docs, s.pdf_dir, force=args.force)
    elif args.action == "import":
        report = import_local(docs, args.src, s.pdf_dir)
    else:
        report = {d.doc_id: ("ok" if verify(d, s.pdf_dir) else "MISSING/MISMATCH") for d in docs}
    _print(report)
    _print(f"total pages in manifest: {sum(d.pages for d in docs)}")
    return 0 if all(str(v).startswith(("ok", "imported")) for v in report.values()) else 1


def cmd_ingest(args) -> int:
    from .ingest import ingest_corpus
    from .ocr import TesseractOCR

    s = get_settings()
    docs = _docs(s)
    if args.doc:
        docs = [d for d in docs if d.doc_id in set(args.doc)]
    ocr = (
        None
        if args.no_ocr
        else TesseractOCR(s.tesseract_cmd, dpi=s.ocr_dpi, cache_dir=s.data_dir / "ocr_cache")
    )
    if ocr is not None and not ocr.available:
        logging.warning("OCR unavailable (%s): routed pages will keep their text layer", ocr.info.error)
    summary = ingest_corpus(docs, s.pdf_dir, s.data_dir, ocr)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.doc:
        (s.results_dir / "ingest_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _print({k: v for k, v in summary.items() if k != "documents"})
    if not args.skip_index:
        return cmd_index(args)
    return 0


def cmd_index(args) -> int:
    from .index import build_index

    s = get_settings()
    info = build_index(s, text_source=getattr(args, "text_source", "cascade"))
    _print(info)
    return 0


def cmd_gold(args) -> int:
    from . import annotation

    s = get_settings()
    if args.action == "sample":
        out = annotation.sample_pages(s, per_stratum=args.per_stratum, seed=args.seed)
    elif args.action == "prepare":
        out = annotation.prepare(s, dpi=args.dpi)
    elif args.action == "agree":
        out = annotation.agreement_report(s)
    elif args.action == "adjudicate":
        out = annotation.adjudicate(s, threshold=args.threshold)
    else:
        raise SystemExit(f"unknown action {args.action}")
    _print(out)
    return 0


def cmd_eval_extraction(args) -> int:
    from .eval_extraction import run_extraction_eval

    s = get_settings()
    out = run_extraction_eval(s, reference=args.reference)
    _print(out["summary_table"])
    return 0


def cmd_eval_retrieval(args) -> int:
    from .eval_retrieval import run_retrieval_eval

    s = get_settings()
    out = run_retrieval_eval(s, k=args.k, modes=args.modes, text_source=args.text_source)
    _print(out["summary_table"])
    return 0


def cmd_ask(args) -> int:
    from .answer import Answerer

    s = get_settings()
    ans = Answerer.from_settings(s)
    result = ans.ask(args.question, mode=args.mode, k=args.k)
    _print(result.to_dict() if args.json else result.render())
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("dastavez.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dastavez", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="download / verify corpus PDFs")
    c.add_argument("action", choices=["download", "verify", "import"])
    c.add_argument("--force", action="store_true")
    c.add_argument("--src", type=Path, help="directory of already-downloaded PDFs (for `import`)")
    c.set_defaults(func=cmd_corpus)

    i = sub.add_parser("ingest", help="parse, route, OCR and index the corpus")
    i.add_argument("--doc", action="append", help="only this doc_id (repeatable)")
    i.add_argument("--no-ocr", action="store_true", help="never run OCR (routes become ocr_unavailable)")
    i.add_argument("--skip-index", action="store_true")
    i.set_defaults(func=cmd_ingest, text_source="cascade")

    x = sub.add_parser("index", help="build retrieval index from parsed pages")
    x.add_argument("--text-source", choices=["cascade", "pypdf"], default="cascade")
    x.set_defaults(func=cmd_index)

    g = sub.add_parser("gold", help="double-annotation tooling")
    g.add_argument("action", choices=["sample", "prepare", "agree", "adjudicate"])
    g.add_argument("--per-stratum", type=int, default=10)
    g.add_argument("--seed", type=int, default=20261016)
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--threshold", type=float, default=0.98, help="auto-accept if annotators agree >= this")
    g.set_defaults(func=cmd_gold)

    e = sub.add_parser("eval-extraction", help="score pypdf / layout / cascade against gold")
    e.add_argument("--reference", choices=["auto", "gold", "silver"], default="auto")
    e.set_defaults(func=cmd_eval_extraction)

    r = sub.add_parser("eval-retrieval", help="retrieval + citation evaluation")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--modes", nargs="+", default=["lexical", "dense", "hybrid", "hybrid_rerank"])
    r.add_argument(
        "--text-source",
        choices=["cascade", "pypdf"],
        default="cascade",
        help="which page text the index was built from (pypdf = ablation)",
    )
    r.set_defaults(func=cmd_eval_retrieval)

    a = sub.add_parser("ask", help="answer a question with page-level citations")
    a.add_argument("question")
    a.add_argument("--mode", default="hybrid_rerank", choices=["lexical", "dense", "hybrid", "hybrid_rerank"])
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    sv = sub.add_parser("serve", help="run the FastAPI service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
