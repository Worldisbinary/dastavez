# AI-transcribed reference set (30 pages, not human-verified)

These 30 files cover every page in `gold/sample.json` (10 per stratum: english_digital,
hindi_digital, scanned). They were **transcribed by Claude (Anthropic)**, an AI assistant, from the
rendered page images in `gold/pages/` (`dastavez gold prepare`), so the extraction evaluation has a
reference while no human annotation exists. They are **not** human gold and **not** double-annotated.

* Independence: the transcriber worked only from the page images (and zoomed crops of them). It did
  not look at pypdf, pdfplumber, Tesseract or any other extractor output, `data/` caches or
  `results/` before or while transcribing.
* Process: one transcription pass following `docs/ANNOTATION_GUIDE.md`, then one self-review pass
  against the image. No second annotator, no adjudication, no human verification.
* History: 12 pages were done in an earlier session, the remaining 18 later the same way.
* Judgment calls: handwriting (dates, file numbers, signatures) is skipped per the guide; flowchart
  labels are read in flow order; in the PMAY-U 2.0 Hindi PDF the colon-like glyph is typed `ः`
  after a Devanagari letter and `:` after digits or punctuation.
* Known risks: possible misreadings of small or degraded Devanagari, and an AI transcriber may share
  error patterns with OCR (circularity), which would flatter OCR-based systems.

`dastavez eval-extraction` uses these files only when `gold/final/` is empty. The annotation tooling
supports human single or double annotation (`gold/annotator_a/`, `gold/annotator_b/`, adjudication
into `gold/final/`); once `gold/final/` is populated it replaces this reference set automatically.
