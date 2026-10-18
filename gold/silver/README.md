# Silver references (PROVISIONAL, not gold)

These 12 files are **single-pass transcriptions produced by an AI assistant (Claude)** by reading
200-dpi renders of the page images, so the extraction-evaluation pipeline could be exercised before
human annotation exists. They are *not* the double-annotated gold set and every number computed
from them is labelled `silver` / provisional.

* Selection: 4 pages per stratum, chosen from `gold/sample.json` with `random.Random(1).sample(ids, 4)`.
* The transcriber did not look at any system output (pypdf, pdfplumber, Tesseract) for these pages
  before or while transcribing; it followed the rules in `docs/ANNOTATION_GUIDE.md`.
* Known risks: one annotator, no adjudication, possible misreadings of small/degraded Devanagari,
  and an AI transcriber may share failure modes with OCR. Treat them as a smoke test.

Once `gold/final/` is populated by the double-annotation workflow, `dastavez eval-extraction`
automatically uses the human gold instead.
