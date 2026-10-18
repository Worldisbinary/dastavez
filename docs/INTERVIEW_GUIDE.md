# Dastavez: interview guide

Everything here matches the code in this repo and the numbers in `results/`. If a number in an
interview answer is not in `results/`, do not say it.

---

## 1. Two-minute pitch

> Indian government scheme documents are where citizens and field officers look up eligibility,
> amounts and procedures, but they are messy PDFs: some English, many Hindi, some scanned
> gazettes. I built **Dastavez**, a question-answering system over 16 real scheme documents
> (378 pages: PM-KISAN, PMAY-Urban 2.0 in English and Hindi, PM SVANidhi, Stand-Up India,
> AB PM-JAY, MGNREGA/Haryana rules, PM Ujjwala) where **every answer sentence carries a
> page-level citation and is checked against that page before it is shown**.
>
> The interesting part turned out to be extraction, not the LLM. Plain `pypdf` gives unusable
> text for most of the Hindi and scanned pages: the Hindi PDFs use legacy 8-bit fonts, so the
> "text" is Kruti-Dev-style mojibake like `vkokl ;kstuk`, and scans have no text at all. 171 of
> the 378 pages get routed to OCR, and OCR output is accepted on 157 of them. So I built a **routed OCR cascade**: every
> page gets layout-aware extraction first, cheap heuristics (empty layer, undecodable glyphs,
> impossible Devanagari sequences, wrong script for the document language, non-word ratio,
> full-page images) decide whether it goes to Tesseract `hin+eng`, and OCR output only replaces
> the text layer if it beats it on a reference-free quality score. On a provisional reference set
> the mean text similarity goes from about 0.50 to 0.93 overall, and from about 0.22 to 0.93 on
> Hindi pages. The double-annotated human gold set is built with tooling in the repo.
>
> Retrieval is hybrid: BM25 and multilingual-e5 dense vectors fused with reciprocal rank fusion,
> then a multilingual cross-encoder reranker; storage is Postgres + pgvector HNSW + Postgres
> full-text search, with an in-process fallback for dev. I evaluate lexical vs dense vs hybrid vs
> hybrid+rerank on 58 questions (10 deliberately unanswerable) with recall@k, MRR, citation
> precision and not-found accuracy, with the abstention threshold calibrated on a held-out dev split.

(Re-check the numbers against `results/` before you use them; they change if the corpus,
models or gold set change.)

## 2. End-to-end walkthrough

1. **Corpus** (`corpus/manifest.json`, `dastavez corpus download`): 16 PDFs from government sites,
   each pinned by SHA-256. PDFs are not committed; downloads go to `corpus/pdfs/` (treated as
   untrusted input, size-capped, hash-verified, mismatches quarantined as `.mismatch`).
2. **Parse** (`parse.py`): baseline `pypdf.extract_text()` per page, plus `pdfplumber`
   layout-aware parsing: de-duplicated "fake bold" glyphs, line grouping, two-column gutter
   detection, headings from font size / bold relative to the page's median body size, tables via
   `find_tables()` rendered as `cell | cell` rows in place, printed page labels from the margins,
   and raster-image coverage of the page.
3. **Route** (`routing.py`): rules in a fixed order: empty -> undecodable glyphs (U+FFFD / private
   use) -> low printable ratio -> broken Devanagari -> legacy-font script mismatch -> garbled
   latin -> scan with an existing OCR layer -> keep.
4. **OCR** (`ocr.py`): pypdfium2 renders the page at 300 dpi greyscale, Tesseract 5 LSTM
   (`--oem 1 --psm 3`) with `eng` or `hin+eng`; cached by (pdf hash, page, lang, dpi, version).
5. **Arbitrate**: OCR replaces the text layer only if its reference-free quality score is higher
   by a margin; otherwise the page is recorded as `ocr_rejected`. Every page records its route and reason.
6. **Chunk** (`chunking.py`): heading-aware, never crosses a page, ~900 chars max, sentence-level
   splitting, tables kept whole or split by rows; every chunk keeps `doc_id`, `page`, `page_label`, `heading`.
7. **Index** (`index.py`, `store.py`): e5 embeddings (with `passage:` / `query:` prefixes), cached
   per text hash; in-process BM25 + numpy, or Postgres (pgvector HNSW `vector_cosine_ops`, GIN on a
   `tsvector`) when `DASTAVEZ_DATABASE_URL` is set.
8. **Retrieve** (`retrieve.py`): top-30 lexical + top-30 dense -> RRF (k=60) -> top-20 reranked by
   `bge-reranker-v2-m3` (int8 ONNX) -> top-k.
9. **Answer** (`answer.py`): gate on evidence strength (calibrated threshold); if it fails, return
   "Not found in the documents." Otherwise Gemini (if a key is set) is asked for <= 4 sentences each
   ending in `[doc_id p.N]`; without a key, an extractive fallback quotes the best sentences.
10. **Verify** (`verify.py`): each sentence's cited page must contain every number in the
    sentence, every quoted span, and >= 60% of its content tokens. Failing sentences are dropped;
    if none survive, the answer is "not found".
11. **Evaluate**: `eval-extraction` (pypdf vs layout vs cascade, per stratum, similarity + CER
    with bootstrap CIs) and `eval-retrieval` (four retrieval modes, citation and abstention metrics).
12. **Serve**: `dastavez ask`, and FastAPI `POST /ask` + a minimal page that expands each citation
    into its page snippet and a link to the source PDF page.

## 3. Design decisions, trade-offs and alternatives

### Why citation-first
Government-scheme answers are acted on (eligibility, money). A fluent but wrong answer is worse
than "not found". Citations at page level let a user check the source in seconds, and they make
the system *testable*: citation precision is measurable, free-form fluency is not.
Trade-off: lower recall of answers (more abstentions), and extractive answers read less naturally.
Alternative: post-hoc attribution (generate, then search for support); that tends to retrofit
citations onto unsupported claims, which is exactly what we want to avoid.

### Why hybrid retrieval
Scheme questions mix exact tokens (`PMAY-U`, `₹25 lakh`, `5.2.6`, `D1`, scheme acronyms) where
BM25 is strong, with paraphrase ("who can't get PM-KISAN" vs "exclusions") and cross-lingual
queries where dense is strong. Hybrid gets both failure modes covered. Cost: two indexes, a fusion step.

### BM25 in one breath
score(q,d) = sum over query terms t of IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl)).
`k1` (1.5) saturates term frequency (the 10th "scheme" adds little); `b` (0.75) normalises by
document length so long chunks are not favoured. IDF here is Lucene's
ln((N - df + 0.5)/(df + 0.5) + 1), which never goes negative. Implemented from scratch in
`lexical.py` and unit-tested against hand-computed values.

### Reciprocal rank fusion
RRF(d) = sum_i 1/(k + rank_i(d)), k = 60. It uses ranks, not scores, so you do not need to
calibrate BM25 scores (unbounded) against cosine similarities (bounded); k damps the dominance of
the very top ranks. Alternative: weighted score fusion (needs normalisation and tuning) or a
learned fusion (needs training data we do not have).

### Cross-encoder vs bi-encoder
A bi-encoder embeds query and passage separately (fast, indexable, used for recall over the whole
corpus). A cross-encoder reads (query, passage) jointly with full attention between them (much
more precise, but O(candidates) model calls per query, so only for re-ordering ~20 candidates).
Here: e5-large bi-encoder for recall, bge-reranker-v2-m3 for precision; both multilingual.
The reranker's sigmoid score also doubles as the "is anything relevant?" signal for abstention.

### HNSW vs IVFFlat (pgvector)
HNSW: graph index, best recall/latency trade-off, no training step, can be built on an empty
table and stays good under inserts; costs more memory and slower builds. IVFFlat: k-means lists,
must be built after data exists (centroids), recall depends on `lists`/`probes` and degrades as
data drifts. For a small, growing corpus HNSW is the safer default (m=16, ef_construction=64,
ef_search=80). At 1,156 chunks exact search would also be fine; the index is there for scale.

### Chunking strategy
Page boundaries are hard boundaries because the citation unit is the page. Within a page, headings
start new chunks (a heading usually starts a new rule), long paragraphs are split on sentence
boundaries (including the Devanagari danda), tables stay whole when they fit. The nearest heading
is carried as context and prepended to the text that is embedded. Trade-off: a fact that spans a
page break is split; the carried heading mitigates but does not fix that.

### Why OCR only on routed pages
OCR is slow (~3 s/page here), and on clean digital English pages it is *worse* than the text layer
(it introduces character errors where the PDF had exact Unicode). Routing keeps the exact text
where it exists and spends OCR only where the text layer is unusable. The arbitration step adds a
safety net: OCR output that scores lower than the text layer is rejected (14 pages).

### Devanagari font-encoding problems
Many Indian government PDFs were typeset with legacy 8-bit fonts (Kruti Dev, Chanakya, DevLys).
The glyphs look like Hindi, but the bytes are Latin characters mapped to Hindi shapes, so text
extraction returns `vkoklu vkSj 'kgjh dk;Z ea=ky;` for "आवासन और शहरी कार्य मंत्रालय". Unicode
fonts with a broken ToUnicode CMap give a different failure: conjuncts and reordered matras
disappear (`धानमंी` for `प्रधानमंत्री`), or decode to U+FFFD. Detection: (a) expected Hindi but
almost no Devanagari and a low word-likeness score, (b) orthographically impossible sequences
(a matra after an anusvara, a matra at word start, two matras in a row), (c) replacement /
private-use characters. Conversion tables for Kruti Dev exist, but each legacy font needs its own
map and they are lossy; OCR handles all of them uniformly.

### CER vs similarity
CER = Levenshtein(h, r)/|r|: interpretable ("1.9% of characters wrong"), but unbounded above when
the hypothesis is long garbage and undefined for empty references. Normalised similarity
1 - Indel/(|h|+|r|) is bounded in [0, 1] and symmetric, which makes means across pages stable.
I report both. Both are computed after the same normalisation (NFC, casefold, whitespace
collapse, quote/dash unification, Devanagari digits -> ASCII) so formatting does not count as an error.

### Inter-annotator agreement
Two people transcribe the same pages independently; their similarity / CER against each other
is the ceiling on what any metric can resolve. If annotators disagree on 3% of characters, a
system difference of 1% is noise. Pages where they disagree are adjudicated against the image
(identical pages are accepted automatically). This is the text analogue of Cohen's kappa for labels.

### Routing thresholds
Set once by looking at per-page text-layer statistics across the corpus before the gold sample
was drawn; never tuned on a reference transcription. They are deliberately coarse (e.g. word-like
ratio < 0.6), because a mis-route is cheap: arbitration catches OCR that is worse.

### Abstention threshold
Calibrated on a fixed 20% dev split of the questions (by hash of the question id), reported on
the other 80%. With 10 unanswerable questions (only 1 in dev) this is a small sample; say so.
The first run picked the first optimal cut, which sat exactly on the lowest dev positive (0.99)
and refused 29% of answerable questions; it was changed to the max-margin midpoint (0.572) and
both runs are kept in `results/` (`archive/` holds run 1). Be upfront that this change was made
after seeing run 1, and that it fixes a degenerate tie-break rather than tuning to the test set.

## 4. Likely questions and crisp answers

1. **What does the system do in one line?** Answers questions about Indian scheme documents with
   page citations that are checked against the page, and says "not found" otherwise.
2. **How big is the corpus?** 16 documents, 378 pages: 219 English born-digital, 139 Hindi
   born-digital, 20 scanned (bilingual gazettes and letters). Real public PDFs, pinned by SHA-256.
3. **Why not just use pypdf?** On this corpus pypdf returns usable text for English digital pages
   but mojibake or nothing for legacy-font Hindi and scans; the router sent 171 pages to OCR and
   accepted the OCR text on 157.
4. **How do you detect a bad text layer without ground truth?** Empty, U+FFFD/private-use glyphs,
   printable ratio, impossible Devanagari sequences, script mismatch with the document language,
   share of word-like tokens, and image coverage. All are cheap string statistics.
5. **What OCR do you use and why?** Tesseract 5 LSTM with `hin+eng`. Open source, runs offline,
   has Hindi models, installs via apt in CI. Cloud OCR (Google Vision) would likely be more
   accurate on degraded scans but adds cost, a network dependency and data-sharing concerns.
6. **What happens if OCR is worse than the text layer?** The page keeps its text layer and is
   recorded as `ocr_rejected`; 14 pages took that path (mostly empty/graphic pages).
7. **How did you measure extraction quality?** Mean normalised similarity and CER against
   reference transcriptions of a fixed-seed stratified sample (10 pages per stratum). Human
   double-annotated gold is in progress; current numbers are against a provisional single-pass reference.
8. **Why stratify the sample?** A random 30 pages from this corpus would be mostly English digital
   pages, where everything works; stratifying guarantees the hard strata are measured.
9. **What is the provisional result?** Overall similarity 0.500 (pypdf) -> 0.928 (cascade); Hindi
   0.216 -> 0.932; scanned 0.298 -> 0.860; English unchanged at ~0.99. n = 12 pages, so wide CIs.
10. **Why is "layout" barely better than pypdf on similarity?** Layout parsing changes structure
    (reading order, tables, headings), which the whitespace-collapsing similarity mostly ignores;
    its benefit shows up in chunking and citations, not in character accuracy.
11. **Worst page?** A skewed, two-column scanned Haryana gazette page (similarity 0.67): Tesseract's
    page segmentation dropped lines next to margin notes. Fix: deskew + `--psm 4/6` per column.
12. **Why BM25 and not TF-IDF cosine?** BM25 saturates term frequency and normalises length; it is
    the standard lexical baseline that consistently beats raw TF-IDF for passage retrieval.
13. **Why RRF and not a weighted sum?** Scores are on different scales; RRF needs no calibration and
    is robust. One parameter (k=60, the value from the original paper).
14. **Which embedding model and why?** `intfloat/multilingual-e5-large` (1024-d) via fastembed/ONNX:
    strong multilingual retrieval including Hindi, runs locally without PyTorch. Fallbacks: Gemini
    embeddings with a key, a hashing embedder offline.
15. **Which reranker?** `BAAI/bge-reranker-v2-m3` int8 ONNX: multilingual cross-encoder, ~0.6 GB.
    A larger Jina multilingual reranker failed to load in memory on this laptop.
16. **How do you store vectors?** Postgres + pgvector, HNSW with cosine ops; Postgres full-text
    search for the lexical side. Locally an in-process store with the same interface.
17. **Why hash tokens into the Postgres tsvector?** Postgres' text-search parser can split Devanagari
    combining marks depending on server locale. Tokenising in Python and storing ASCII hashes makes
    lexical search identical across environments.
18. **Is Postgres FTS the same as your BM25?** No. `ts_rank_cd` is cover-density ranking, not BM25.
    Evaluation numbers come from the in-process BM25; the Postgres path is integration-tested in CI.
19. **How does the verifier work?** Lexical entailment proxy: numbers and quotes must appear on the
    cited page, and >= 60% of content tokens. It is not NLI; it catches wrong numbers and wrong pages,
    not subtle negation.
20. **Can the verifier be fooled?** Yes: "X is not eligible" vs "X is eligible" share all tokens. An
    NLI model (or the reranker scoring sentence vs page) would be the next step.
21. **What if the LLM cites a page that was not retrieved?** The verifier checks against the actual
    page text, retrieved or not; a page id that does not exist fails as `page_not_found`.
22. **How do you decide "not found"?** The top reranker probability must exceed a threshold
    calibrated on the dev split; and if every generated sentence fails verification, the answer is
    "not found" too.
23. **What was the effect of reranking?** On 48 answerable questions: recall@5 0.745 (hybrid) ->
    0.848 (hybrid + rerank), MRR@10 0.871 -> 0.953, citation precision 0.524 -> 0.667. Honest
    caveat: plain RRF hybrid (R@5 0.745) was *worse* than dense-only (0.791); BM25 adds noise here
    and only the reranker turns the bigger candidate pool into a win. Cost: 11.9 s/query on CPU.
    Against the raw-pypdf index the cascade improves hybrid+rerank R@5 from 0.785 to 0.848, and
    BM25 recall@5 on Hindi questions from 0.300 to 0.521.
24. **How were questions written?** By me (drafted with an AI assistant) from the documents, each
    with gold pages found by searching the page text; marked for owner review. Ten are deliberately
    unanswerable (schemes not in the corpus, or facts true in the world but absent from the documents).
25. **Isn't writing questions from the documents biased toward lexical retrieval?** Yes: questions
    reuse document wording. That inflates BM25. Paraphrased and cross-lingual questions are the fix.
26. **Recall@k on what unit?** Page level: a retrieved chunk counts if its page is a gold page.
    Recall@k = fraction of gold pages among the pages of the top-k chunks.
27. **Citation precision?** Fraction of cited pages in an answer that are gold pages, averaged over
    answered, answerable questions.
28. **Why is the reported model latency so high?** CPU-only ONNX on a laptop; the cross-encoder scores
    20 pairs per query. A GPU, a smaller reranker, or reranking fewer candidates fixes it.
29. **How would you scale this to 10,000 documents?** Batch OCR workers with the same routing, a
    queue, pgvector HNSW (already there), FTS in Postgres, embedding cache by content hash (already
    there), and evaluation per document family.
30. **How do you keep it reproducible?** Pinned PDF hashes, fixed seeds for sampling and bootstrap,
    OCR cache keyed by engine version, results JSON committed, CI with Tesseract and pgvector.
31. **What is in CI?** ruff lint + format check; pytest on Python 3.11-3.13 with Tesseract eng+hin
    and a pgvector service container, so OCR and Postgres tests run there.
32. **Security concerns?** PDFs are untrusted: never executed, hash-verified, size-capped,
    parsed by pure-Python/pdfium; no secrets in the repo; the API escapes all HTML it renders.
33. **Why Hindi questions?** The Hindi documents are only searchable after OCR, so Hindi recall is
    the clearest downstream measure of the cascade (compare `retrieval_eval.md` with
    `retrieval_eval_pypdf.md`).
34. **What would you do with one more week?** Human gold annotation (in progress), deskew + column
    detection before OCR, NLI-based verification, paraphrased/cross-lingual questions, and a
    Gemini-generated answer evaluation with the same verifier.

## 5. Honest weak spots (and how to answer them)

* **The headline extraction numbers are provisional.** They use a 12-page single-pass reference
  transcribed by an AI assistant, not the double-annotated human gold. Say exactly that, and that
  the tooling for the human gold is done and the resume number will be updated from it.
* **Small evaluation sets.** 12 reference pages; 58 questions with 10 unanswerable. Confidence
  intervals are wide; differences of a few points between retrieval modes are not significant.
* **Questions were drafted by the builder (with AI help) from the documents.** This favours
  lexical matching. Owner review and paraphrased questions are pending.
* **No LLM in the measured numbers.** Without a Gemini key the evaluated answers are extractive;
  the generation path is unit-tested with a fake generator but not benchmarked.
* **Verifier is lexical.** It will pass a sentence with flipped negation.
* **OCR is Tesseract fast models on CPU.** Degraded, skewed scans still fail (worst page 0.67).
* **Corpus is 16 documents.** Enough to exercise the hard cases, not to claim coverage of
  Indian schemes in general.
* **Postgres numbers are not separately benchmarked.** The pgvector path is exercised by CI
  integration tests; the reported retrieval numbers come from the in-process index.
