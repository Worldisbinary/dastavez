# Annotation guide: double-transcribing the gold pages

This guide is for the two annotators (Annotator A = the project owner, Annotator B = a friend).
You will each type out the text of the same 30 PDF pages **independently**. The two versions are
compared automatically, disagreements are resolved, and the result becomes the gold set that the
extraction numbers in the README are computed against.

Time needed: about 10-20 minutes per page, so roughly 5-8 hours each. Hindi and scanned pages take
longer than English ones.

---

## 0. One-time setup

```bash
git clone <repo> && cd dastavez
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
dastavez corpus download            # fetches the PDFs and checks their SHA-256
```

The page sample is already fixed in `gold/sample.json` (seed 20261016, 10 pages per stratum).
**Do not re-run `dastavez gold sample`**: everyone must transcribe the same pages.

Render the page images and create the empty files:

```bash
dastavez gold prepare
```

This creates:

| Path | What it is |
|---|---|
| `gold/pages/<id>.png` | the page image you transcribe from |
| `gold/annotator_a/<id>.txt` | Annotator A types here |
| `gold/annotator_b/<id>.txt` | Annotator B types here |
| `gold/checklist.csv` | tick-list of all 30 pages |

`<id>` looks like `pmay_u2_guidelines_2024_hi__p024` (document id, then the 1-based PDF page number).
You can also open the original PDF at that page if the PNG is hard to read; the page number in the
id is the PDF's page index, not the number printed on the page.

## 1. The golden rule: work independently

* Do **not** look at the other annotator's files, and do not discuss specific pages until both of
  you have finished all 30.
* Do **not** copy text out of the PDF (Ctrl+C), and do **not** run OCR or use Google Lens / ChatGPT.
  The whole point is an independent human reference for judging the machines. Type it yourself.
* Do **not** look at `gold/silver/` (an AI-made provisional reference) or at `data/`.

## 2. Tools

* Any plain-text editor that saves **UTF-8** (VS Code, Notepad++, Windows Notepad is fine).
* Hindi typing: Google Input Tools / Windows "Hindi Phonetic" keyboard, or the Microsoft Indic
  Language Input Tool. Make sure the output is **Unicode Devanagari**, not a legacy font like
  Kruti Dev (if you see "dk;Z" instead of "कार्य", the setup is wrong).

## 3. What to transcribe

Transcribe **all text a reader can see on the page**, in reading order:

* running headers and footers, the printed page number, headings, body text, footnotes,
  table contents, margin notes (side headings), signatures' printed names and titles.

Skip:

* logos, the national emblem and the tiny text inside it (e.g. "सत्यमेव जयते" under the emblem),
  stamps, handwriting, watermarks, decorative pictures;
* fill-in blanks (`__________`) and dotted leader lines in tables of contents (keep the words
  and page numbers, drop the dots);
* bullet symbols (•, ●, ➢, -) at the start of list items. Keep list *numbering* such as
  `1.`, `(a)`, `(iv)`, `5.2.1`, `(क)`.

## 4. Reading order and layout

* Top to bottom, left to right.
* **Margin notes / side headings** (common in gazettes): put the margin note as its own line just
  *before* the paragraph it sits next to.
* **Two columns:** finish the left column, then the right column.
* Put a blank line between paragraphs. Inside a paragraph, line breaks do not matter (the scorer
  collapses all whitespace), so you may type a paragraph as one long line.
* A word broken across lines with a hyphen (`imple-` / `mentation`) is typed joined
  (`implementation`). A real hyphen (`PM-KISAN`, `समय-समय`) stays.

## 5. Tables

* One table row per line, cells separated by ` | ` (space, pipe, space).
* Read cells left to right. If a cell has several lines, join them with spaces into one cell.
* Leave out empty cells (do not type `|  |`).
* Example:
  ```
  Type of Unit | Carpet Area applicable for TIG | Permissible TIG amount
  Single Bedroom | Maximum 30 sqm | GoI: ₹3,000/sqm State Share: ₹2,000/sqm
  ```

## 6. Characters and spelling

* **Type exactly what is printed, including typos** ("enemployment", "पचांयती"). Do not correct
  the government's spelling. If a word is genuinely unreadable, type `[?]` in its place.
* **Numbers:** type digits as printed. If the page uses Devanagari digits (१२३) you may type
  either ASCII (123) or Devanagari digits; the scorer maps both to ASCII. Keep separators as
  printed: `1,50,000`, `₹3 लाख`, `Rs.6000/-`.
* **Rupee sign:** `₹` if printed as ₹; `Rs.` / `रू0` / `रु.` if printed that way.
* **Superscripts** (`25th`, `1st`): type inline as `25th`.
* **Quotes and dashes:** any quote style is fine (the scorer unifies " " ' ' and – — -).
* **Case:** type as printed (the scorer ignores case, but keep it natural).

### Hindi specifics (matras and conjuncts)

The scorer compares Unicode characters, so the *logical* order matters, not how it looks.

* Type the consonant first, then the matra, even for `ि`: कि = क + ि (your keyboard does this).
* Conjuncts use halant (्): `प्र` = प + ् + र, `क्ष` = क + ् + ष, `त्र` = त + ् + र,
  `ज्ञ` = ज + ् + ञ, `श्र` = श + ् + र.
* Reph: `र्य` in कार्य = र + ् + य; it is drawn on top but typed first.
* Nukta forms (`ड़`, `ढ़`, `फ़`, `ज़`): type them only if the dot is printed. Many government
  documents omit it; follow the page.
* Chandrabindu (ँ) vs anusvara (ं): follow the page; if you cannot tell, use what the page uses
  elsewhere for the same word.
* `ॅं` combinations (e.g. `पुस्तिकाएॅं`) exist in some legacy-font documents; type what you see.
* Danda `।` ends sentences; type the danda, not a full stop or a pipe `|`.
* Abbreviation zero `0` after a letter (`का0 आ0`, `रू0`) is a typed zero, keep it.

## 7. Saving

* Save each file as UTF-8 (no special encoding), same name as created by `gold prepare`.
* An empty file means "not done yet". When you finish a page, tick it in `gold/checklist.csv`.

## 8. After both annotators finish: agreement

```bash
dastavez gold agree
```

This writes `results/annotation_agreement.json` and prints, per stratum and overall:

* **mean similarity** between A and B (1 = identical after normalisation),
* **mean CER** (character error rate) of A measured against B and vice versa,
* a list of pages with similarity below 0.95.

What the numbers mean: CER = (insertions + deletions + substitutions) / length of the reference.
Two careful annotators typically reach CER below 1-2% on clean English pages; Hindi and degraded
scans are usually worse. Low agreement on a page is useful information: the page is genuinely
hard, and machine scores on it deserve less weight.

## 9. Adjudication

```bash
dastavez gold adjudicate
```

* Pages where A and B are **identical after normalisation** are copied to `gold/final/` automatically.
* Every other page gets `gold/adjudication/<id>.md` with a word diff like
  `[-A: 10000 -][+B: 1000 +]`. Sit together, look at the page image for each difference, decide
  what is actually printed, and save the corrected full text as `gold/final/<id>.txt`
  (start by copying A's file, then fix it).
* Re-run `dastavez gold adjudicate` until it reports `needs_adjudication: []` and
  `missing_annotations: 0`.

Do not "split the difference" when unsure: re-check the image, and use `[?]` only if neither of
you can read it.

## 10. Scoring the extractors against your gold

```bash
dastavez eval-extraction            # uses gold/final/ automatically once it exists
```

Results land in `results/extraction_eval_gold.{json,md}`. Copy the table into the README
(replacing the provisional silver table) and update the resume bullet with these numbers.

Commit `gold/annotator_a/`, `gold/annotator_b/`, `gold/final/`, `gold/adjudication/` and the
results files, so the evaluation is reproducible by anyone.
