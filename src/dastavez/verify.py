"""Citation verifier: does the cited page actually support the sentence?

A sentence is *supported* by a page when all of these hold:
1. every quoted span ("...") of 3+ words appears verbatim (after normalisation) on the page;
2. every number in the sentence appears on the page (numbers are where hallucinations hurt);
3. at least ``min_recall`` of the sentence's content tokens occur on the page
   (verbatim containment short-circuits to score 1.0).
Sentences without a citation, or whose citations all fail, are dropped (strict mode) or
flagged (lenient mode). This is a lexical entailment proxy, not NLI; see the limitations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .textutil import normalize_for_eval, tokenize

CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\s*(?:,\s*)?p(?:age)?\.?\s*(\d+)\]")
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{8,})[\"”]")
_NUM_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[Citation]
    supported: bool
    score: float
    reason: str
    supporting: Citation | None = None
    span: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class VerificationResult:
    checks: list[SentenceCheck] = field(default_factory=list)

    @property
    def supported(self) -> list[SentenceCheck]:
        return [c for c in self.checks if c.supported]

    @property
    def dropped(self) -> list[SentenceCheck]:
        return [c for c in self.checks if not c.supported]

    @property
    def support_rate(self) -> float:
        return (len(self.supported) / len(self.checks)) if self.checks else 0.0


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", sentence)).strip()


def parse_citations(sentence: str) -> list[Citation]:
    return [Citation(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(sentence)]


def _best_span(claim: str, page_text: str, width: int = 280) -> str:
    """Return a window of the page around the densest run of claim tokens (for display)."""
    toks = set(tokenize(claim))
    words = page_text.split()
    if not words:
        return ""
    hits = [1 if set(tokenize(w)) & toks else 0 for w in words]
    win = 40
    best, best_i = -1, 0
    run = sum(hits[:win])
    for i in range(max(1, len(words) - win + 1)):
        if i:
            run += (hits[i + win - 1] if i + win - 1 < len(hits) else 0) - hits[i - 1]
        if run > best:
            best, best_i = run, i
    span = " ".join(words[best_i : best_i + win])
    return span[: width * 2]


def support(claim: str, page_text: str, min_recall: float = 0.6) -> tuple[bool, float, str]:
    claim_n = normalize_for_eval(claim)
    page_n = normalize_for_eval(page_text)
    if not claim_n or not page_n:
        return False, 0.0, "empty"
    if claim_n.rstrip(" .।") in page_n:
        return True, 1.0, "verbatim"
    for q in _QUOTE_RE.findall(claim):
        if len(q.split()) >= 3 and normalize_for_eval(q) not in page_n:
            return False, 0.0, "quote_not_on_page"
    claim_toks = tokenize(claim)
    page_toks = set(tokenize(page_text))
    nums = [t for t in claim_toks if _NUM_RE.match(t)]
    missing_nums = [n for n in nums if n not in page_toks]
    if missing_nums:
        return False, 0.0, f"number_not_on_page:{','.join(missing_nums[:3])}"
    if not claim_toks:
        return False, 0.0, "no_content_tokens"
    recall = sum(t in page_toks for t in claim_toks) / len(claim_toks)
    return recall >= min_recall, round(recall, 4), ("overlap" if recall >= min_recall else "low_overlap")


def verify_sentences(
    sentences: list[str], page_texts: dict[tuple[str, int], str], min_recall: float = 0.6
) -> VerificationResult:
    result = VerificationResult()
    for s in sentences:
        cites = parse_citations(s)
        claim = strip_citations(s)
        if not claim:
            continue
        if not cites:
            result.checks.append(SentenceCheck(claim, [], False, 0.0, "no_citation"))
            continue
        best = SentenceCheck(claim, cites, False, 0.0, "page_not_found")
        for c in cites:
            text = page_texts.get((c.doc_id, c.page))
            if text is None:
                continue
            ok, score, reason = support(claim, text, min_recall)
            if ok and (not best.supported or score > best.score):
                best = SentenceCheck(claim, cites, True, score, reason, c, _best_span(claim, text))
            elif not best.supported and (score >= best.score or best.reason == "page_not_found"):
                best = SentenceCheck(claim, cites, False, score, reason, None)
        result.checks.append(best)
    return result
