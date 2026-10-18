import json

import pytest

from dastavez import annotation
from dastavez.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DASTAVEZ_HOME", str(tmp_path))
    monkeypatch.delenv("DASTAVEZ_DATA_DIR", raising=False)
    s = Settings()
    pages_dir = s.data_dir / "pages"
    pages_dir.mkdir(parents=True)
    recs = []
    for doc, stratum, lang, n in [
        ("en_doc", "english_digital", "en", 30),
        ("hi_doc", "hindi_digital", "hi", 20),
        ("scan_doc", "scanned", "hi+en", 6),
    ]:
        for p in range(1, n + 1):
            text = "" if p == 1 else f"page {p} of {doc} " * 10  # page 1 is blank -> ineligible
            recs.append(
                {
                    "doc_id": doc,
                    "page": p,
                    "stratum": stratum,
                    "language": lang,
                    "final_text": text,
                    "pypdf_text": text,
                    "layout_text": text,
                    "ocr_text": None,
                }
            )
    for doc in ("en_doc", "hi_doc", "scan_doc"):
        with open(pages_dir / f"{doc}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                if r["doc_id"] == doc:
                    f.write(json.dumps(r) + "\n")
    return s


def test_sampling_is_stratified_deterministic_and_skips_blank_pages(settings):
    out = annotation.sample_pages(settings, per_stratum=4, seed=7)
    assert out["n"] == 12
    first = json.loads((settings.gold_dir / "sample.json").read_text(encoding="utf-8"))["pages"]
    assert {p["stratum"] for p in first} == set(annotation.STRATA)
    assert all(p["page"] != 1 for p in first)
    assert annotation.sample_pages(settings, per_stratum=4, seed=7)["status"].startswith("exists")
    again = annotation.sample_pages(settings, per_stratum=4, seed=7, force=True)
    assert again["n"] == 12
    second = json.loads((settings.gold_dir / "sample.json").read_text(encoding="utf-8"))["pages"]
    assert first == second


def test_sampling_caps_at_pool_size(settings):
    out = annotation.sample_pages(settings, per_stratum=50, seed=1)
    assert out["eligible"] == {"english_digital": 29, "hindi_digital": 19, "scanned": 5}
    assert out["n"] == 29 + 19 + 5


def _write(settings, who, sid, text):
    p = settings.gold_dir / who / f"{sid}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_agreement_and_adjudication_flow(settings):
    annotation.sample_pages(settings, per_stratum=1, seed=3)
    ids = [p["sample_id"] for p in annotation.load_sample(settings)]
    a_id, b_id, c_id = ids
    _write(settings, "annotator_a", a_id, "योजना की अवधि 5 वर्ष है।")
    _write(settings, "annotator_b", a_id, "योजना की  अवधि ५ वर्ष है।")  # identical after normalisation
    _write(settings, "annotator_a", b_id, "Loan of Rs 10000 for one year")
    _write(settings, "annotator_b", b_id, "Loan of Rs 1000 for one year")
    # c_id annotated by A only
    _write(settings, "annotator_a", c_id, "only one annotator")

    rep = annotation.agreement_report(settings)
    assert rep["pages_double_annotated"] == 2
    assert rep["summary"]["all"]["n"] == 2
    assert len(rep["low_agreement"]) == 0  # 0.98 similarity > 0.95
    adj = annotation.adjudicate(settings, threshold=0.99)
    assert adj["accepted_identical"] == 1
    assert adj["needs_adjudication"] == [b_id]
    assert adj["missing_annotations"] == 1
    assert (settings.gold_dir / "final" / f"{a_id}.txt").exists()
    diff = (settings.gold_dir / "adjudication" / f"{b_id}.md").read_text(encoding="utf-8")
    assert "[-A: 10000 -][+B: 1000 +]" in diff

    kind, refs = annotation.reference_texts(settings)
    assert kind == "gold" and set(refs) == {a_id}


def test_reference_falls_back_to_silver(settings):
    annotation.sample_pages(settings, per_stratum=1, seed=3)
    sid = annotation.load_sample(settings)[0]["sample_id"]
    _write(settings, "silver", sid, "provisional text")
    kind, refs = annotation.reference_texts(settings)
    assert kind == "silver" and refs == {sid: "provisional text"}


def test_word_diff():
    assert annotation.word_diff("a b c", "a x c") == "a [-A: b -][+B: x +] c"
