import json
from pathlib import Path

import pytest

from dastavez.config import Settings
from dastavez.corpus import load_manifest, sha256_file, verify

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.json"


def test_manifest_is_complete_and_consistent():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = load_manifest(MANIFEST)
    assert len(docs) == len(data["documents"]) >= 10
    assert data["total_pages"] == sum(d.pages for d in docs)
    assert len({d.doc_id for d in docs}) == len(docs)
    for d in docs:
        assert len(d.sha256) == 64 and d.source_url.startswith("https://")
        assert d.language in {"en", "hi", "hi+en"} and d.source_kind in {"digital", "scanned"}
    strata = {d.stratum for d in docs}
    assert strata == {"english_digital", "hindi_digital", "scanned"}


def test_verify_rejects_tampered_file(tmp_path):
    docs = load_manifest(MANIFEST)
    d = docs[0]
    (tmp_path / d.filename).write_bytes(b"%PDF-1.4 not the real file")
    assert not verify(d, tmp_path)
    assert sha256_file(tmp_path / d.filename) != d.sha256


@pytest.mark.corpus
@pytest.mark.skipif(not (ROOT / "corpus" / "pdfs").exists(), reason="corpus not downloaded")
def test_real_broken_hindi_pdf_is_routed_to_ocr():
    """The PMAY-U 2.0 Hindi PDF's text layer is legacy-font mojibake; routing must catch it."""
    from dastavez.parse import layout_pages
    from dastavez.routing import decide

    s = Settings(home=ROOT)
    doc = next(d for d in load_manifest(MANIFEST) if d.doc_id == "pmay_u2_guidelines_2024_hi")
    if not verify(doc, s.pdf_dir):
        pytest.skip("PDF missing")
    pages = list(layout_pages(s.pdf_dir / doc.filename))
    routed = sum(decide(p.text, doc.language, p.image_coverage).needs_ocr for p in pages[:20])
    assert routed >= 18
