from dastavez.chunking import chunk_pages


def test_every_chunk_keeps_its_page_and_never_crosses_pages(toy_pages):
    chunks = chunk_pages(toy_pages)
    pages = {(p["doc_id"], p["page"]): p["final_text"] for p in toy_pages}
    assert chunks
    for c in chunks:
        assert (c.doc_id, c.page) in pages
        # all chunk text must come from that page alone
        for line in c.text.split("\n"):
            assert line in pages[(c.doc_id, c.page)]


def test_headings_are_attached_and_carried():
    pages = [
        {
            "doc_id": "d",
            "page": 1,
            "page_label": "i",
            "blocks": [
                {"kind": "heading", "text": "3. Eligibility"},
                {
                    "kind": "paragraph",
                    "text": "A" * 50 + " first page text that is long enough to stand alone as a chunk." * 3,
                },
            ],
        },
        {
            "doc_id": "d",
            "page": 2,
            "page_label": "ii",
            "blocks": [
                {"kind": "paragraph", "text": "Continuation on the next page without its own heading. " * 4},
            ],
        },
    ]
    chunks = chunk_pages(pages)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].heading == "3. Eligibility"
    assert chunks[1].heading == "3. Eligibility"  # carried over the page break as context
    assert chunks[1].page_label == "ii"
    assert chunks[0].chunk_id == "d:p1:c0"


def test_long_paragraphs_split_on_sentences_within_limit():
    text = " ".join(f"Sentence number {i} explains one rule of the scheme in detail." for i in range(60))
    chunks = chunk_pages(
        [{"doc_id": "d", "page": 5, "blocks": [{"kind": "paragraph", "text": text}]}], max_chars=300
    )
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 * 1.2 for c in chunks)
    assert all(c.page == 5 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_tables_kept_whole_when_small():
    table = "Vertical | Assistance\nBLC | Rs 2.5 lakh\nAHP | Rs 2.5 lakh"
    chunks = chunk_pages([{"doc_id": "d", "page": 1, "blocks": [{"kind": "table", "text": table}]}])
    assert len(chunks) == 1 and chunks[0].text == table


def test_fallback_to_final_text_when_no_blocks():
    chunks = chunk_pages([{"doc_id": "d", "page": 9, "final_text": "Only plain text here, from OCR output."}])
    assert len(chunks) == 1 and chunks[0].page == 9
