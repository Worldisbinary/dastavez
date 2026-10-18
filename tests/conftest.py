from __future__ import annotations

import pytest

from dastavez.answer import Answerer
from dastavez.chunking import chunk_pages
from dastavez.embeddings import HashingEmbedder
from dastavez.rerank import LexicalReranker
from dastavez.retrieve import Retriever
from dastavez.store import MemoryStore

TOY_PAGES = [
    {
        "doc_id": "pmkisan",
        "page": 1,
        "page_label": "1",
        "blocks": [
            {"kind": "heading", "text": "1. Scheme"},
            {
                "kind": "paragraph",
                "text": "Under the PM-KISAN scheme an income support of Rs. 6000 per year is provided to all "
                "landholding farmer families. The amount is released in three equal instalments of Rs. 2000 each.",
            },
        ],
    },
    {
        "doc_id": "pmkisan",
        "page": 2,
        "page_label": "2",
        "blocks": [
            {"kind": "heading", "text": "4. Exclusion Criteria"},
            {
                "kind": "paragraph",
                "text": "Institutional landholders are excluded. Families in which a member paid income tax in the "
                "last assessment year are not eligible for benefit under the scheme.",
            },
        ],
    },
    {
        "doc_id": "pmayu_hi",
        "page": 7,
        "page_label": "5",
        "blocks": [
            {"kind": "heading", "text": "पात्रता"},
            {
                "kind": "paragraph",
                "text": "ईडब्ल्यूएस परिवारों की वार्षिक आय तीन लाख रुपये तक होनी चाहिए। "
                "आवास का न्यूनतम कारपेट क्षेत्र 30 वर्ग मीटर होगा।",
            },
        ],
    },
    {
        "doc_id": "svanidhi",
        "page": 3,
        "page_label": "3",
        "blocks": [
            {
                "kind": "paragraph",
                "text": "Street vendors are eligible for a first working capital loan of up to Rs. 15,000. "
                "Timely repayment makes the vendor eligible for a second tranche of up to Rs. 25,000.",
            }
        ],
    },
]


def _final_text(p: dict) -> str:
    return "\n\n".join(b["text"] for b in p["blocks"])


@pytest.fixture
def toy_pages() -> list[dict]:
    return [dict(p, final_text=_final_text(p)) for p in TOY_PAGES]


@pytest.fixture
def toy_store(toy_pages) -> tuple[MemoryStore, HashingEmbedder]:
    chunks = chunk_pages(
        toy_pages, {"pmkisan": "PM-KISAN Guidelines", "pmayu_hi": "PMAY-U 2.0", "svanidhi": "SVANidhi"}
    )
    emb = HashingEmbedder(256)
    return MemoryStore(chunks, emb.embed_documents([c.embed_text() for c in chunks])), emb


@pytest.fixture
def toy_answerer(toy_store, toy_pages) -> Answerer:
    store, emb = toy_store
    retriever = Retriever(store, emb, LexicalReranker(store.idf()), candidates=10, rerank_top=10)
    page_texts = {(p["doc_id"], p["page"]): p["final_text"] for p in toy_pages}
    return Answerer(retriever, page_texts, generator=None, gates={"lexical": 0.5})
