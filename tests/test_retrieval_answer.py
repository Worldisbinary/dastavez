import numpy as np

from dastavez.answer import NOT_FOUND, extractive_answer
from dastavez.embeddings import HashingEmbedder
from dastavez.retrieve import Retriever


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = (
        e.embed_documents(["income support for farmers"]),
        e.embed_documents(["income support for farmers"]),
    )
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5
    assert e.embed_documents([]).shape == (0, 128)


def test_memory_store_lexical_and_dense(toy_store):
    store, emb = toy_store
    lex = store.lexical("working capital loan street vendors", 3)
    assert lex[0][0].startswith("svanidhi:p3")
    den = store.dense(emb.embed_query("कारपेट क्षेत्र वर्ग मीटर"), 2)
    assert den[0][0].startswith("pmayu_hi:p7")


def test_all_modes_return_page_tagged_hits(toy_answerer):
    r: Retriever = toy_answerer.retriever
    for mode in ("lexical", "dense", "hybrid", "hybrid_rerank"):
        hits = r.search("How many instalments does PM-KISAN pay?", mode=mode, k=3)
        assert hits, mode
        assert all(isinstance(h.chunk.page, int) for h in hits)
        assert hits[0].chunk.doc_id == "pmkisan", mode


def test_hybrid_contains_union_of_signals(toy_answerer):
    r = toy_answerer.retriever
    hits = r.search("income tax exclusion", mode="hybrid", k=10)
    ids = {h.chunk.chunk_id for h in hits}
    assert any(i.startswith("pmkisan:p2") for i in ids)
    assert all("lexical" in h.ranks and "dense" in h.ranks for h in hits)


def test_answer_is_cited_and_verified(toy_answerer):
    res = toy_answerer.ask("What is the income support per year under PM-KISAN?", mode="hybrid_rerank", k=3)
    assert not res.not_found
    assert ("pmkisan", 1) in res.citations
    assert all(s.supported and s.supporting for s in res.sentences)
    assert "[pmkisan p.1]" in res.answer


def test_hindi_question_gets_hindi_citation(toy_answerer):
    res = toy_answerer.ask("ईडब्ल्यूएस परिवारों की वार्षिक आय कितनी होनी चाहिए?", k=3)
    assert not res.not_found
    assert res.citations[0] == ("pmayu_hi", 7)


def test_unanswerable_returns_not_found(toy_answerer):
    res = toy_answerer.ask("What is the interest rate on Sukanya Samriddhi deposits?", k=3)
    assert res.not_found and res.answer == NOT_FOUND


def test_extractive_answer_quotes_with_citations(toy_answerer):
    hits = toy_answerer.retriever.search("second tranche loan", mode="lexical", k=2)
    sents = extractive_answer("second tranche loan amount", hits)
    assert sents and sents[0].endswith("[svanidhi p.3]")


def test_generator_output_is_still_verified(toy_answerer):
    class FakeGen:
        name = "fake"

        def generate(self, question, hits):
            return (
                "Farmers receive Rs. 6000 per year [pmkisan p.1]. "
                "They also get a free tractor worth Rs. 500000 [pmkisan p.1]."
            )

    toy_answerer.generator = FakeGen()
    res = toy_answerer.ask("What do farmers receive under PM-KISAN per year?", k=3)
    assert not res.not_found
    assert len(res.sentences) == 1 and "6000" in res.sentences[0].sentence
    assert len(res.dropped) == 1 and res.dropped[0].reason.startswith("number_not_on_page")

    class NotFoundGen:
        name = "fake"

        def generate(self, question, hits):
            return "NOT_FOUND"

    toy_answerer.generator = NotFoundGen()
    assert toy_answerer.ask("What do farmers receive under PM-KISAN per year?", k=3).not_found
