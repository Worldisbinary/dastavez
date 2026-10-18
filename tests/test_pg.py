"""pgvector integration tests. Run in CI against the pgvector/pgvector service container;
skipped locally when DASTAVEZ_DATABASE_URL is not set."""

import os

import pytest

from dastavez.chunking import chunk_pages
from dastavez.embeddings import HashingEmbedder

pytestmark = pytest.mark.pg
DSN = os.environ.get("DASTAVEZ_DATABASE_URL")
needs_pg = pytest.mark.skipif(not DSN, reason="DASTAVEZ_DATABASE_URL not set")


@pytest.fixture
def pg_store(toy_pages):
    from dastavez.store import PgVectorStore

    emb = HashingEmbedder(64)
    chunks = chunk_pages(toy_pages)
    store = PgVectorStore(DSN, dim=64, table="dastavez_test_chunks")
    store.create_schema(drop=True)
    store.upsert(chunks, emb.embed_documents([c.embed_text() for c in chunks]))
    yield store, emb, chunks
    store.conn.execute("DROP TABLE IF EXISTS dastavez_test_chunks")
    store.close()


@needs_pg
def test_schema_has_hnsw_and_gin_indexes(pg_store):
    store, _, chunks = pg_store
    assert store.count() == len(chunks)
    rows = store.conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'dastavez_test_chunks'"
    ).fetchall()
    defs = " ".join(r[0] for r in rows).lower()
    assert "using hnsw" in defs and "vector_cosine_ops" in defs and "using gin" in defs


@needs_pg
def test_full_text_search_english_and_hindi(pg_store):
    store, _, _ = pg_store
    assert store.lexical("working capital loan street vendors", 3)[0][0].startswith("svanidhi:p3")
    assert store.lexical("कारपेट क्षेत्र", 3)[0][0].startswith("pmayu_hi:p7")
    assert store.lexical("zzzz qqqq", 3) == []


@needs_pg
def test_dense_search_and_get(pg_store):
    store, emb, _ = pg_store
    hits = store.dense(emb.embed_query("income support Rs. 6000 per year landholding farmer families"), 2)
    assert hits[0][0].startswith("pmkisan:p1")
    got = store.get([hits[0][0]])
    assert got[0].page == 1 and got[0].doc_id == "pmkisan"


@needs_pg
def test_retriever_over_pgvector(pg_store):
    from dastavez.rerank import LexicalReranker
    from dastavez.retrieve import Retriever

    store, emb, _ = pg_store
    r = Retriever(store, emb, LexicalReranker())
    for mode in ("lexical", "dense", "hybrid", "hybrid_rerank"):
        assert r.search("street vendors second tranche loan", mode=mode, k=2)[0].chunk.doc_id == "svanidhi"
