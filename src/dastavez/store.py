"""Chunk stores: an in-process index (dev/tests) and PostgreSQL + pgvector (production).

Both expose the same three primitives the retriever needs:
``lexical(query, k)``, ``dense(vector, k)`` and ``get(ids)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .lexical import BM25
from .textutil import tokenize


class MemoryStore:
    backend = "memory"

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.emb = embeddings
        self.bm25 = BM25([f"{c.heading}\n{c.text}" for c in chunks])

    def lexical(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(self.chunks[i].chunk_id, s) for i, s in self.bm25.search(query, k)]

    def dense(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.emb is None or len(self.chunks) == 0:
            return []
        sims = self.emb @ vector.astype(np.float32)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.lexsort((idx, -sims[idx]))]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in idx]

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        return [self.by_id[i] for i in ids if i in self.by_id]

    def idf(self) -> dict[str, float]:
        return {t: self.bm25.idf(t) for t in self.bm25.df}

    # ------------------------------------------------------------------ persistence
    def save(self, folder: Path, meta: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        if self.emb is not None:
            np.save(folder / "embeddings.npy", self.emb)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> tuple[MemoryStore, dict]:
        with open(folder / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        emb_path = folder / "embeddings.npy"
        emb = np.load(emb_path) if emb_path.exists() else None
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, emb), meta


# ---------------------------------------------------------------------- PostgreSQL


def lexeme(token: str) -> str:
    """Map a token to an ASCII lexeme so Postgres' text-search parser never splits Devanagari
    combining marks (its behaviour there depends on the server locale)."""
    return "x" + hashlib.blake2b(token.encode("utf-8"), digest_size=6).hexdigest()


def lexemes(text: str) -> str:
    return " ".join(lexeme(t) for t in tokenize(text))


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id   text PRIMARY KEY,
    doc_id     text NOT NULL,
    page       integer NOT NULL,
    page_label text,
    heading    text NOT NULL DEFAULT '',
    title      text NOT NULL DEFAULT '',
    body       text NOT NULL,
    lex        text NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('simple', lex)) STORED,
    embedding  vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table}
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS {table}_tsv ON {table} USING gin (tsv);
CREATE INDEX IF NOT EXISTS {table}_doc_page ON {table} (doc_id, page);
"""


class PgVectorStore:
    """pgvector HNSW (cosine) for dense search + Postgres full-text search for lexical search.

    Lexical ranking uses ``ts_rank_cd`` over an OR-query of the question's lexemes; it is not
    BM25 (no length saturation the same way), which is one reason the in-process index keeps
    a true BM25 for evaluation.
    """

    backend = "pgvector"

    def __init__(self, dsn: str, dim: int, table: str = "dastavez_chunks", ef_search: int = 80):
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        self.table, self.dim, self.ef_search = table, dim, ef_search
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def create_schema(self, drop: bool = False) -> None:
        if drop:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        for stmt in SCHEMA.format(table=self.table, dim=self.dim).split(";"):
            if stmt.strip():
                self.conn.execute(stmt)

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        sql = f"""
            INSERT INTO {self.table} (chunk_id, doc_id, page, page_label, heading, title, body, lex, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET body = EXCLUDED.body, lex = EXCLUDED.lex,
                heading = EXCLUDED.heading, embedding = EXCLUDED.embedding
        """
        with self.conn.cursor() as cur:
            cur.executemany(
                sql,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.page,
                        c.page_label,
                        c.heading,
                        c.title,
                        c.text,
                        lexemes(f"{c.heading}\n{c.text}"),
                        np.asarray(e, dtype=np.float32),
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )

    def lexical(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(lexeme(t) for t in tokenize(query)))
        if not terms:
            return []
        tsq = " | ".join(terms)  # lexemes are [x0-9a-f]+ only: safe inside to_tsquery
        rows = self.conn.execute(
            f"""SELECT chunk_id, ts_rank_cd(tsv, q) AS r
                FROM {self.table}, to_tsquery('simple', %s) q
                WHERE tsv @@ q ORDER BY r DESC, chunk_id LIMIT %s""",
            (tsq, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def dense(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self.conn.transaction():
            self.conn.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            rows = self.conn.execute(
                f"""SELECT chunk_id, 1 - (embedding <=> %s) AS sim
                    FROM {self.table} ORDER BY embedding <=> %s LIMIT %s""",
                (np.asarray(vector, np.float32), np.asarray(vector, np.float32), k),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        rows = self.conn.execute(
            f"SELECT chunk_id, doc_id, page, page_label, heading, body, title FROM {self.table} WHERE chunk_id = ANY(%s)",
            (list(ids),),
        ).fetchall()
        by = {r[0]: Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows}
        return [by[i] for i in ids if i in by]

    def count(self) -> int:
        return int(self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
