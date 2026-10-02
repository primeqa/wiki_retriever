"""
SQLite-backed document store with dict-like access.

Provides O(log n) lookups by ID without loading the entire store into memory.
Used as a drop-in replacement for the in-memory dict returned by
``DenseRetriever._load_doc_store()``.

Schema::

    CREATE TABLE docs (
        id   TEXT PRIMARY KEY,
        doc  TEXT NOT NULL
    );
"""

import sqlite3
import os


class SQLiteDocStore:
    """Dict-like read-only wrapper around a SQLite ``docs`` table.

    Supports ``store.get(key)``, ``store[key]``, ``len(store)``,
    and ``key in store`` — enough to be a drop-in for a plain dict
    wherever ``DenseRetriever`` uses the doc store.
    """

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SQLite doc store not found: {path}")
        # Read-only mode, URI-based opening
        self._path = path
        self._conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.execute("PRAGMA mmap_size=268435456")  # 256 MB mmap
        self._len = None  # lazily cached

    def get(self, key, default=None):
        row = self._conn.execute(
            "SELECT doc FROM docs WHERE id = ?", (str(key),)
        ).fetchone()
        return row[0] if row else default

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key):
        row = self._conn.execute(
            "SELECT 1 FROM docs WHERE id = ? LIMIT 1", (str(key),)
        ).fetchone()
        return row is not None

    def __len__(self):
        if self._len is None:
            self._len = self._conn.execute(
                "SELECT COUNT(*) FROM docs"
            ).fetchone()[0]
        return self._len

    def tables(self):
        """Return a list of table names in the database."""
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [r[0] for r in rows]

    def columns(self, table='docs'):
        """Return a list of column names for the given table."""
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in rows]

    def search(self, id_pattern):
        """Return all (id, doc) pairs where the id matches a LIKE pattern.

        Use ``%`` as wildcard. E.g. ``store.search('wiki_en%')``
        returns all entries whose id starts with ``wiki_en``.
        """
        return self._conn.execute(
            "SELECT id, doc FROM docs WHERE id LIKE ?", (id_pattern,)
        ).fetchall()

    def sample(self, n=10):
        """Return up to *n* (id, doc) pairs from the table."""
        return self._conn.execute(
            "SELECT id, doc FROM docs LIMIT ?", (n,)
        ).fetchall()

    def __repr__(self):
        return f"SQLiteDocStore({self._path!r}, n={len(self):,})"

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()


# ── Build helper ────────────────────────────────────────────────────

def create_sqlite_doc_store(path: str, items, batch_size: int = 50_000) -> int:
    """Create a SQLite doc store from an iterable of ``(id, doc)`` pairs.

    Args:
        path: Output ``.db`` file path (will be overwritten if it exists).
        items: Iterable of ``(id_str, doc_str)`` tuples.
        batch_size: Rows per INSERT transaction (default 50 000).

    Returns:
        Number of rows inserted.
    """
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE docs (id TEXT PRIMARY KEY, doc TEXT NOT NULL)"
    )

    n = 0
    batch = []
    for pair in items:
        batch.append(pair)
        if len(batch) >= batch_size:
            conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)", batch)
            conn.commit()
            n += len(batch)
            batch.clear()

    if batch:
        conn.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)", batch)
        conn.commit()
        n += len(batch)

    conn.execute("PRAGMA journal_mode=DELETE")  # compact on close
    conn.close()
    return n