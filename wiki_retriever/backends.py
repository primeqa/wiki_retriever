"""
Vector database backend registry and helpers.

Extracted from benchmark_db.py so callers don't pull in the benchmark harness.
"""

import os


# ── Backend registry ────────────────────────────────────────────────
BACKENDS = {
    "lancedb": {
        "module": "dense_lancedb",
        "class": "DenseLanceDB",
        "env_unix_socket": "LANCEDB_USE_UNIX_SOCKET",
        "socket_prefix": ".lancedb_",
        "default_port": 8766,
    },
}


def infer_backend(db_uri: str) -> str:
    """Return the included backend; reject unsupported database formats."""
    if db_uri and (db_uri.endswith(".db") or db_uri.startswith(("http://", "https://"))):
        raise ValueError("wiki_retriever includes LanceDB only; provide a LanceDB directory")
    return "lancedb"


def _import_backend(backend: str):
    """Lazily import and return the retriever class for *backend*."""
    import importlib
    info = BACKENDS[backend]
    mod = importlib.import_module("." + info["module"], package=__package__)
    return getattr(mod, info["class"])
