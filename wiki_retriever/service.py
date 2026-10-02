"""Text search HTTP API. One embedding model is shared across RL requests."""
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=32000)
    top_k: int = Field(default=5, ge=1, le=100)


def create_app(db, table, model, endpoint=None, device=None, retriever=None,
               doc_store_path=None):
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal retriever
        if retriever is None:
            from .dense_lancedb import DenseLanceDB
            if endpoint:
                retriever = DenseLanceDB(model, endpoint, db, table, use_api=False,
                                         doc_store_path=doc_store_path)
            else:
                from .dense_retriever import DenseRetrieverLocal
                class LocalLanceDB(DenseRetrieverLocal):
                    _backend_name = "lancedb"
                    def _init_backend(self):
                        from .lancedb_client_adapter import LanceDBClientAdapter
                        self.client = LanceDBClientAdapter(self.db_uri)
                        self.engine_type = "lancedb"
                retriever = LocalLanceDB(model, db, table, use_api=False, device=device,
                                         doc_store_path=doc_store_path)
            if not retriever.client.has_collection(table):
                raise ValueError(f"Table {table!r} does not exist")
        try:
            yield
        finally:
            retriever.client.close()

    app = FastAPI(title="Wiki Retriever", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"healthy": retriever.client.has_collection(table), "table": table}

    @app.post("/search")
    def search(request: SearchRequest):
        # Legacy retriever timing counters and local encoders share mutable state.
        with lock:
            return {"results": retriever.search(request.query, request.top_k)}

    return app
