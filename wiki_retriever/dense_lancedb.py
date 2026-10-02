import os
from .lancedb_server_management import create_lancedb_server
from .dense_retriever import DenseRetriever

class DenseLanceDB(DenseRetriever):
    """
    Dense retriever backed by LanceDB.

    *db_uri* must be a local directory path.  A gRPC server is optionally
    started for multi-client access.
    """

    _backend_name = "lancedb"

    def __init__(self, model_name, endpoint, db_uri,
                 collection_name="nq_granite125m_512_100_20250530",
                 use_api: bool = True,
                 doc_store_path: str = None,
                 start_server: bool = True):
        """
        Args:
            model_name: Embedding model identifier.
            endpoint:   RITS embedding endpoint.
            db_uri: Path to LanceDB database directory.
            collection_name: LanceDB table name.
            use_api: Use gRPC server mode.
                None discovers an existing server, otherwise uses local access.
            start_server: Whether API mode may start a new gRPC server.
            doc_store_path: Optional path to a serialized document store file
                (see ``DenseRetriever``).
        """
        if db_uri is None:
            raise ValueError(
                "db_uri is required for DenseLanceDB. "
                "Provide a path to the LanceDB database directory."
            )
        self._start_server = start_server
        super().__init__(model_name, endpoint, db_uri, collection_name, use_api,
                         doc_store_path=doc_store_path)

    def _init_backend(self):
        socket_path = None
        if os.environ.get("LANCEDB_USE_UNIX_SOCKET", "").lower() in ("true", "1", "yes"):
            import hashlib
            db_dir = os.path.dirname(os.path.abspath(self.db_uri))
            db_hash = hashlib.sha256(self.db_uri.encode()).hexdigest()[:16]
            socket_path = os.path.join(db_dir, f".lancedb_{db_hash}.sock")

        self.client = create_lancedb_server(
            db_path=self.db_uri,
            socket_path=socket_path,
            use_api=self._use_api,
            start_server=self._start_server,
            port=8766,
        )
        self.engine_type = "api" if self.client.use_api else "lancedb"
