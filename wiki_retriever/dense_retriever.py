from typing import List, Union, Any, LiteralString
from openai import OpenAI, BadRequestError
import os
import re
import bz2
import gzip
import json
import lzma
import pickle
import yaml
try:
    from .timer import timer
    timer_available = True
except ImportError:
    timer_available = False

class DenseRetriever():
    """
    Base class for dense vector retrieval against a vector database.

    Handles embedding computation (via OpenAI-compatible RITS API), search
    result formatting, and timing.  Subclasses only need to implement
    ``_init_backend()`` to set ``self.client`` and ``self.engine_type``.

    Attributes:
        model (str): Embedding model name.
        db_uri (str): Database URI — a URL, file path, or directory path
            depending on the backend.
        client: Backend client (MilvusClient, MilvusServer, LanceDBServer, etc.).
        collection_name (str): Collection/table name in the vector database.
        engine_type (str): "api" for gRPC server mode, or a backend-specific
            identifier (e.g. "milvus", "lancedb") for direct mode.
    """

    # Subclasses should set this to "milvus", "lancedb", etc.
    _backend_name: str = "base"
    id_format = re.compile("(.*)[-_](\\d+)-(\\d+)")

    def __init__(self, model_name: str,
                 endpoint: str,
                 db_uri: str,
                 collection_name: str = "nq_granite125m_512_100_20250530",
                 use_api: bool = True,
                 doc_store_path: str = None):
        """
        Common initialisation shared by all dense retriever backends.

        Args:
            model_name: Embedding model identifier.
            endpoint:   RITS embedding endpoint (base URL, without /v1).
            db_uri:     Database URI, **or** a path to a ``.yaml``, ``.json``,
                        or ``.jsonl`` config file whose ``db_uri`` (and
                        optionally ``doc_store_path``) fields are used instead.
                        When a plain URI is given, interpretation depends on
                        the backend:
                        - Milvus: a URL (``http://host:port``) for a remote
                          server, or a local ``.db`` file path.
                        - LanceDB: a local directory path.
            collection_name: Collection/table name.  Hyphens are converted
                to underscores automatically.
            use_api: Whether to use gRPC server mode for local databases.
                Ignored when *db_uri* points to a remote server.
            doc_store_path: Optional path to a serialized document store file
                mapping original document IDs to document content.  Supported
                formats (auto-detected by extension):

                * ``.pkl`` / ``.pickle`` — pickle protocol 5 (fastest load,
                  recommended for large stores)
                * ``.json`` — UTF-8 JSON (human-readable fallback)

                When provided and ``search()`` is called with
                ``return_docs=True``, each result is augmented with a ``'doc'``
                key containing the full document looked up via
                ``_get_orig_docid()``.
        """
        # Config-file indirection ----------------------------------------
        # If db_uri points to a .yaml/.json/.jsonl config file, read it and
        # replace db_uri (and optionally doc_store_path) from its contents.
        collection_name, db_uri, doc_store_path, _ = self.read_db_config(db_uri, collection_name, doc_store_path)

        # Timer ----------------------------------------------------------
        if timer_available:
            self.tm = timer("retrieval")
        else:
            self.tm = None

        # Embedding client ------------------------------------------------
        api_key = os.environ.get("RITS_API_KEY")
        if api_key is None:
            raise ValueError(
                "RITS_API_KEY environment variable is not set. "
                "Please set it to use the embedding API."
            )

        self.openai = OpenAI(
            api_key=api_key,
            base_url=endpoint + '/v1',
            default_headers={'RITS_API_KEY': api_key},
        )
        self.model = model_name

        # Normalise collection name ---------------------------------------
        self.collection_name = collection_name

        # Store common params for _init_backend() -------------------------
        self.db_uri = db_uri
        self._use_api = use_api

        # Backend-specific setup (sets self.client, self.engine_type) -----
        self._init_backend()
        if timer_available:
            self.tm.add_timing("Init::init_vector_db")

        # Document store (optional) ---------------------------------------
        self.doc_store = (
            DenseRetriever._load_doc_store(doc_store_path)
            if doc_store_path is not None
            else None
        )
        if timer_available:
            self.tm.add_timing("Init::load_doc_store")

        # Counters --------------------------------------------------------
        self.num_queries = 0
        print('=== done initializing model')

    @staticmethod
    def read_db_config(db_uri: str, collection_name: Any | None=None, doc_store_path: str | None=None) -> tuple[
        Any | None, LiteralString, LiteralString, str | None]:
        backend = None
        if db_uri is not None:
            _ext = os.path.splitext(db_uri)[1].lower()
            if _ext in ('.yaml', '.yml', '.json', '.jsonl'):
                config_dir = os.path.dirname(db_uri)
                dd = DenseRetriever._read_config(db_uri)
                db_uri = dd['db_uri']
                doc_store_path = dd.get('doc_store_path', doc_store_path)
                collection_name = dd.get('collection_name', collection_name)
                backend = dd.get('backend', None)
                # Check to see if the doc_store_path and db_uri are relative paths and adjust with the directory
                # of the file db_uri:
                doc_store_path = DenseRetriever._adjust_path(config_dir, doc_store_path)
                db_uri = DenseRetriever._adjust_path(config_dir, db_uri)
        return collection_name, db_uri, doc_store_path, backend

    @staticmethod
    def _adjust_path(config_dir: str | bytes | LiteralString | Any, file_path: Any | None) -> LiteralString:
        orig_file_path = file_path
        if file_path is not None and not os.path.isabs(file_path):
            file_path = os.path.join(config_dir, file_path)
            if not os.path.exists(file_path):
                raise ValueError(f"Neither {orig_file_path} nor {file_path} exist - fix the db_uri config file.")
        return file_path

    # -- abstract --------------------------------------------------------
    def _init_backend(self):
        """Subclasses must set ``self.client`` and ``self.engine_type``."""
        raise NotImplementedError

    # -- embedding -------------------------------------------------------
    def __call__(self, texts: Union[List[str], str]) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.compute_embedding(texts)

    @property
    def tokenizer(self):
        """Return tokenizer - not available for embedding models"""
        raise NotImplementedError(
            "Tokenizer not available for embedding models. "
            "Use the model's native tokenizer."
        )

    def compute_embedding(self, text) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        query_vector = self.openai.embeddings.create(model=self.model, input=text)
        if len(query_vector.data) == 1:
            return query_vector.data[0].embedding
        return [q.embedding for q in query_vector.data]

    def get_query(self, text, domains: list[str] = None,
                  exclude_domains: list[str] = None):
        try:
            query_vector = self.compute_embedding(text)
        except BadRequestError:
            # most likely due to query exceeding seq length limits
            query_vector = self.compute_embedding(' '.join(text.split()[:200]))
        return query_vector

    # -- search ----------------------------------------------------------
    def search(self, query, top_k, return_docs: bool = False):
        self.num_queries += 1
        if timer_available:
            self.tm.mark()

        query_embedding = self.compute_embedding(query)
        if timer_available:
            self.tm.add_timing("run::compute_embedding")

        # Direct-client mode wraps embedding in a list; API mode passes it bare.
        if self.engine_type == "api":
            search_data = query_embedding
        else:
            search_data = [query_embedding]


        res = self.client.search(
            collection_name=self.collection_name,
            data=search_data,
            limit=top_k,
            output_fields=["title", "text", "url"],
        )
        if timer_available:
            self.tm.add_timing(f"run::{self._backend_name}_{self.engine_type}_search")

        # Flatten nested lists (some backends return [[results]])
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            res = res[0]

        documents = []
        for document in res:
            raw_id = document.get('id') if return_docs else None

            if 'entity' in document:
                document['entity']['score'] = document['distance']
                doc = dict(document['entity'], id=document.get('id'))
            else:
                document['score'] = document['distance']
                doc = document

            if return_docs and raw_id is not None and self.doc_store is not None:
                orig_id = DenseRetriever._get_orig_docid(raw_id)
                print("Document ID:", orig_id)
                doc['doc'] = self.doc_store.get(orig_id)
                if doc['doc'] is None:
                    # Check to see if the document id is still looking processed ID:
                    docid = DenseRetriever._get_orig_docid(orig_id)
                    print("Document ID:", docid)
                    if docid != orig_id:
                        doc['doc'] = self.doc_store.get(orig_id)
                    if doc['doc'] is None:
                        # try to convert the ID to int
                        try:
                            doc['doc'] = self.doc_store.get(int(docid))
                        except ValueError:
                            pass

            documents.append(doc)

        if timer_available:
            self.tm.add_timing("run::format_results")
        return documents

    # -- timings ---------------------------------------------------------
    def display_timings(self):
        if self.tm is not None:
            self.tm.display_timing(
                self.tm.milliseconds_since_beginning(),
                keys={"queries": self.num_queries},
            )

    def _read_config(path: str) -> dict:
        """
        Read a config file and return its contents as a dict.

        Supported formats (auto-detected by extension):

        * ``.yaml`` / ``.yml`` — YAML
        * ``.json`` — JSON
        * ``.jsonl`` — JSON Lines; the **first** line is parsed

        The returned dict must contain at least a ``db_uri`` key.  An
        optional ``doc_store_path`` key is also recognised.

        Args:
            path: Path to the config file.

        Returns:
            dict with at least a ``db_uri`` key.

        Raises:
            KeyError: If the config file does not contain ``db_uri``.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.yaml', '.yml'):
            with open(path, 'r', encoding='utf-8') as f:
                dd = yaml.safe_load(f)
        elif ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                dd = json.load(f)
        elif ext == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                dd = json.loads(f.readline())
        else:
            raise ValueError(
                f"Unsupported config file extension: {ext!r}. "
                "Use .yaml, .yml, .json, or .jsonl."
            )
        if 'db_uri' not in dd:
            raise KeyError(
                f"Config file {path!r} must contain a 'db_uri' key."
            )
        return dd

    def _load_doc_store(path: str):
        """
        Load a document store from *path* and return a dict-like object.

        The serialization format is inferred from the file extension(s):

        * ``.db`` / ``.sqlite`` — SQLite database (lazy lookups, no memory overhead).
        * ``.pkl`` / ``.pickle`` — plain pickle (loaded into memory).
        * ``.pkl.gz`` / ``.pickle.gz`` — gzip-compressed pickle.
        * ``.pkl.bz2`` / ``.pickle.bz2`` — bzip2-compressed pickle.
        * ``.pkl.xz`` / ``.pickle.xz`` — xz/lzma-compressed pickle.
        * ``.json`` — UTF-8 JSON (human-readable fallback).

        Args:
            path: Path to the serialized document store file.

        Returns:
            dict (or dict-like ``SQLiteDocStore``) mapping document IDs
            to document content.

        Raises:
            ValueError: If the file extension is not recognised.
            FileNotFoundError: If *path* does not exist.
        """
        from pathlib import Path as _Path
        suffixes = [s.lower() for s in _Path(path).suffixes]
        # SQLite database
        if suffixes and suffixes[-1] in ('.db', '.sqlite'):
            from .sqlite_doc_store import SQLiteDocStore
            return SQLiteDocStore(path)
        # compressed pickle: e.g. docs.pkl.gz / docs.pkl.bz2 / docs.pkl.xz
        if len(suffixes) >= 2 and suffixes[-2] in ('.pkl', '.pickle'):
            _openers = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}
            opener = _openers.get(suffixes[-1])
            if opener:
                with opener(path, 'rb') as f:
                    return pickle.load(f)
        # plain pickle
        if suffixes and suffixes[-1] in ('.pkl', '.pickle'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        # JSON
        if suffixes and suffixes[-1] == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise ValueError(
            f"Unsupported doc_store file: {path!r}. "
            "Use .db (SQLite), .pkl.gz / .pkl.bz2 / .pkl.xz (compressed), .pkl, or .json."
        )

    def _get_orig_docid(id):
        """
        Determines and returns the original document ID by processing the input ID.

        If the input ID is an integer, it is returned as-is. For string IDs, it identifies
        and returns the substring up to the second-to-last occurrence of a hyphen ("-"),
        or returns the original string if there are fewer than two hyphens.

        Args:
            id (int | str): The ID to process. It can be an integer or a string
                containing hyphens.

        Returns:
            int | str: The processed original document ID. If the input ID is an
                integer, it is returned directly. If it is a string, the function
                returns the substring up to the second-to-last hyphen, or the original
                string if the hyphen criteria are not met.
        """
        if isinstance(id, int):
            return id
        m = DenseRetriever.id_format.match(id)
        if m:
            return m.group(1)
        else:
            return id

class DenseRetrieverLocal(DenseRetriever):
    def __init__(self, model_name_or_path: str,
                 db_uri: str,
                 collection_name: str = "nq_granite125m_512_100_20250530",
                 use_api: bool = True,
                 doc_store_path: str = None,
                 use_query_prompt = False,
                 device='cpu'):
        """
        Common initialisation shared by all dense retriever backends.

        Args:
            model_name_or_path: Embedding model identifier.
            db_uri:     Database URI, **or** a path to a ``.yaml``, ``.json``,
                        or ``.jsonl`` config file whose ``db_uri`` (and
                        optionally ``doc_store_path``) fields are used instead.
                        When a plain URI is given, interpretation depends on
                        the backend:
                        - Milvus: a URL (``http://host:port``) for a remote
                          server, or a local ``.db`` file path.
                        - LanceDB: a local directory path.
            collection_name: Collection/table name.  Hyphens are converted
                to underscores automatically.
            use_api: Whether to use gRPC server mode for local databases.
                Ignored when *db_uri* points to a remote server.
            doc_store_path: Optional path to a serialized document store file
                mapping original document IDs to document content.  Supported
                formats (auto-detected by extension):

                * ``.pkl`` / ``.pickle`` — pickle protocol 5 (fastest load,
                  recommended for large stores)
                * ``.json`` — UTF-8 JSON (human-readable fallback)

                When provided and ``search()`` is called with
                ``return_docs=True``, each result is augmented with a ``'doc'``
                key containing the full document looked up via
                ``_get_orig_docid()``.
            use_query_prompt: whether embedding model uses a query prompt
            device: device to load model on
        """
        # Config-file indirection ----------------------------------------
        # If db_uri points to a .yaml/.json/.jsonl config file, read it and
        # replace db_uri (and optionally doc_store_path) from its contents.
        collection_name, db_uri, doc_store_path, _ = self.read_db_config(db_uri, collection_name, doc_store_path)

        # Timer ----------------------------------------------------------
        if timer_available:
            self.tm = timer("retrieval")
        else:
            self.tm = None

        # Embedding client ------------------------------------------------
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.use_query_prompt = use_query_prompt

        # Normalise collection name ---------------------------------------
        self.collection_name = collection_name

        # Store common params for _init_backend() -------------------------
        self.db_uri = db_uri
        self._use_api = use_api

        # Backend-specific setup (sets self.client, self.engine_type) -----
        self._init_backend()
        if timer_available:
            self.tm.add_timing("Init::init_vector_db")

        # Document store (optional) ---------------------------------------
        self.doc_store = (
            DenseRetriever._load_doc_store(doc_store_path)
            if doc_store_path is not None
            else None
        )
        if timer_available:
            self.tm.add_timing("Init::load_doc_store")

        # Counters --------------------------------------------------------
        self.num_queries = 0
        print('=== done initializing model')

    
    @property
    def tokenizer(self):
        """Return tokenizer"""
        return self.model.tokenizer

    def compute_embedding(self, text) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        if not isinstance(text, list):
            text = [text]

        if self.use_query_prompt:
            # models like Qwen 3 need query prompt
            query_vector = self.model.encode(text, prompt_name="query", show_progress_bar=False).tolist()
        else:
            query_vector = self.model.encode(text, show_progress_bar=False).tolist()
    
        return query_vector[0] if len(query_vector) == 1 else query_vector
