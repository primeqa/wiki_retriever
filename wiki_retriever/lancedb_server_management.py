"""
LanceDB server management module.

Mirrors milvus_server_management.py but uses LanceDB as the vector database backend.
Reuses the same gRPC protocol and ServerRegistry infrastructure for cross-machine
server discovery and multi-client support.
"""

import os
import threading
import time
import socket
import random
import fcntl
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy

# Direct file execution needs the package root for relative imports.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "wiki_retriever"

from .server_common import ServerRegistry, ServerInfo, MilvusAPIClient, Utils

class LanceDBServerInstance:
    """Singleton server instance that manages the LanceDB database and serves gRPC requests.

    Use create_instance() static method to create new instances. Direct instantiation
    is not allowed to ensure proper singleton pattern implementation.
    """

    _instance = None
    _lock_file = None
    _lock_fd = None

    @staticmethod
    def create_instance(db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8766):
        """Create a new LanceDBServerInstance with thread-safety.

        Args:
            db_path: Path to LanceDB database directory
            socket_path: Unix domain socket path (if None, uses TCP with host:port)
            host: Host address to bind server to (for TCP mode)
            port: Port to run server on (for TCP mode)

        Returns:
            New or existing LanceDBServerInstance

        Raises:
            RuntimeError: If server is already running
        """
        lock_dir = os.path.dirname(os.path.abspath(db_path))
        lock_file = os.path.join(lock_dir, ".lancedb_instance.lock")
        LanceDBServerInstance._lock_file = lock_file

        try:
            LanceDBServerInstance._lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY | os.O_EXCL)

            print(f"Process {os.getpid()} locking LanceDB instance")
            fcntl.flock(LanceDBServerInstance._lock_fd, fcntl.LOCK_EX)

            if not LanceDBServerInstance._instance:
                LanceDBServerInstance._instance = LanceDBServerInstance(db_path, socket_path, host, port)

            print(f"Process {os.getpid()} finished creating LanceDB instance (server not started yet).", flush=True)
            return LanceDBServerInstance._instance

        except FileExistsError:
            try:
                LanceDBServerInstance._attempt_lock_cleanup(lock_file)
                return None
            except OSError:
                return None

        except OSError as e:
            raise RuntimeError(f"Failed to acquire instance lock: {e}")
        finally:
            if LanceDBServerInstance._lock_fd is not None:
                fcntl.flock(LanceDBServerInstance._lock_fd, fcntl.LOCK_UN)
                os.close(LanceDBServerInstance._lock_fd)
                LanceDBServerInstance._lock_fd = None

    @staticmethod
    def _attempt_lock_cleanup(lock_file: str):
        if os.path.exists(lock_file):
            file_age = time.time() - os.path.getmtime(lock_file)
            if file_age > 10:
                try:
                    os.remove(lock_file)
                    print(f"Removed stale lock file (age: {file_age:.1f}s)")
                    return True
                except OSError:
                    pass
        return False

    @staticmethod
    def remove_creation_lock():
        """Remove the LanceDBServerInstance lock."""
        try:
            os.remove(LanceDBServerInstance._lock_file)
        except OSError:
            pass

    @staticmethod
    def creation_in_progress():
        if LanceDBServerInstance._lock_fd is not None:
            LanceDBServerInstance._attempt_lock_cleanup(LanceDBServerInstance._lock_file)
            return os.path.exists(LanceDBServerInstance._lock_file)
        else:
            return False

    def __new__(cls, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8766):
        db_path = os.path.abspath(db_path)
        instance = super().__new__(cls)
        instance._initialized = False
        instance.registry = ServerRegistry(db_path, registry_dir=os.path.join(
            os.path.dirname(os.path.abspath(db_path)), '.lancedb_servers'))

        server_info = instance.registry.get_server_info()
        if server_info and instance.registry.is_server_alive(server_info):
            location = server_info.socket_path if server_info.socket_path else f"{server_info.host}:{server_info.port}"
            raise RuntimeError(f"Server already running at {location}")

        return instance

    def __init__(self, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8766):
        """Initialize LanceDBServerInstance (private constructor)."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing LanceDBServerInstance")

        self.db_path = os.path.abspath(db_path)
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self.client = None
        self.grpc_server = None
        self.server_thread = None
        self.running = False
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO)

        self.registry = ServerRegistry(self.db_path, registry_dir=os.path.join(
            os.path.dirname(self.db_path), '.lancedb_servers'))

        with self.registry._acquire_lock():
            if hasattr(self, '_initialized') and self._initialized:
                return
            self._initialize_client()
            self._initialized = True

    def _get_external_ip(self) -> str:
        """Get the external IP address of this machine."""
        if self.host != "0.0.0.0":
            return self.host

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def _initialize_client(self):
        """Initialize the LanceDB client adapter."""
        try:
            os.makedirs(self.db_path, exist_ok=True)

            from .lancedb_client_adapter import LanceDBClientAdapter
            self.client = LanceDBClientAdapter(uri=self.db_path)
            self.logger.info(f"LanceDB client initialized with database at: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize LanceDB client: {e}")
            raise

    def start_server(self, threaded: bool = True):
        """Start the gRPC server and register it."""
        if self.running:
            location = self.socket_path if self.socket_path else f"{self.host}:{self.port}"
            self.logger.warning(f"Server already running on {location}")
            return

        def run_server():
            try:
                from .milvus_grpc_server import serve_grpc

                if self.socket_path:
                    self.logger.info(f"Starting LanceDB gRPC server on Unix socket: {self.socket_path}")
                else:
                    self.logger.info(f"Starting LanceDB gRPC server on {self.host}:{self.port}")

                self.grpc_server = serve_grpc(self.client, self.host, self.port, socket_path=self.socket_path)
                self.grpc_server.wait_for_termination()
            except Exception as e:
                self.logger.error(f"Server error: {e}")

        if threaded:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            time.sleep(2)

            if self._is_server_running():
                if self.socket_path:
                    self.registry.register_server(pid=os.getpid(), socket_path=self.socket_path)
                    self.running = True
                    self.logger.info(f"Server successfully started on Unix socket {self.socket_path}")
                else:
                    external_ip = self._get_external_ip()
                    self.registry.register_server(pid=os.getpid(), host=external_ip, port=self.port)
                    self.running = True
                    self.logger.info(f"Server successfully started on {external_ip}:{self.port}")
            else:
                raise RuntimeError("Failed to start server")
        else:
            run_server()

    def _is_server_running(self) -> bool:
        """Check if the server is running."""
        try:
            from .milvus_grpc_client import MilvusGRPCClient
            client = MilvusGRPCClient("localhost", self.port)
            result = client.health_check()
            client.close()
            return result
        except Exception:
            return False

    def stop_server(self):
        """Stop the server and unregister it."""
        self.running = False
        self.registry.unregister_server()
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except Exception:
                pass
        self.logger.info("Server stop requested and unregistered")


class LanceDBServer:
    """Main LanceDB Server class that handles both server and client functionality."""

    def __init__(self, db_path: str, socket_path: str = None, host: str = "0.0.0.0", port: int = 8766,
                 use_api: bool = None, start_server: bool = True,
                 registry_dir: str = None):
        """
        Initialize LanceDBServer with cross-machine support.

        Args:
            db_path: Path to the LanceDB database directory
            socket_path: Unix domain socket path (if provided, uses Unix sockets instead of TCP)
            host: Host address for API server (use "0.0.0.0" to bind to all interfaces)
            port: Port for API server
            use_api: If True, use API client. If False, use direct file access. If None, auto-detect
            start_server: Whether to start the API server automatically
            registry_dir: Directory for server registry files (defaults to {db_dir}/.lancedb_servers)
        """
        self.db_path = os.path.abspath(db_path)
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self.use_api = use_api
        self.server_instance = None
        self.client = None
        self.api_client = None

        if registry_dir is None:
            registry_dir = os.path.join(os.path.dirname(self.db_path), '.lancedb_servers')

        self.registry = ServerRegistry(self.db_path, registry_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)

        self.registry.cleanup_if_dead()

        if self.use_api is None:
            self.use_api = self._should_use_api()

        if self.use_api:
            self._initialize_api_client(start_server)
        else:
            self._initialize_direct_client()

    def _should_use_api(self) -> bool:
        """Determine whether to use API mode based on server availability."""
        server_info = self.registry.get_server_info()
        if server_info and self.registry.is_server_alive(server_info):
            if server_info.socket_path:
                self.logger.info(f"Found existing server at {server_info.socket_path}")
                self.socket_path = server_info.socket_path
            else:
                self.logger.info(f"Found existing server at {server_info.host}:{server_info.port}")
                self.host = server_info.host
                self.port = server_info.port
            return True

        return False

    def _initialize_direct_client(self):
        """Initialize direct file-based LanceDB client."""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            from .lancedb_client_adapter import LanceDBClientAdapter
            self.client = LanceDBClientAdapter(uri=self.db_path)
            self.logger.info(f"Direct LanceDB client initialized: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize direct client: {e}")
            raise

    def _initialize_api_client(self, start_server: bool = True):
        num_tries = 4
        trial = 1
        pid = os.getpid()

        while trial <= num_tries:
            nap_time = random.randint(10, 500) / 1000
            self.logger.info(f"Process id {pid} sleeping for {nap_time} seconds.")
            time.sleep(nap_time)
            existing_server_info = self.registry.get_server_info()

            if existing_server_info and self.registry.is_server_alive(existing_server_info):
                if existing_server_info.socket_path:
                    self.socket_path = existing_server_info.socket_path
                    self.logger.info(f"Using existing server at {existing_server_info.socket_path}")
                else:
                    self.host = existing_server_info.host
                    self.port = existing_server_info.port
                    self.logger.info(f"Using existing server at {existing_server_info.host}:{existing_server_info.port}")
                break
            else:
                if start_server:
                    if LanceDBServerInstance.creation_in_progress():
                        self.logger.warning(
                            f"Process: {pid}, trial {trial}: Creation in progress...")
                        time.sleep(2)
                        continue
                    else:
                        self.logger.warning(
                            f"Process: {pid}, trial {trial}: No running server found, starting new instance...")
                    try:
                        self.server_instance = LanceDBServerInstance.create_instance(self.db_path, self.socket_path, self.host, self.port)
                        if self.server_instance is None:
                            trial += 1
                            time.sleep(2)
                            continue
                        self.server_instance.start_server(threaded=True)
                        existing_server_info = self.registry.get_server_info()
                        if existing_server_info:
                            if existing_server_info.socket_path:
                                self.socket_path = existing_server_info.socket_path
                                self.logger.info(f"Starting server at {existing_server_info.socket_path}")
                            else:
                                self.host = existing_server_info.host
                                self.port = existing_server_info.port
                                self.logger.info(f"Starting server at {existing_server_info.host}:{existing_server_info.port}")
                        break
                    except Exception as e:
                        self.logger.warning(f"Received an error: {e} in process {pid}, "
                                            f"this is the try number {trial}, {num_tries-trial+1} remaining.")
                        time.sleep(1)
                        trial += 1
                        if trial > num_tries:
                            raise RuntimeError(f"Failed {trial} times: no server running for database {self.db_path} and start_server=False")
                else:
                    if trial >= num_tries:
                        raise RuntimeError(f"No server running for database {self.db_path} and start_server=False")

        self.api_client = MilvusAPIClient(socket_path=self.socket_path, host=self.host, port=self.port)

    def get_server_info(self) -> Optional[ServerInfo]:
        """Get information about the current server."""
        return self.registry.get_server_info()

    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection (table) exists."""
        if self.use_api:
            return self.api_client.has_collection(collection_name)
        else:
            return self.client.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        """List all collections (tables)."""
        if self.use_api:
            return self.api_client.list_collections()
        else:
            return self.client.list_collections()

    def search(self, collection_name: str,
               query_vector: List[float] | numpy.ndarray = None,
               data: List[float] | numpy.ndarray = None,
               limit: int = 10, output_fields: List[str] = None,
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search.

        Args:
            collection_name: Name of the table to search
            query_vector: Query vector for similarity search
            data: Alternative name for query_vector
            limit: Maximum number of results to return
            output_fields: Fields to include in results
            search_params: Additional search parameters

        Returns:
            List of search results
        """
        if output_fields is None:
            output_fields = ["*"]
        if search_params is None:
            search_params = {}
        if data is not None:
            query_vector = data
        if isinstance(query_vector, numpy.ndarray):
            query_vector = query_vector.tolist()
        if self.use_api:
            return self.api_client.search(collection_name=collection_name, query_vector=query_vector, limit=limit,
                                          output_fields=output_fields, search_params=search_params)
        else:
            try:
                results = self.client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=limit,
                    output_fields=output_fields,
                    search_params=search_params
                )

                search_results = Utils.extract_hits(results[0], output_fields)
                return search_results

            except Exception as e:
                self.logger.error(f"Search error: {e}")
                raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.use_api:
            return self.api_client.get_collection_stats(collection_name)
        else:
            return self.client.get_collection_stats(collection_name)

    def close(self):
        """Close the client connection."""
        if self.client:
            self.client.close()
            self.client = None
        if self.server_instance:
            self.server_instance.stop_server()


# Convenience functions
def create_lancedb_server(db_path: str,
                          socket_path: str = None,
                          host: str = "0.0.0.0",
                          port: int = 8766,
                          use_api: bool = None,
                          start_server: bool = True,
                          registry_dir: str = None) -> LanceDBServer:
    """
    Create a LanceDBServer instance with cross-machine support.

    Args:
        db_path: Path to the LanceDB database directory
        socket_path: Unix domain socket path (if provided, uses Unix sockets for faster IPC)
        host: API server host (use "0.0.0.0" to bind to all interfaces, for TCP mode)
        port: API server port (for TCP mode)
        use_api: Whether to use API mode (auto-detect if None)
        start_server: Whether to start server automatically
        registry_dir: Directory for server registry files

    Returns:
        LanceDBServer instance
    """
    return LanceDBServer(db_path, socket_path, host, port, use_api, start_server, registry_dir)


def list_running_servers(registry_dir: str = None) -> List[Dict[str, Any]]:
    """
    List all running LanceDB servers.

    Args:
        registry_dir: Directory to search for server registry files

    Returns:
        List of server information dictionaries
    """
    if registry_dir is None:
        registry_dir = os.path.join(os.getcwd(), '.lancedb_servers')

    if not os.path.exists(registry_dir):
        return []

    servers = []
    import json
    for filename in os.listdir(registry_dir):
        if filename.startswith('server_') and filename.endswith('.json'):
            try:
                filepath = os.path.join(registry_dir, filename)
                with open(filepath, 'r') as f:
                    server_data = json.load(f)

                registry = ServerRegistry(server_data['db_path'], registry_dir)
                server_info = ServerInfo.from_dict(server_data)
                if registry.is_server_alive(server_info):
                    servers.append(server_data)
                else:
                    registry._cleanup_registry()
            except Exception:
                continue

    return servers


# Example usage and testing
if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Run the LanceDB server discovery example using ./test_db/lancedb. "
                    "For a standalone gRPC server, use wiki-retriever serve-grpc."
    ).parse_args()

    print("Creating first instance (server mode)...")
    server1 = create_lancedb_server("./test_db/lancedb", host="0.0.0.0", port=8766, use_api=True)

    print("Server info:", server1.get_server_info())
    print("Collections:", server1.list_collections())

    print("\nCreating second instance (client mode)...")
    server2 = create_lancedb_server("./test_db/lancedb", use_api=True)

    print("Collections via API:", server2.list_collections())

    print("\nRunning servers:")
    for server_info in list_running_servers("./test_db/.lancedb_servers"):
        print(f"  {server_info['host']}:{server_info['port']} - {server_info['db_path']}")

    server1.close()
    server2.close()
