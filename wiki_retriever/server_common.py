"""
Shared server infrastructure: registry/discovery, server info, and the gRPC API client.

Extracted from milvus_server_management.py so the LanceDB path does not depend
on pymilvus or flask.
"""

import os
import json
import time
import socket
import hashlib
import fcntl
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager


class Utils:
    @staticmethod
    def extract_hits(results, output_fields: List[str] = None) -> List[Dict]:
        search_results = []
        for hit in results:
            hit_dict = {
                'id': hit.get('id'),
                'distance': hit.get('distance')
            }
            # Add output fields to the result
            if output_fields:
                vals = hit['entity']
                for field in output_fields:
                    if field in vals:
                        hit_dict[field] = vals[field]
            search_results.append(hit_dict)

        return search_results


@dataclass
class ServerInfo:
    """Information about a running server"""
    pid: int
    db_path: str
    start_time: float
    socket_path: str = None  # Unix domain socket path
    host: str = None  # TCP host (legacy)
    port: int = None  # TCP port (legacy)
    machine_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'pid': self.pid,
            'db_path': self.db_path,
            'start_time': self.start_time,
            'machine_id': self.machine_id,
            'socket_path': self.socket_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServerInfo':
        return cls(**data)


class ServerRegistry:
    """Manages server registration and discovery across machines"""
    
    def __init__(self, db_path: str, registry_dir: str = None):
        self.db_path = os.path.abspath(db_path)
        
        # Create a unique identifier for this database
        # self.db_hash = hashlib.md5(self.db_path.encode()).hexdigest()[:8]
        self.db_hash = hashlib.sha256(self.db_path.encode()).hexdigest()[:16]

        # Default registry directory
        if registry_dir is None:
            registry_dir = os.path.join(os.path.dirname(self.db_path), '.milvus_servers')
        
        self.registry_dir = os.path.abspath(registry_dir)
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Registry file for this specific database
        self.registry_file = os.path.join(self.registry_dir, f"server_{self.db_hash}.json")
        self.lock_file = f"{self.registry_file}.lock"
        
        self.logger = logging.getLogger(__name__)
        self.lock_fd = None
        self._acquire_lock()

    def _get_machine_id(self) -> str:
        """Get a unique identifier for this machine"""
        try:
            # Try to get machine ID from various sources
            # if os.path.exists('/etc/machine-id'):
            #     with open('/etc/machine-id', 'r') as f:
            #         return f.read().strip()
            # elif os.path.exists('/var/lib/dbus/machine-id'):
            #     with open('/var/lib/dbus/machine-id', 'r') as f:
            #         return f.read().strip()
            # else:
                # Fallback: use hostname + MAC address
            import uuid
            mac = hex(uuid.getnode())
            hostname = socket.gethostname()
            return hashlib.md5(f"{hostname}:{mac}".encode()).hexdigest()
        except Exception:
            # Last resort: just use hostname
            return socket.gethostname()

    @contextmanager
    def _acquire_lock(self):
        """Acquire file lock as context manager"""
        try:
            self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            yield
        except OSError as e:
            self.logger.error(f"Failed to acquire file lock: {e}")
            raise
        finally:
            if self.lock_fd is not None:
                try:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                    os.close(self.lock_fd)
                except OSError as e:
                    self.logger.warning(f"Failed to release file lock: {e}")

    def __del__(self):
        """Release file lock on destruction"""
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
            except OSError as e:
                self.logger.warning(f"Failed to release file lock: {e}")

    def register_server(self, pid: int, socket_path: str = None, host: str = None, port: int = None) -> ServerInfo:
        """Register a server instance (supports both Unix socket and TCP)"""
        server_info = ServerInfo(
            pid=pid,
            db_path=self.db_path,
            start_time=time.time(),
            machine_id=self._get_machine_id(),
            socket_path=socket_path,
            host=host,
            port=port
        )

        with open(self.registry_file, 'w') as f:
            json.dump(server_info.to_dict(), f, indent=2)

        if socket_path:
            self.logger.info(f"Registered server at {socket_path} for database {self.db_path}")
        else:
            self.logger.info(f"Registered server {host}:{port} for database {self.db_path}")
        return server_info
    
    def get_server_info(self) -> Optional[ServerInfo]:
        """Get information about the registered server"""
        if not os.path.exists(self.registry_file):
            return None
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            return ServerInfo.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Registry file is corrupted or missing, clean it up
            self._cleanup_registry()
            return None
    
    def is_server_alive(self, server_info: ServerInfo) -> bool:
        """Check if a registered server is still alive"""
        if server_info is None:
            return False

        try:
            is_same_machine = server_info.machine_id == self._get_machine_id()

            # For same-machine servers, first check if process exists
            if is_same_machine:
                try:
                    os.kill(server_info.pid, 0)  # Check if process exists
                except (OSError, ProcessLookupError):
                    # Process doesn't exist locally, server is dead
                    return False

            # Now verify server responds (for both local and remote servers)
            if server_info.socket_path:
                # Unix socket: check if socket file exists and is connectable
                if not os.path.exists(server_info.socket_path):
                    return False
                try:
                    import socket as sock
                    s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect(server_info.socket_path)
                    s.close()
                    return True
                except Exception:
                    return False
            else:
                # TCP: use gRPC health check (works for both local and remote)
                # from milvus_grpc_client import MilvusGRPCClient
                from .milvus_grpc_client import MilvusGRPCClient
                try:
                    client = MilvusGRPCClient(server_info.host, server_info.port)
                    result = client.health_check()
                    client.close()
                    return result
                except Exception as e:
                    # If this is a remote server, log more details about the failure
                    if not is_same_machine:
                        self.logger.debug(f"Remote server health check failed for {server_info.host}:{server_info.port}: {e}")
                    return False

        except Exception as e:
            self.logger.debug(f"Server health check failed: {e}")
            return False
    
    def cleanup_if_dead(self) -> bool:
        """Clean up registry if server is dead, returns True if cleaned up"""
        server_info = self.get_server_info()
        if server_info and not self.is_server_alive(server_info):
            self._cleanup_registry()
            return True
        return False
    
    def _cleanup_registry(self):
        """Remove the registry file"""
        try:
            if os.path.exists(self.registry_file):
                os.remove(self.registry_file)
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception:
            pass
    
    def unregister_server(self):
        """Unregister the current server"""
        self._cleanup_registry()



class MilvusAPIClient:
    """gRPC client for communicating with MilvusServer across machines"""

    def __init__(self, socket_path: str = None, host: str = "127.0.0.1", port: int = 8765):
        from .milvus_grpc_client import MilvusGRPCClient

        self.socket_path = socket_path
        self.host = host
        self.port = port

        # gRPC client (supports both Unix socket and TCP)
        self.grpc_client = MilvusGRPCClient(host=host, port=port, socket_path=socket_path)
        self.logger = logging.getLogger(__name__)

    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return self.grpc_client.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.grpc_client.list_collections()

    def search(self, collection_name: str,
               data: List[float]|None=None,
               query_vector: List[float] = None,
               limit: int = 10, output_fields: List[str] = None,
               search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector search via gRPC"""

        # Handle query_vector vs data parameter
        if query_vector is not None:
            data = query_vector

        if data is None:
            raise ValueError("Either 'data' or 'query_vector' must be provided")

        # Call gRPC client with error handling for missing collections
        try:
            return self.grpc_client.search(
                collection_name=collection_name,
                query_vector=data,
                limit=limit,
                output_fields=output_fields or ["*"]
            )
        except Exception as e:
            # Check if this is a collection not found error
            error_msg = str(e)
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                # Get available collections
                try:
                    available_collections = self.grpc_client.list_collections()
                    # ANSI color codes: green for collection names
                    colored_collections = ', '.join([f"\033[92m{col}\033[0m" for col in available_collections]) if available_collections else 'none'
                    self.logger.error(f"Collection '{collection_name}' not found. Available collections: {colored_collections}")
                except Exception:
                    pass  # If we can't list collections, just re-raise original error
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        # Note: This is not implemented in the gRPC service yet
        raise NotImplementedError("Collection stats not yet supported via gRPC")

    def close(self):
        """Close the gRPC client"""
        self.grpc_client.close()
