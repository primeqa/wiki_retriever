"""
gRPC client implementation for Milvus operations.

This module provides a gRPC client that connects to a MilvusGRPCServer
to perform remote Milvus database operations.
"""

import grpc
import logging
from typing import List, Dict, Any, Optional

# Import generated protobuf classes
from . import milvus_service_pb2
from . import milvus_service_pb2_grpc


class MilvusGRPCClient:
    """Client for connecting to MilvusGRPCServer."""

    def __init__(self, host: str = "localhost", port: int = 8765,
                 socket_path: Optional[str] = None, timeout: int = 30):
        """
        Initialize the gRPC client.

        Args:
            host: Server host address (for TCP mode)
            port: Server port (for TCP mode)
            socket_path: Unix domain socket path (if provided, uses Unix socket instead of TCP)
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Determine connection address
        if socket_path:
            self.address = f'unix://{socket_path}'
            self.logger.info(f"Connecting to gRPC server via Unix socket: {socket_path}")
        else:
            self.address = f'{host}:{port}'
            self.logger.info(f"Connecting to gRPC server at {self.address}")

        # Create channel and stub
        self.channel = grpc.insecure_channel(self.address)
        self.stub = milvus_service_pb2_grpc.MilvusServiceStub(self.channel)

        self.logger.info("gRPC client initialized")

    def search(self, collection_name: str, query_vector: List[float],
               limit: int = 10, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            output_fields: Fields to include in results

        Returns:
            List of search results as dictionaries
        """
        try:
            # Create request
            request = milvus_service_pb2.SearchRequest(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                output_fields=output_fields or ["*"]
            )

            # Make gRPC call
            response = self.stub.Search(request, timeout=self.timeout)

            # Convert response to list of dicts
            results = []
            for result in response.results:
                result_dict = {
                    'id': result.id,
                    'distance': result.distance,
                }
                # Add entity fields
                for key, value in result.entity.items():
                    result_dict[key] = value

                results.append(result_dict)

            self.logger.debug(f"Search returned {len(results)} results")
            return results

        except grpc.RpcError as e:
            self.logger.error(f"gRPC search error: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            request = milvus_service_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=self.timeout)
            return response.healthy
        except grpc.RpcError as e:
            self.logger.error(f"Health check RPC error: {e.code()}")
            return False
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        List all collections in the database.

        Returns:
            List of collection names
        """
        try:
            request = milvus_service_pb2.ListCollectionsRequest()
            response = self.stub.ListCollections(request, timeout=self.timeout)
            return list(response.collections)
        except grpc.RpcError as e:
            self.logger.error(f"gRPC list collections error: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            self.logger.error(f"List collections error: {e}")
            raise

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            request = milvus_service_pb2.HasCollectionRequest(
                collection_name=collection_name
            )
            response = self.stub.HasCollection(request, timeout=self.timeout)
            return response.exists
        except grpc.RpcError as e:
            self.logger.error(f"gRPC has collection error: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            self.logger.error(f"Has collection error: {e}")
            raise

    def close(self):
        """Close the gRPC channel."""
        try:
            self.channel.close()
            self.logger.info("gRPC client closed")
        except Exception as e:
            self.logger.warning(f"Error closing gRPC client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Example usage of the gRPC client."""
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error("Usage: python milvus_grpc_client.py <host:port> <collection_name>")
        sys.exit(1)

    # Parse address
    address_parts = sys.argv[1].split(':')
    host = address_parts[0]
    port = int(address_parts[1]) if len(address_parts) > 1 else 8765
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info(f"Connecting to gRPC server at {host}:{port}")

    with MilvusGRPCClient(host=host, port=port) as client:
        # Health check
        logger.info("Performing health check...")
        healthy = client.health_check()
        logger.info(f"Server health: {'OK' if healthy else 'FAILED'}")

        if not healthy:
            logger.error("Server is not healthy, exiting")
            sys.exit(1)

        # List collections
        logger.info("Listing collections...")
        collections = client.list_collections()
        logger.info(f"Collections: {collections}")

        # Check if specific collection exists
        if collection_name:
            logger.info(f"Checking if collection '{collection_name}' exists...")
            exists = client.has_collection(collection_name)
            logger.info(f"Collection exists: {exists}")

            if exists:
                # Example search (using dummy vector)
                logger.info("Performing example search...")
                dummy_vector = [0.1] * 768  # Adjust dimension as needed
                results = client.search(
                    collection_name=collection_name,
                    query_vector=dummy_vector,
                    limit=5
                )
                logger.info(f"Search returned {len(results)} results")
                for i, result in enumerate(results[:3]):
                    logger.info(f"  Result {i+1}: id={result['id']}, distance={result['distance']}")


if __name__ == "__main__":
    main()
