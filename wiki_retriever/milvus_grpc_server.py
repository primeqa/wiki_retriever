"""
gRPC server implementation for Milvus operations.

This module provides a gRPC server that wraps a MilvusClient to enable
remote access to Milvus database operations across processes and machines.
"""

import grpc
from concurrent import futures
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pymilvus import MilvusClient

# Import generated protobuf classes
from . import milvus_service_pb2
from . import milvus_service_pb2_grpc


class MilvusServicer(milvus_service_pb2_grpc.MilvusServiceServicer):
    """Implementation of the MilvusService gRPC service."""

    def __init__(self, client: "MilvusClient"):
        """
        Initialize the servicer with a Milvus client.

        Args:
            client: MilvusClient instance to use for operations
        """
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.logger.info("MilvusServicer initialized")

    def Search(self, request, context):
        """
        Handle search requests.

        Args:
            request: SearchRequest protobuf message
            context: gRPC context

        Returns:
            SearchResponse protobuf message
        """
        try:
            # Extract parameters from request
            collection_name = request.collection_name
            query_vector = list(request.query_vector)
            limit = request.limit
            output_fields = list(request.output_fields) if request.output_fields else ["*"]

            self.logger.debug(f"Search request: collection={collection_name}, limit={limit}")

            # Perform search using MilvusClient
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=limit,
                output_fields=output_fields
            )

            # Convert results to protobuf format
            response = milvus_service_pb2.SearchResponse()

            # Results is a list with one element (since we search with one vector)
            if results and len(results) > 0:
                for hit in results[0]:
                    result = response.results.add()
                    result.id = str(hit.get('id', ''))
                    result.distance = float(hit.get('distance', 0.0))

                    # Add entity fields
                    if 'entity' in hit:
                        for key, value in hit['entity'].items():
                            result.entity[key] = str(value)
                    else:
                        # Fields might be at top level
                        for field in output_fields:
                            if field in hit and field not in ['id', 'distance']:
                                result.entity[field] = str(hit[field])

            self.logger.debug(f"Search returned {len(response.results)} results")
            return response

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Search failed: {str(e)}")
            return milvus_service_pb2.SearchResponse()

    def HealthCheck(self, request, context):
        """
        Handle health check requests.

        Args:
            request: HealthCheckRequest protobuf message
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf message
        """
        try:
            # Try to list collections as a health check
            self.client.list_collections()
            return milvus_service_pb2.HealthCheckResponse(healthy=True)
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return milvus_service_pb2.HealthCheckResponse(healthy=False)

    def ListCollections(self, request, context):
        """
        Handle list collections requests.

        Args:
            request: ListCollectionsRequest protobuf message
            context: gRPC context

        Returns:
            ListCollectionsResponse protobuf message
        """
        try:
            collections = self.client.list_collections()
            return milvus_service_pb2.ListCollectionsResponse(collections=collections)
        except Exception as e:
            self.logger.error(f"List collections error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list collections: {str(e)}")
            return milvus_service_pb2.ListCollectionsResponse()

    def HasCollection(self, request, context):
        """
        Handle has collection requests.

        Args:
            request: HasCollectionRequest protobuf message
            context: gRPC context

        Returns:
            HasCollectionResponse protobuf message
        """
        try:
            exists = self.client.has_collection(request.collection_name)
            return milvus_service_pb2.HasCollectionResponse(exists=exists)
        except Exception as e:
            self.logger.error(f"Has collection error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to check collection: {str(e)}")
            return milvus_service_pb2.HasCollectionResponse(exists=False)


def serve_grpc(client: "MilvusClient", host: str = "0.0.0.0", port: int = 8765,
               max_workers: int = 20, socket_path: Optional[str] = None) -> grpc.Server:
    """
    Start a gRPC server for Milvus operations.

    Args:
        client: MilvusClient instance to use for operations
        host: Host address to bind to (for TCP mode)
        port: Port to listen on (for TCP mode)
        max_workers: Maximum number of worker threads
        socket_path: Unix domain socket path (if provided, uses Unix socket instead of TCP)

    Returns:
        grpc.Server instance
    """
    logger = logging.getLogger(__name__)

    # Create server with thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Add servicer to server
    servicer = MilvusServicer(client)
    milvus_service_pb2_grpc.add_MilvusServiceServicer_to_server(servicer, server)

    # Bind to address
    if socket_path:
        # Unix domain socket mode
        address = f'unix://{socket_path}'
        logger.info(f"Starting gRPC server on Unix socket: {socket_path}")
    else:
        # TCP mode
        address = f'{host}:{port}'
        logger.info(f"Starting gRPC server on {address}")

    if not server.add_insecure_port(address):
        raise RuntimeError(f"Could not bind gRPC server to {address}")

    # Start server
    server.start()
    logger.info(f"gRPC server started successfully on {address}")

    return server


def main():
    """Example usage of the gRPC server."""
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error("Usage: python milvus_grpc_server.py <db_path> [port]")
        sys.exit(1)

    db_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

    logger.info(f"Initializing Milvus client with database: {db_path}")
    from pymilvus import MilvusClient
    client = MilvusClient(uri=db_path)

    logger.info(f"Starting gRPC server on port {port}")
    server = serve_grpc(client, port=port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    main()
