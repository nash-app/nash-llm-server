from typing import Optional
import asyncio
from mcp.client import ClientSession
from mcp.transport.stdio import stdio_client
from mcp.client.stdio import StdioServerParameters


class MCPClientSingleton:
    _instance: Optional['MCPClientSingleton'] = None
    _client: Optional[ClientSession] = None
    _read_write = None
    _lock = asyncio.Lock()

    def __init__(self):
        raise RuntimeError("Use get_instance() instead")

    @classmethod
    async def get_instance(cls) -> ClientSession:
        """Get or create the singleton MCP client instance."""
        if not cls._instance:
            async with cls._lock:
                if not cls._instance:
                    cls._instance = object.__new__(cls)
                    
                    # Configure the MCP server parameters
                    server_params = StdioServerParameters(
                        command="nash",  # Command to run the MCP server
                        args=[],  # Optional command line arguments
                        env=None  # Use current environment
                    )
                    
                    # Start the MCP server process and get stdio 
                    # streams
                    cls._read_write = await stdio_client(
                        server_params
                    ).__aenter__()
                    read, write = cls._read_write
                    
                    # Create and initialize the client session
                    cls._client = ClientSession(read, write)
                    await cls._client.initialize()
                    
        return cls._client

    @classmethod
    async def close(cls):
        """Close the MCP client connection if it exists."""
        if cls._client:
            await cls._client.close()
            cls._client = None
            
        if cls._read_write:
            read, write = cls._read_write
            await write.aclose()
            await read.aclose()
            cls._read_write = None
            
        cls._instance = None 