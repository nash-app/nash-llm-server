import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPHandler:
    def __init__(self):
        self.session = None
        self.client = None
        self._initialized = asyncio.Event()
        
        # Get MCP path from environment
        self.nash_path = os.getenv('NASH_PATH')
        if not self.nash_path:
            raise ValueError(
                "NASH_PATH environment variable not set. "
                "Please set it to the nash-mcp repository root."
            )
            
        self.server_params = StdioServerParameters(
            command=os.path.join(self.nash_path, ".venv/bin/mcp"),
            args=["run", os.path.join(
                self.nash_path, "src/nash_mcp/server.py"
            )],
            env=None
        )
    
    async def _maintain_session(self, read, write):
        """Initialize MCP session."""
        self.session = ClientSession(read, write)
        await self.session.initialize()
        self._initialized.set()
        # No need for while loop - just keep session alive
        try:
            await asyncio.Future()  # Keep session alive until cancelled
        finally:
            self._initialized.clear()
            self.session = None
    
    async def initialize(self):
        """Initialize MCP client session."""
        self.client = stdio_client(self.server_params)
        async with self.client as (read, write):
            self.session = ClientSession(read, write)
            await self.session.initialize()
            self._initialized.set()
            # Keep the context manager alive
            while True:
                await asyncio.sleep(1)
    
    async def close(self):
        """Close MCP client session and cleanup."""
        if self.session:
            self.session = None
        if self.client:
            self.client = None
        self._initialized.clear()
    
    async def execute_method(self, method: str, **args):
        """Execute an MCP method with arguments.
        
        Args:
            method: Name of MCP method to call
            **args: Arguments to pass to the method
            
        Returns:
            Result from MCP method call
        
        Raises:
            ValueError: If session not initialized or method doesn't exist
            TimeoutError: If initialization times out
        """
        # Wait for initialization
        if not await self._initialized.wait():
            raise TimeoutError("MCP session initialization timed out")
            
        if not self.session:
            raise ValueError("MCP session not initialized")
            
        if not hasattr(self.session, method):
            raise ValueError(f"Method '{method}' not found on MCP client")
            
        client_method = getattr(self.session, method)
        return await client_method(**args) 
