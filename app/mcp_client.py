from typing import Optional
import asyncio
import os
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import anyio


class MCPClientSingleton:
    _instance: Optional['MCPClientSingleton'] = None
    _client: Optional[ClientSession] = None
    _read_write = None
    _lock = asyncio.Lock()
    _initializing = False
    _task_group = None
    _initialized = False

    def __init__(self):
        raise RuntimeError("Use get_instance() instead")

    @classmethod
    async def get_instance(cls) -> ClientSession:
        """Get or create the singleton MCP client instance."""
        if not cls._instance:
            async with cls._lock:
                if not cls._instance and not cls._initializing:
                    try:
                        print("\nStarting MCP client initialization...")
                        cls._initializing = True
                        cls._instance = object.__new__(cls)
                        
                        # Get the nash installation path from environment
                        nash_path = os.getenv('NASH_PATH')
                        if not nash_path:
                            msg = (
                                "NASH_PATH environment variable not set. "
                                "Please set it to the nash-mcp repository root."
                            )
                            raise RuntimeError(msg)
                        
                        print(f"Using NASH_PATH: {nash_path}")
                        nash_path = Path(nash_path)
                        venv_mcp = nash_path / ".venv/bin/mcp"
                        server_script = nash_path / "src/nash_mcp/server.py"
                        
                        if not venv_mcp.exists():
                            msg = (
                                f"MCP CLI not found at {venv_mcp}. "
                                "Please ensure nash-mcp is properly installed."
                            )
                            raise RuntimeError(msg)
                        
                        if not server_script.exists():
                            msg = (
                                f"Server script not found at {server_script}. "
                                "Please ensure nash-mcp source code is available."
                            )
                            raise RuntimeError(msg)
                        
                        print("Found required files, configuring server parameters...")
                        # Configure the MCP server parameters
                        server_params = StdioServerParameters(
                            command=str(venv_mcp),
                            args=["run", str(server_script)],
                            env=None  # Use current environment
                        )
                        
                        print("Creating task group...")
                        # Create a task group for managing the connection
                        cls._task_group = anyio.create_task_group()
                        await cls._task_group.__aenter__()
                        
                        try:
                            print("Starting MCP server process...")
                            # Start the MCP server process and get stdio streams
                            stdio_client_instance = stdio_client(
                                server_params
                            )
                            print("Getting stdio streams...")
                            cls._read_write = await (
                                stdio_client_instance.__aenter__()
                            )
                            read, write = cls._read_write
                            
                            print("Creating client session...")
                            # Create and initialize the client session
                            cls._client = ClientSession(read, write)
                            print("Initializing client session...")
                            await cls._client.initialize()
                            
                            print("Waiting for server to be ready...")
                            # Wait for server to be ready
                            max_retries = 5
                            retry_delay = 1.0
                            last_error = None
                            for attempt in range(max_retries):
                                try:
                                    print(f"Attempt {attempt + 1}/{max_retries} to check server readiness...")
                                    # Try a simple request to check if server is ready
                                    tools = await cls._client.list_tools()
                                    print(f"Successfully retrieved {len(tools)} tools from server")
                                    cls._initialized = True
                                    print("Server is ready!")
                                    break
                                except Exception as e:
                                    last_error = e
                                    print(f"Server not ready yet (attempt {attempt + 1}/{max_retries}): {str(e)}")
                                    if attempt < max_retries - 1:
                                        print(f"Waiting {retry_delay} seconds before next attempt...")
                                        await asyncio.sleep(retry_delay)
                            else:
                                error_msg = f"MCP server failed to initialize after {max_retries} attempts"
                                if last_error:
                                    error_msg += f". Last error: {str(last_error)}"
                                raise RuntimeError(error_msg)
                                
                        except Exception as e:
                            print(f"Error during initialization: {str(e)}")
                            # Clean up stdio client if initialization fails
                            if cls._read_write:
                                read, write = cls._read_write
                                await write.aclose()
                                await read.aclose()
                            raise e
                            
                    except Exception as e:
                        print(f"Error in get_instance: {str(e)}")
                        cls._instance = None
                        if cls._task_group:
                            await cls._task_group.__aexit__(
                                type(e), e, None
                            )
                        raise e
                    finally:
                        cls._initializing = False
                        
        if not cls._initialized:
            raise RuntimeError("MCP client not fully initialized")
            
        return cls._client

    @classmethod
    async def close(cls):
        """Close the MCP client connection if it exists."""
        print("\nClosing MCP client...")
        try:
            if cls._client:
                print("Closing client session...")
                await cls._client.close()
                cls._client = None
                print("Client session closed")
            
            if cls._read_write:
                print("Closing stdio streams...")
                read, write = cls._read_write
                await write.aclose()
                await read.aclose()
                cls._read_write = None
                print("Stdio streams closed")
            
            if cls._task_group:
                print("Closing task group...")
                await cls._task_group.__aexit__(None, None, None)
                cls._task_group = None
                print("Task group closed")
            
            cls._instance = None
            cls._initializing = False
            cls._initialized = False
            print("MCP client closed successfully")
        except Exception as e:
            print(f"Error during MCP client shutdown: {str(e)}")
            raise 