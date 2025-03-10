import asyncio
import os
from dotenv import load_dotenv
from app.mcp_handler import MCPHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Need to load environment variables before MCPHandler is instantiated
if not os.getenv('NASH_PATH'):
    load_dotenv()


async def test_mcp():
    mcp = MCPHandler()
    
    try:
        # Start initialization in background
        init_task = asyncio.create_task(mcp.initialize())
        
        # Give it a moment to initialize
        await asyncio.sleep(1)
        
        print("\nListing available MCP tools...")
        result = await mcp.execute_method("list_tools")
        print("Available tools:", result)
        
    except Exception as e:
        print(f"Error during MCP testing: {e}")
    finally:
        if 'init_task' in locals():
            init_task.cancel()
            try:
                await init_task
            except asyncio.CancelledError:
                pass
        await mcp.close()


async def test_basic_mcp():
    print("Starting basic MCP test...")
    
    # Setup server parameters
    nash_path = os.getenv('NASH_PATH')
    if not nash_path:
        raise ValueError("NASH_PATH environment variable not set")
        
    server_params = StdioServerParameters(
        command=os.path.join(nash_path, ".venv/bin/mcp"),
        args=["run", os.path.join(nash_path, "src/nash_mcp/server.py")],
        env=None
    )
    
    print("Connecting to MCP server...")
    client = stdio_client(server_params)
    
    async with client as (read, write):
        print("Creating session...")
        session = ClientSession(read, write)
        print("Initializing session...")
        await session.initialize()
        
        print("\nListing tools...")
        tools = await session.list_tools()
        print(f"Available tools: {tools}")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_mcp())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
