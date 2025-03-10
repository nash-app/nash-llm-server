import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()


async def test_basic_mcp():
    # Setup server parameters
    nash_path = os.getenv('NASH_PATH')
    if not nash_path:
        raise ValueError("NASH_PATH environment variable not set")
        
    server_params = StdioServerParameters(
        command=os.path.join(nash_path, ".venv/bin/mcp"),
        args=["run", os.path.join(nash_path, "src/nash_mcp/server.py")],
        env=None
    )
    
    try:
        print("Starting MCP test...")
        async with stdio_client(server_params) as (read, write):
            print("Client connected, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()
                
                print("\nListing tools...")
                tools = await session.list_tools()
                print(f"Available tools: {tools}")
                
                print("\nTesting list_installed_packages...")
                result = await session.call_tool("list_installed_packages")
                print(f"Installed packages: {result}")
                
                # Keep connection alive
                try:
                    while True:
                        await asyncio.sleep(1)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\nReceived shutdown signal, cleaning up...")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutdown requested during startup...")
    except Exception as e:
        print(f"\nError during MCP test: {str(e)}")
    finally:
        print("MCP test complete, goodbye!")


def main():
    try:
        asyncio.run(test_basic_mcp())
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
