import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler


async def chat():
    configure_llm()
    messages = []
    
    # Initialize MCP
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Stream AI response
            print("\nAssistant: ", end="", flush=True)
            assistant_message = ""
            
            async for chunk in stream_llm_response(messages):
                if "error" in chunk:
                    print("\nError:", chunk)
                    break
                    
                if "[DONE]" not in chunk:
                    content = json.loads(
                        chunk.replace("data: ", "")
                    ).get("content", "")
                    print(content, end="", flush=True)
                    assistant_message += content
            
            # Add assistant response to history and check for TOOL CALL
            if assistant_message:
                # Check for TOOL_CALL at start of message
                if assistant_message.lstrip().startswith("TOOL_CALL"):
                    print("\nTOOL CALL")
                    # Extract and parse JSON after TOOL_CALL
                    msg = assistant_message.lstrip()
                    json_str = msg[len("TOOL_CALL"):].strip()
                    try:
                        tool_data = json.loads(json_str)
                        print("Tool data:", tool_data)
                        
                        # Execute tool call
                        tool_name = tool_data['tool']
                        arguments = tool_data['arguments']
                        tool_result = await mcp.call_tool(tool_name, arguments=arguments)
                        print("\nTool result:", tool_result)
                        
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse tool data: {e}")
                    except KeyError as e:
                        msg = f"Invalid tool data format: missing {e}"
                        print(msg)
                    except Exception as e:
                        print(f"Tool call failed: {e}")
                        
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    finally:
        # Clean up MCP
        await mcp.close()
    
    print("\nChat ended. Final message count:", len(messages))


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
        # Ensure MCP cleanup on keyboard interrupt
        asyncio.run(MCPHandler.get_instance().close())
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        # Ensure MCP cleanup on unexpected errors
        asyncio.run(MCPHandler.get_instance().close())
