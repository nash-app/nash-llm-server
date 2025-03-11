import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompt_generator import generate_tool_system_prompt


def convert_tools_to_dict(tools_result):
    """Convert MCP tools result to JSON-serializable format."""
    tools = []
    for tool in tools_result.tools:
        # Print tool for debugging
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        })
    return {"tools": tools}


async def chat():
    configure_llm()
    messages = []
    
    # Initialize MCP and get tool definitions
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    tools = await mcp.list_tools()
    
    # Convert tools to JSON-serializable format
    tools_dict = convert_tools_to_dict(tools)
    
    # Generate system prompt with tool definitions
    system_prompt = generate_tool_system_prompt(
        tool_definitions=json.dumps(tools_dict, indent=2)
    )
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    print(system_prompt)
    
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
                    try:
                        parsed = json.loads(chunk.replace("data: ", ""))
                        content = parsed.get("content", "")
                        print(f"\nDEBUG Raw chunk: {chunk}")
                        print(f"DEBUG Parsed content: {content}")
                        print(content, end="", flush=True)
                        assistant_message += content
                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse chunk: {chunk}")
                        print(f"Error: {e}")
                        
            print("\nDEBUG Full assistant message:", assistant_message)
            
            # Add assistant response to history and check for TOOL CALL
            if assistant_message:
                # Check for function call tag
                if "<function_call>" in assistant_message:
                    print("\nFUNCTION CALL DETECTED")
                    # Extract content between function call tags
                    start_tag = "<function_call>"
                    end_tag = "</function_call>"
                    start_idx = (
                        assistant_message.find(start_tag) + len(start_tag)
                    )
                    end_idx = assistant_message.find(end_tag)
                    
                    if start_idx > -1 and end_idx > -1:
                        json_str = assistant_message[start_idx:end_idx].strip()
                        try:
                            function_calls = json.loads(json_str)
                            print("Function calls:", function_calls)
                            
                            # Execute each function call
                            for call in function_calls:
                                function = call.get("function", {})
                                tool_name = function.get("name")
                                arguments = function.get("arguments", {})
                                
                                if tool_name:
                                    print(
                                        f"\nExecuting {tool_name} with args:",
                                        arguments
                                    )
                                    tool_result = await mcp.call_tool(
                                        tool_name,
                                        arguments=arguments
                                    )
                                    print("\nTool result:", tool_result)
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse function data: {e}")
                        except Exception as e:
                            print(f"Function call failed: {e}")
                        
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
