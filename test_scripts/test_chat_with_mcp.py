import asyncio
import json

from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt


def get_api_config():
    """Get API configuration from user input."""
    print("\nChoose API provider:")
    print("1. OpenAI")
    print("2. Anthropic")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            if choice == '1':
                provider = 'OpenAI'
                model = 'o3-mini'
            else:
                provider = 'Anthropic'
                model = 'claude-3-7-sonnet-latest'
                
            api_key = input(f"\nEnter your {provider} API key: ").strip()
            if api_key:
                return {
                    'api_key': api_key,
                    'model': model
                }
        print("Invalid choice or empty API key. Please try again.")


async def chat():
    # Get API configuration from user
    config = get_api_config()
    configure_llm(api_key=config['api_key'])
    
    messages = []
    request_id = 0  # Initialize request counter
    
    # Initialize MCP and get tool definitions
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    tools = await mcp.list_tools()

    system_prompt = get_system_prompt(tools)

    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
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
            request_id += 1  # Increment request counter
            print(f"\n=== Making LLM Request (ID: {request_id}) ===")
            
            async for chunk in stream_llm_response(
                messages,
                model=config['model'],
                request_id=str(request_id)
            ):
                if "error" in chunk:
                    print("\nError:", chunk)
                    break
                    
                if "[DONE]" not in chunk:
                    try:
                        parsed = json.loads(chunk.replace("data: ", ""))
                        if "request_id" in parsed:
                            chunk_request_id = parsed["request_id"]
                            if chunk_request_id != str(request_id):
                                print(
                                    f"\n⚠️  Request ID mismatch: "
                                    f"expected {request_id}, "
                                    f"got {chunk_request_id}"
                                )
                        content = parsed.get("content", "")
                        print(content, end="", flush=True)
                        assistant_message += content
                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse chunk: {chunk}")
                        print(f"Error: {e}")
            
            # Add assistant response to history and check for function calls
            if assistant_message:
                # Check for function call tag
                if "<function_call>" in assistant_message:
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
                            
                            # Execute each function call
                            for call in function_calls:
                                # Parse function call data
                                if isinstance(call, str):
                                    # If the call is a string, try to parse it as JSON
                                    try:
                                        call = json.loads(call)
                                    except json.JSONDecodeError:
                                        # If it's not JSON, skip this call
                                        continue
                                
                                function = call.get("function", {})
                                tool_name = function.get("name")
                                arguments = function.get("arguments", {})
                                
                                if tool_name:
                                    print(f"\n=== Making MCP Tool Call: {tool_name} ===")
                                    print(f"Arguments: {json.dumps(arguments, indent=2)}")
                                    
                                    tool_result = await mcp.call_tool(
                                        tool_name,
                                        arguments=arguments
                                    )
                                    
                                    # Add tool result to message history using Claude format
                                    tool_message = {
                                        "role": "assistant",
                                        "content": str(tool_result),  # Use the string result directly
                                        "tool_calls": [{
                                            "id": str(request_id),
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": json.dumps(
                                                    arguments
                                                )
                                            }
                                        }]
                                    }
                                    messages.append(tool_message)
                                    
                                    print("\nAssistant: ", end="", flush=True)
                                    print(f"Tool result from {tool_name}:")
                                    print(str(tool_result))
                                    
                                    # Get LLM's response to the tool result
                                    request_id += 1
                                    print(f"\n=== Making LLM Request (ID: {request_id}) ===")
                                    print("Getting response to tool result...")
                                    
                                    async for chunk in stream_llm_response(
                                        messages,
                                        model=config['model'],
                                        request_id=str(request_id)
                                    ):
                                        if "error" in chunk:
                                            print("\nError:", chunk)
                                            break
                                            
                                        if "[DONE]" not in chunk:
                                            try:
                                                parsed = json.loads(
                                                    chunk.replace("data: ", "")
                                                )
                                                if "request_id" in parsed:
                                                    chunk_request_id = (
                                                        parsed["request_id"]
                                                    )
                                                    if chunk_request_id != str(
                                                        request_id
                                                    ):
                                                        print(
                                                            f"\n⚠️  Request ID "
                                                            f"mismatch: expected "
                                                            f"{request_id}, got "
                                                            f"{chunk_request_id}"
                                                        )
                                                content = parsed.get(
                                                    "content", ""
                                                )
                                                print(
                                                    content, 
                                                    end="", 
                                                    flush=True
                                                )
                                                assistant_message += content
                                            except json.JSONDecodeError as e:
                                                print(
                                                    f"\nError parsing chunk: {e}"
                                                )
                        except json.JSONDecodeError as e:
                            print(f"\nError parsing function data: {e}")
                        except Exception as e:
                            print(f"\nError executing function: {e}")
                        
                messages.append({
                    "role": "assistant", 
                    "content": assistant_message
                })
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    finally:
        # Clean up MCP
        await mcp.close()
    
    print("\nChat ended.")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
        asyncio.run(MCPHandler.get_instance().close())
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        asyncio.run(MCPHandler.get_instance().close())
