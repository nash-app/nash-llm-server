import asyncio
import json
from typing import Dict, List, Optional, Any

from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt


def get_api_config() -> Dict[str, str]:
    """Get API configuration from user input."""
    print("\nChoose API provider:")
    print("1. OpenAI")
    print("2. Anthropic")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            provider = 'OpenAI' if choice == '1' else 'Anthropic'
            model = 'o3-mini' if choice == '1' else 'claude-3-7-sonnet-latest'
            api_key = input(f"\nEnter your {provider} API key: ").strip()
            
            if api_key:
                return {'api_key': api_key, 'model': model}
        print("Invalid choice or empty API key. Please try again.")


async def process_llm_chunk(
    chunk: str,
    request_id: int,
    assistant_message: str
) -> tuple[str, Optional[str]]:
    """Process a chunk from the LLM stream.
    
    Returns:
        Tuple of (updated assistant message, error message if any)
    """
    if "error" in chunk:
        return assistant_message, f"\nError: {chunk}"
        
    if "[DONE]" in chunk:
        return assistant_message, None
        
    try:
        parsed = json.loads(chunk.replace("data: ", ""))
        
        # Verify request ID if present
        if "request_id" in parsed:
            chunk_request_id = parsed["request_id"]
            if chunk_request_id != str(request_id):
                print(
                    f"\n⚠️  Request ID mismatch: expected {request_id}, "
                    f"got {chunk_request_id}"
                )
        
        content = parsed.get("content", "")
        print(content, end="", flush=True)
        return assistant_message + content, None
        
    except json.JSONDecodeError as e:
        return assistant_message, f"\nFailed to parse chunk: {chunk}\nError: {e}"


async def execute_tool_call(
    call_data: Dict[str, Any],
    mcp: MCPHandler,
    messages: List[Dict[str, Any]],
    request_id: int,
    model: str
) -> tuple[int, str]:
    """Execute a tool call and process its result.
    
    Returns:
        Tuple of (new request_id, updated assistant message)
    """
    # Parse function call data
    if isinstance(call_data, str):
        try:
            call_data = json.loads(call_data)
        except json.JSONDecodeError:
            return request_id, ""
    
    function = call_data.get("function", {})
    tool_name = function.get("name")
    arguments = function.get("arguments", {})
    
    if not tool_name:
        return request_id, ""
        
    print(f"\n=== Making MCP Tool Call: {tool_name} ===")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    # Execute tool
    tool_result = await mcp.call_tool(tool_name, arguments=arguments)
    
    # Add tool result to message history
    tool_message = {
        "role": "assistant",
        "content": str(tool_result)
    }
    messages.append(tool_message)
    
    print(f"\nAssistant: Tool result from {tool_name}:")
    print(str(tool_result))
    
    # Get LLM's response to tool result
    request_id += 1
    print(f"\n=== Making LLM Request (ID: {request_id}) ===")
    print("\nMessages being sent to LLM:")
    for idx, msg in enumerate(messages, 1):
        print(f"\n{idx}. Role: {msg['role']}")
        print(f"Content: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"Content: {msg['content']}")
    print("\nGetting response from LLM...")
    
    assistant_message = ""
    async for chunk in stream_llm_response(
        messages,
        model=model,
        request_id=str(request_id)
    ):
        new_message, error = await process_llm_chunk(
            chunk, request_id, assistant_message
        )
        if error:
            print(error)
            break
        assistant_message = new_message
            
    return request_id, assistant_message


async def process_function_call(
    assistant_message: str,
    messages: List[Dict[str, Any]],
    mcp: MCPHandler,
    request_id: int,
    model: str
) -> tuple[int, str]:
    """Process function call in the assistant's message.
    
    Returns:
        Tuple of (new request_id, final assistant message)
    """
    if "<function_call>" not in assistant_message:
        return request_id, assistant_message
        
    start_tag = "<function_call>"
    end_tag = "</function_call>"
    start_idx = assistant_message.find(start_tag) + len(start_tag)
    end_idx = assistant_message.find(end_tag)
    
    if start_idx <= -1 or end_idx <= -1:
        return request_id, assistant_message
        
    try:
        json_str = assistant_message[start_idx:end_idx].strip()
        call_data = json.loads(json_str)
        
        # Since we know there's only one function call, process it directly
        function = call_data.get("function", {})
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        
        if not tool_name:
            return request_id, assistant_message
            
        print(f"\n=== Making MCP Tool Call: {tool_name} ===")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        # Execute tool
        tool_result = await mcp.call_tool(tool_name, arguments=arguments)
        
        # Add tool result to message history
        tool_message = {
            "role": "assistant",
            "content": str(tool_result)
        }
        messages.append(tool_message)
        
        print(f"\nAssistant: Tool result from {tool_name}:")
        print(str(tool_result))
        
        # Get LLM's response to tool result
        request_id += 1
        print(f"\n=== Making LLM Request (ID: {request_id}) ===")
        print("\nMessages being sent to LLM:")
        for idx, msg in enumerate(messages, 1):
            print(f"\n{idx}. Role: {msg['role']}")
            print(f"Content: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"Content: {msg['content']}")
        print("\nGetting response from LLM...")
        
        assistant_message = ""
        async for chunk in stream_llm_response(
            messages,
            model=model,
            request_id=str(request_id)
        ):
            new_message, error = await process_llm_chunk(
                chunk, request_id, assistant_message
            )
            if error:
                print(error)
                break
            assistant_message = new_message
            
    except json.JSONDecodeError as e:
        print(f"\nError parsing function data: {e}")
    except Exception as e:
        print(f"\nError executing function: {e}")
        
    return request_id, assistant_message


async def chat():
    """Main chat loop."""
    # Initialize
    config = get_api_config()
    configure_llm(api_key=config['api_key'])
    
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    
    messages = [{
        "role": "system",
        "content": get_system_prompt(await mcp.list_tools())
    }]
    request_id = 0
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            messages.append({"role": "user", "content": user_input})
            
            # Get initial LLM response
            print("\nAssistant: ", end="", flush=True)
            assistant_message = ""
            request_id += 1
            
            print(f"\n=== Making LLM Request (ID: {request_id}) ===")
            print("\nMessages being sent to LLM:")
            for idx, msg in enumerate(messages, 1):
                print(f"\n{idx}. Role: {msg['role']}")
                print(f"Content: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"Content: {msg['content']}")
            print("\nGetting response from LLM...")
            async for chunk in stream_llm_response(
                messages,
                model=config['model'],
                request_id=str(request_id)
            ):
                new_message, error = await process_llm_chunk(
                    chunk, request_id, assistant_message
                )
                if error:
                    print(error)
                    break
                assistant_message = new_message
            
            # Process any function calls
            if assistant_message:
                # Add the assistant's response to messages before processing function call
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                request_id, assistant_message = await process_function_call(
                    assistant_message, messages, mcp,
                    request_id, config['model']
                )
                # Update the last assistant message with the final response
                messages[-1]["content"] = assistant_message
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    finally:
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
