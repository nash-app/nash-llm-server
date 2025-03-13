import json


async def process_tool_call(message_text, mcp):
    """
    Process an assistant's message to identify and execute tool calls.
    
    Args:
        message_text (str): The assistant's message text
        mcp: MCPHandler instance
        
    Returns:
        dict: {
            'tool_call_made': bool,
            'tool_name': str or None,
            'arguments': dict or None,
            'result': object or None,
            'formatted_result': str or None
        }
    """
    # Check if the message contains a function call
    if "<tool_call>" not in message_text:
        return {
            'tool_call_made': False,
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None
        }
    
    # Extract the function call JSON
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = message_text.find(start_tag) + len(start_tag)
    end_idx = message_text.find(end_tag)
    
    if start_idx <= 0 or end_idx <= 0:
        return {
            'tool_call_made': False,
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None
        }
    
    try:
        # Parse the JSON
        json_str = message_text[start_idx:end_idx].strip()
        function_call = json.loads(json_str)
        
        # Get function details
        if isinstance(function_call, list):
            # Handle the first function call in the list
            call = function_call[0]
            function = call.get("function", {})
        else:
            # Direct function call object
            function = function_call.get("function", {})
        
        if not isinstance(function, dict):
            raise ValueError(f"Expected function to be a dict, got: {type(function)}")
        
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        
        if not tool_name:
            return {
                'tool_call_made': False,
                'tool_name': None,
                'arguments': None,
                'result': None,
                'formatted_result': None
            }
        
        # Execute the tool
        tool_result = await mcp.call_tool(tool_name, arguments=arguments)
        
        # Extract the text content from the tool result
        result_text = ""
        
        # Try to extract text from the content field if available
        if hasattr(tool_result, 'content') and tool_result.content:
            # If it's a list of content items
            if isinstance(tool_result.content, list):
                for content_item in tool_result.content:
                    if hasattr(content_item, 'text'):
                        result_text += content_item.text
            # If it's a single content item
            elif hasattr(tool_result.content, 'text'):
                result_text = tool_result.content.text
        
        # If we couldn't extract text content, fall back to string representation
        if not result_text:
            result_text = str(tool_result)
        
        # Format the result
        is_error = hasattr(tool_result, 'isError') and tool_result.isError
        if is_error:
            formatted_result = f"<tool_results>\n<e>{result_text}</e>\n</tool_results>"
        else:
            formatted_result = f"<tool_results>\n{result_text}\n</tool_results>"
        
        return {
            'tool_call_made': True,
            'tool_name': tool_name,
            'arguments': arguments,
            'result': tool_result,
            'formatted_result': formatted_result,
            'is_error': is_error
        }
        
    except json.JSONDecodeError as e:
        return {
            'tool_call_made': False,
            'error': f"Error parsing function data: {e}",
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None
        }
    except Exception as e:
        return {
            'tool_call_made': False,
            'error': f"Error executing function: {e}",
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None
        }
