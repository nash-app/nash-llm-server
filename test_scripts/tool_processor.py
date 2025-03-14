from .tool_parser import parse_tool_call, format_tool_result


async def process_tool_call(message_text, mcp):
    """
    Process an assistant's message to identify and execute tool calls (async version).
    
    Args:
        message_text (str): The assistant's message text
        mcp: MCPHandler instance for direct async calls
        
    Returns:
        dict: {
            'tool_call_made': bool,
            'tool_name': str or None,
            'arguments': dict or None,
            'result': object or None,
            'formatted_result': str or None
        }
    """
    # Use the shared parser to extract tool information
    parsed = parse_tool_call(message_text)
    
    # If no tool call was found or there was an error, return early
    if not parsed['tool_call_found']:
        return {
            'tool_call_made': False,
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None,
            'error': parsed.get('error')
        }
    
    # Extract the parsed information
    tool_name = parsed['tool_name']
    arguments = parsed['arguments']
    
    try:
        # Execute the tool (async version)
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
        
        # Check if result indicates an error
        is_error = hasattr(tool_result, 'isError') and tool_result.isError
        
        # Use shared formatter for consistent output
        formatted_result = format_tool_result(result_text, is_error)
        
        return {
            'tool_call_made': True,
            'tool_name': tool_name,
            'arguments': arguments,
            'result': tool_result,
            'formatted_result': formatted_result,
            'is_error': is_error
        }
    except Exception as e:
        return {
            'tool_call_made': False,
            'error': f"Error executing tool: {e}",
            'tool_name': None,
            'arguments': None,
            'result': None,
            'formatted_result': None
        }
