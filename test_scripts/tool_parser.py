import json
from typing import Dict, Any


def parse_tool_call(message_text: str) -> Dict[str, Any]:
    """
    Parse a message to extract tool call information.
    
    Args:
        message_text (str): The message text that might contain a tool call
        
    Returns:
        dict: {
            'tool_call_found': bool,
            'tool_name': str or None,
            'arguments': dict or None,
            'error': str or None
        }
    """
    # Check if the message contains a tool call
    if "<tool_call>" not in message_text:
        return {
            'tool_call_found': False,
            'tool_name': None,
            'arguments': None,
            'error': None
        }
    
    # Extract the tool call JSON
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = message_text.find(start_tag) + len(start_tag)
    end_idx = message_text.find(end_tag)
    
    if start_idx <= 0 or end_idx <= 0:
        return {
            'tool_call_found': False,
            'tool_name': None,
            'arguments': None,
            'error': "Tool call tags found but couldn't extract content"
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
            return {
                'tool_call_found': False,
                'tool_name': None,
                'arguments': None,
                'error': f"Expected function to be a dict, got: {type(function)}"
            }
        
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        
        if not tool_name:
            return {
                'tool_call_found': False,
                'tool_name': None,
                'arguments': None,
                'error': "Tool call found but no tool name specified"
            }
        
        return {
            'tool_call_found': True,
            'tool_name': tool_name,
            'arguments': arguments,
            'error': None
        }
        
    except json.JSONDecodeError as e:
        return {
            'tool_call_found': False,
            'tool_name': None,
            'arguments': None,
            'error': f"Error parsing tool call JSON: {e}"
        }
    except Exception as e:
        return {
            'tool_call_found': False,
            'tool_name': None,
            'arguments': None,
            'error': f"Error processing tool call: {e}"
        }


def format_tool_result(result: Any, is_error: bool = False) -> str:
    """
    Format a tool result for display and inclusion in messages.
    
    Args:
        result: The tool execution result
        is_error: Whether the result represents an error
        
    Returns:
        str: Formatted tool result
    """
    result_text = str(result)
    
    if is_error:
        return f"<tool_results>\n<e>{result_text}</e>\n</tool_results>"
    else:
        return f"<tool_results>\n{result_text}\n</tool_results>"
