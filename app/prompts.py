import json


def convert_tools_to_dict(tools_result):
    """Convert MCP tools result to JSON-serializable format."""
    tools = []
    for tool in tools_result.tools:
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        })
    return {"tools": tools}


def format_instructions():
    return """When a user's request requires using a tool, you MUST format your response like this:

First, say any additional information you need to say, then make ONE and ONLY ONE tool call using this exact format:
<function_call>
{
    "function": {
        "name": "tool_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
}
</function_call>

If you are requesting a tool call, DO NOT SEND A SINGLE CHARACTER AFTER "</function_call>". Your response should end immediately.
If you are being prompted with your latest context as a tool call result, analyze it and continue the conversation naturally.

IMPORTANT RULES:
- Include ONLY ONE tool call per response
- Use proper JSON escaping for special characters in strings (especially when including code)
- For code or text with quotes, escape with backslashes: \\" for quotes, \\\\ for backslashes
- Ensure all JSON is valid - test your JSON structure mentally before submitting
- Do not nest additional <function_call> tags inside arguments
- Format all arguments as proper JSON key-value pairs

Example for a weather request:
Let me check the weather for you.
<function_call>
{
    "function": {
        "name": "get_current_weather",
        "arguments": {
            "city": "New York"
        }
    }
}
</function_call>

Example with code as an argument:
I'll run this Python code for you.
<function_call>
{
    "function": {
        "name": "execute_python",
        "arguments": {
            "code": "def hello():\\n    print(\\"Hello, world!\\")\\n\\nhello()",
            "filename": "hello.py"
        }
    }
}
</function_call>
"""


def generate_tool_system_prompt(
    tool_definitions: str = "",
    formatting_instructions: str = "",
    user_system_prompt: str = "",
    tool_configuration: str = ""
) -> str:
    """Generate a system prompt for tool usage in Anthropic's format (https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#tool-use-system-prompt)
    
    Args:
        tool_definitions: JSON schema of available tools
        formatting_instructions: Instructions for tool output formatting
        user_system_prompt: Additional system prompt from user
        tool_configuration: Additional tool configuration
        
    Returns:
        Formatted system prompt string
    """
    prompt_template = """In this environment you have access to a set of tools you can use to answer the user's question.
{formatting_instructions}
String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the available tools:
{tool_definitions}
Remember: ALWAYS use the function call format when you need to use a tool.
{user_system_prompt}
{tool_configuration}
""".format(
        formatting_instructions=formatting_instructions,
        tool_definitions=tool_definitions,
        user_system_prompt=user_system_prompt,
        tool_configuration=tool_configuration
    ).strip()
    
    return prompt_template


def get_system_prompt(tools):
    tools_dict = convert_tools_to_dict(tools)
    return generate_tool_system_prompt(tool_definitions=json.dumps(tools_dict, indent=2), formatting_instructions=format_instructions(), user_system_prompt="", tool_configuration="")


SUMMARIZE_SYSTEM_PROMPT = """
Summarize the key points of this conversation while preserving important 
context. Focus on maintaining:
1. Essential information exchanged
2. Important decisions or conclusions
3. Current context needed for continuation
Be concise but ensure no critical details are lost.
"""
