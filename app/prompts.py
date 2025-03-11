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
    return """When a user's request requires using a tool, you MUST format your response in two parts:

1. First, make the tool call using this exact format:
<function_call>[{
    "function": {
        "name": "tool_name",
        "arguments": {"arg1": "value1"}
    }
}]</function_call>

2. After making the tool call, wait for the result before continuing the conversation.

For example, if someone asks about the weather, respond like this:
Let me check the weather for you.
<function_call>[{
    "function": {
        "name": "get_current_weather",
        "arguments": {"city": "New York"}
    }
}]</function_call>
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
