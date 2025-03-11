"""Generate system prompts for tool usage."""


#def generate_tool_system_prompt(
#    tool_definitions: str = "",
#    formatting_instructions: str = "",
#    user_system_prompt: str = "",
#    tool_configuration: str = ""
#) -> str:
#    """Generate a system prompt for tool usage in Anthropic's format.
#    
#    Args:
#        tool_definitions: JSON schema of available tools
#        formatting_instructions: Instructions for tool output formatting
#        user_system_prompt: Additional system prompt from user
#        tool_configuration: Additional tool configuration
#        
#    Returns:
#        Formatted system prompt string
#    """
#    base_prompt = """In this environment you have access to a set of tools you can use to answer the user's question."""
#    
#    # Build the complete prompt with all components
#    components = [
#        base_prompt,
#        formatting_instructions,
#        "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.",
#        "Here are the functions available in JSONSchema format:",
#        tool_definitions,
#        user_system_prompt,
#        tool_configuration
#    ]
#    
#    # Filter out empty components and join with newlines
#    return "\n\n".join(comp for comp in components if comp) 

def generate_tool_system_prompt(
    tool_definitions: str = "",
    formatting_instructions: str = "",
    user_system_prompt: str = "",
    tool_configuration: str = ""
) -> str:
    base_prompt = """You are a helpful AI assistant with access to various tools. \
When a user's request requires using a tool, you MUST format your response in two parts:

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

Here are the available tools:"""

    # Build the complete prompt with all components
    components = [
        base_prompt,
        tool_definitions,
        "Remember: ALWAYS use the function call format when you need to use a tool.",
        user_system_prompt
    ]
    
    # Filter out empty components and join with newlines
    return "\n\n".join(comp for comp in components if comp)
