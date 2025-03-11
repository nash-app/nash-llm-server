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
