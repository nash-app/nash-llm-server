from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .llm_handler import stream_llm_response, configure_llm
from .mcp_handler import MCPHandler


app = FastAPI(title="Nash LLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Or specify: ["GET", "POST"]
    allow_headers=["*"],  # Or specify required headers
)


@app.on_event("startup")
async def startup_event():
    """Configure services on server startup."""
    # Configure LLM
    configure_llm()
    
    # Initialize MCP singleton
    mcp = MCPHandler.get_instance()
    await mcp.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    mcp = MCPHandler.get_instance()
    await mcp.close()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model = data.get("model")
    
    return StreamingResponse(
        stream_llm_response(messages, model),
        media_type="text/event-stream"
    )


@app.post("/v1/mcp/list_tools")
async def list_tools():
    """List all available MCP tools."""
    try:
        mcp = MCPHandler.get_instance()
        tools = await mcp.list_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/list_prompts")
async def list_prompts():
    """List all available prompts."""
    try:
        mcp = MCPHandler.get_instance()
        prompts = await mcp.list_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/get_prompt")
async def get_prompt(request: Request):
    """Get a specific prompt."""
    try:
        data = await request.json()
        prompt_name = data.get("prompt_name")
        arguments = data.get("arguments", {})
        
        if not prompt_name:
            raise HTTPException(
                status_code=400, 
                detail="prompt_name is required"
            )
            
        mcp = MCPHandler.get_instance()
        prompt = await mcp.get_prompt(prompt_name, arguments=arguments)
        return {"prompt": prompt}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/list_resources")
async def list_resources():
    """List all available resources."""
    try:
        mcp = MCPHandler.get_instance()
        resources = await mcp.list_resources()
        return {"resources": resources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/read_resource")
async def read_resource(request: Request):
    """Read a specific resource."""
    try:
        data = await request.json()
        resource_path = data.get("resource_path")
        
        if not resource_path:
            raise HTTPException(
                status_code=400, 
                detail="resource_path is required"
            )
            
        mcp = MCPHandler.get_instance()
        content = await mcp.read_resource(resource_path)
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/call_tool")
async def call_tool(request: Request):
    """Call a specific MCP tool."""
    try:
        data = await request.json()
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(
                status_code=400, 
                detail="tool_name is required"
            )
            
        mcp = MCPHandler.get_instance()
        result = await mcp.call_tool(tool_name, arguments=arguments)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
