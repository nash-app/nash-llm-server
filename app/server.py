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


@app.post("/v1/mcp/{method}")
async def mcp_method(request: Request, method: str):
    """Generic endpoint for all MCP methods."""
    try:
        mcp = MCPHandler.get_instance()
        
        # Get method arguments from request body
        args = await request.json() if await request.body() else {}
        
        if not hasattr(mcp, method):
            raise HTTPException(
                status_code=400,
                detail=f"Method '{method}' not found on MCP handler"
            )
        
        # Get the method and call it with args
        handler_method = getattr(mcp, method)
        result = await handler_method(**args)
        
        print(f"\nMCP {method} result:")
        print(result)
        
        return {"result": result}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"\nError in mcp_method: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error calling MCP method '{method}': {str(e)}"
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
