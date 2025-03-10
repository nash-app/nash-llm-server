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

# Global MCP handler
mcp = None


@app.on_event("startup")
async def startup_event():
    """Configure services on server startup."""
    global mcp
    
    # Configure LLM
    configure_llm()
    
    # Initialize MCP
    mcp = MCPHandler()
    await mcp.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    global mcp
    if mcp:
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
    """Execute an MCP method."""
    global mcp
    
    try:
        if not mcp:
            raise HTTPException(
                status_code=500,
                detail="MCP client not initialized"
            )
            
        # Get method arguments from request body
        args = await request.json() if await request.body() else {}
        
        result = await mcp.execute_method(method, **args)
        return {"result": result}
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling MCP method '{method}': {str(e)}"
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
