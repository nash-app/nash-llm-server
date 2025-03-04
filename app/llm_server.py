from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion
import uvicorn
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Nash LLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Or specify: ["GET", "POST"]
    allow_headers=["*"],  # Or specify required headers
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")


async def stream_llm_response(prompt: str, model: str = None):
    try:
        response = await acompletion(
            model=model or DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        async for chunk in response:
            if chunk and hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    model = data.get("model")
    
    return StreamingResponse(
        stream_llm_response(prompt, model),
        media_type="text/event-stream"
    )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file based on .env.example")
        return
    
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main() 