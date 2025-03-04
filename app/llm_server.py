from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from litellm import acompletion
import uvicorn
import json
from dotenv import load_dotenv
import os
import litellm

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Nash LLM Server")

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"
SYSTEM_PROMPT = "You are a helpful AI assistant."

# Configure LiteLLM to use Helicone proxy
HELICONE_API_KEY = os.getenv('HELICONE_API_KEY')
if HELICONE_API_KEY:
    print(f"Configuring Helicone proxy with API key: {HELICONE_API_KEY[:6]}...")
    litellm.api_base = "https://oai.helicone.ai/v1"
    litellm.headers = {"Helicone-Auth": f"Bearer {HELICONE_API_KEY}"}
else:
    print("Warning: HELICONE_API_KEY not found in environment variables")


async def stream_llm_response(prompt: str, model: str = None):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        print("\nMaking LLM request:")
        print(f"API Base: {litellm.api_base}")
        print(f"Headers: {litellm.headers}")
        print(f"Model: {model or DEFAULT_MODEL}")
        print(f"Messages: {messages}")
        
        response = await acompletion(
            model=model or DEFAULT_MODEL,
            messages=messages,
            stream=True
        )
        
        async for chunk in response:
            if chunk and hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        print(f"\nError in stream_llm_response: {str(e)}")
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

    if not os.getenv("HELICONE_API_KEY"):
        print("Error: HELICONE_API_KEY not found in environment variables.")
        print("Please create a .env file based on .env.example")
        return
    
    print("\nStarting LLM server with configuration:")
    print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')[:6]}...")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"System Prompt: {SYSTEM_PROMPT}")
    print(f"Helicone API Base: {litellm.api_base}")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main() 