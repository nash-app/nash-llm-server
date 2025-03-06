from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion
import uvicorn
import json
from dotenv import load_dotenv
import os
import litellm
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from .prompts import CHAT_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT

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

# Constants
MAX_MESSAGES = 20  # Maximum number of messages before suggesting summarization
MAX_TOTAL_TOKENS = 50000  # Approximate token limit before warning

# Provider-specific base URLs
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
HELICONE_OPENAI_BASE_URL = "https://oai.helicone.ai/v1"
HELICONE_ANTHROPIC_BASE_URL = "https://anthropic.helicone.ai/"

# Configure Helicone if key is provided
HELICONE_API_KEY = os.getenv('HELICONE_API_KEY')
if HELICONE_API_KEY:
    print(f"Configuring Helicone with API key: {HELICONE_API_KEY[:6]}...")
    litellm.headers = {
        "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
    }


class Message(BaseModel):
    """A message in the conversation."""
    role: str = Field(
        ...,
        description="The role of the message sender (user/assistant/system)"
    )
    content: str = Field(..., description="The content of the message")


class BaseRequest(BaseModel):
    """Base request model with common fields."""
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking"
    )
    api_key: str = Field(
        ...,
        description="API key to use for the request"
    )
    api_base_url: str = Field(
        ...,
        description="API base URL to use for the request"
    )


class StreamRequest(BaseRequest):
    """Request model for streaming completions."""
    model: str = Field(
        ...,
        description="Model to use for completion"
    )


class SummarizeRequest(BaseRequest):
    """Request model for conversation summarization."""
    model: str = Field(
        ...,
        description="Model to use for summarization"
    )


def get_helicone_headers(session_id: str = None) -> dict:
    """Get Helicone headers, creating a new session ID if none provided."""
    headers = {}
    if HELICONE_API_KEY:
        headers["Helicone-Auth"] = f"Bearer {HELICONE_API_KEY}"
        new_session = str(uuid.uuid4()) if not session_id else session_id
        print(
            "Server Debug - get_helicone_headers input session_id: "
            f"{session_id}"
        )
        print(
            "Server Debug - get_helicone_headers using session_id: "
            f"{new_session}"
        )
        headers["Helicone-Session-Id"] = new_session
        headers["Helicone-Session-Name"] = "DIRECT"
    return headers


def estimate_tokens(messages: list) -> int:
    """Rough estimation of tokens in messages. 1 token ≈ 4 chars in English."""
    return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)


async def summarize_conversation(
    messages: list,
    model: str,
    api_key: str,
    api_base_url: str,
    session_id: str = None
) -> dict:
    """Summarize a conversation to reduce token count while preserving 
    context."""
    try:
        if not messages:
            return {"error": "No messages to summarize"}
        
        # Set API configuration
        litellm.api_base = api_base_url
        litellm.api_key = api_key
        
        # Keep system message separate if it exists
        system_msg = None
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]
            messages = messages[1:]
        
        # Prepare messages for summarization
        summary_request = [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Please summarize this conversation:\n\n" + 
                "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            }
        ]
        
        # Get headers with session ID
        headers = get_helicone_headers(session_id)
        litellm.headers = headers
        
        response = await acompletion(
            model=model,
            messages=summary_request,
            stream=False
        )
        
        summary = response.choices[0].message.content
        
        # Create new conversation with summary
        summarized_messages = []
        if system_msg:
            summarized_messages.append(system_msg)
        
        summarized_messages.extend([
            {
                "role": "assistant",
                "content": "Previous conversation summary:\n" + summary
            }
        ])
        
        return {
            "success": True,
            "summary": summary,
            "messages": summarized_messages,
            "token_reduction": {
                "before": estimate_tokens(messages),
                "after": estimate_tokens(summarized_messages)
            },
            "session_id": headers['Helicone-Session-Id']
        }
        
    except Exception as e:
        return {"error": f"Error during summarization: {str(e)}"}


@app.post("/v1/chat/summarize")
async def summarize_completion(request: SummarizeRequest):
    try:
        result = await summarize_conversation(
            [msg.dict() for msg in request.messages],
            request.model,
            request.api_key,
            request.api_base_url,
            request.session_id
        )
        return result
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}


async def stream_llm_response(
    messages: list,
    model: str,
    api_key: str,
    api_base_url: str,
    session_id: str = None
):
    try:
        if not messages:
            messages = []
        
        print(
            "\nServer Debug - stream_llm_response received session_id: "
            f"{session_id}"
        )
        
        # Ensure system message is first if not already present
        if not messages or messages[0].get("role") != "system":
            messages.insert(
                0,
                {"role": "system", "content": CHAT_SYSTEM_PROMPT}
            )
        
        # Check conversation length and token count
        num_messages = len(messages)
        estimated_tokens = estimate_tokens(messages)
        
        # Get headers with session ID
        headers = get_helicone_headers(session_id)
        session_id = headers['Helicone-Session-Id']
        print(
            "Server Debug - stream_llm_response using session_id: "
            f"{session_id}"
        )
        
        # Set API configuration
        litellm.api_base = api_base_url
        litellm.api_key = api_key
        litellm.headers = headers
        
        print("\nMaking LLM request:")
        print(f"Session ID: {session_id}")
        print(f"API Base: {litellm.api_base}")
        print(f"Model: {model}")
        print(f"Message Count: {num_messages}")
        print(f"Estimated tokens: {estimated_tokens}")
        print(f"Headers: {headers}")
        if num_messages > 0:
            print(f"First message role: {messages[0]['role']}")
            print(f"Last message role: {messages[-1]['role']}")
        
        # Send session ID in first chunk
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        
        if num_messages > MAX_MESSAGES or estimated_tokens > MAX_TOTAL_TOKENS:
            warning = {
                "warning": "Conversation length exceeds recommended limits",
                "suggestions": [
                    "Summarize the conversation so far and start fresh",
                    "Keep only the most recent and relevant messages",
                    "Clear the conversation while preserving system message"
                ],
                "details": {
                    "message_count": num_messages,
                    "estimated_tokens": estimated_tokens,
                    "limits": {
                        "max_messages": MAX_MESSAGES,
                        "max_tokens": MAX_TOTAL_TOKENS
                    }
                }
            }
            yield f"data: {json.dumps({'warning': warning})}\n\n"
            return
        
        response = await acompletion(
            model=model,
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
    finally:
        # Always send DONE with session ID to ensure client has it
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: StreamRequest):
    return StreamingResponse(
        stream_llm_response(
            [msg.dict() for msg in request.messages],
            request.model,
            request.api_key,
            request.api_base_url,
            request.session_id
        ),
        media_type="text/event-stream"
    )


def main():
    if not os.getenv("HELICONE_API_KEY"):
        print("Warning: HELICONE_API_KEY not found in environment variables.")
        print("Please create a .env file based on .env.example")
    
    print("\nStarting LLM server with configuration:")
    print(f"System Prompt: {CHAT_SYSTEM_PROMPT}")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main() 