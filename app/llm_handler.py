from litellm import acompletion
import json
import os
import litellm
import uuid
from dotenv import load_dotenv
from .prompts import SUMMARIZE_SYSTEM_PROMPT


DEFAULT_MODEL = "gpt-4-turbo"
MAX_MESSAGES = 20  # Maximum number of messages before suggesting summarization
MAX_TOTAL_TOKENS = 50000  # Approximate token limit before warning


def get_helicone_headers(session_id: str = None) -> dict:
    """Get Helicone headers, creating a new session ID if none provided."""
    headers = {}
    helicone_api_key = os.getenv('HELICONE_API_KEY')
    if helicone_api_key:
        headers["Helicone-Auth"] = f"Bearer {helicone_api_key}"
        new_session = str(uuid.uuid4()) if not session_id else session_id
        headers["Helicone-Session-Id"] = new_session
        headers["Helicone-Session-Name"] = "DIRECT"
    return headers


def estimate_tokens(messages: list) -> int:
    """Rough estimation of tokens in messages. 1 token ≈ 4 chars in English."""
    return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)


def configure_llm(api_key: str = None, api_base_url: str = None):
    """Configure LiteLLM with API keys and settings."""
    load_dotenv()

    if api_key:
        litellm.api_key = api_key
    if api_base_url:
        litellm.api_base = api_base_url

    # Configure Helicone if available
    helicone_api_key = os.getenv('HELICONE_API_KEY')
    if helicone_api_key:
        litellm.headers = {
            "Helicone-Auth": f"Bearer {helicone_api_key}",
        }


async def stream_llm_response(
    messages: list = None,
    model: str = None,
    api_key: str = None,
    api_base_url: str = None,
    session_id: str = None
):
    """Stream responses from the LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Optional model override
        api_key: Optional API key override
        api_base_url: Optional API base URL override
        session_id: Optional session ID for tracking

    Yields:
        SSE formatted JSON strings with content or error messages
    """
    try:
        if not messages:
            messages = []

        # Configure LLM with provided credentials
        configure_llm(api_key, api_base_url)
        
        # Get headers with session ID
        headers = get_helicone_headers(session_id)
        litellm.headers.update(headers)
        session_id = headers.get('Helicone-Session-Id')

        # Send session ID in first chunk
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        # Check conversation length and token count
        num_messages = len(messages)
        estimated_tokens = estimate_tokens(messages)

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
            model=model or DEFAULT_MODEL,
            messages=messages,
            stream=True,
            temperature=0.7
        )

        async for chunk in response:
            if chunk and hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
    except GeneratorExit:
        # Handle generator cleanup gracefully
        return
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Always send DONE with session ID to ensure client has it
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"


async def summarize_conversation(
    messages: list,
    model: str,
    api_key: str = None,
    api_base_url: str = None,
    session_id: str = None
) -> dict:
    """Summarize a conversation to reduce token count while preserving context."""
    try:
        if not messages:
            return {"error": "No messages to summarize"}
        
        # Configure LLM with provided credentials
        configure_llm(api_key, api_base_url)
        
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
                "content": (
                    "Please summarize this conversation:\n\n" + 
                    "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                )
            }
        ]
        
        # Get headers with session ID
        headers = get_helicone_headers(session_id)
        litellm.headers.update(headers)
        
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
