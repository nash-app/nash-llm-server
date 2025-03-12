from litellm import acompletion
import json
import litellm
import uuid
from dotenv import load_dotenv
from .prompts import SUMMARIZE_SYSTEM_PROMPT
from typing import Optional


class InvalidAPIKeyError(Exception):
    """Raised when an API key is invalid or missing."""
    pass


def validate_api_key(api_key: Optional[str] = None) -> None:
    """Validate that an API key is present and has the correct format.
    
    Args:
        api_key: The API key to validate
        
    Raises:
        InvalidAPIKeyError: If the API key is missing or invalid
    """
    if not api_key and not litellm.api_key:
        raise InvalidAPIKeyError(
            "No API key provided. Please set a valid API key."
        )
    
    key_to_check = api_key or litellm.api_key
    if not isinstance(key_to_check, str):
        raise InvalidAPIKeyError("API key must be a string.")
        
    # Basic format validation for common API key formats
    if not (key_to_check.startswith('sk-') and len(key_to_check) > 20):
        raise InvalidAPIKeyError(
            "Invalid API key format. API keys should start with 'sk-' "
            "and be at least 20 characters long."
        )


def get_session_id(session_id: str = None) -> str:
    """Get session ID, creating a new one if none provided."""
    return str(uuid.uuid4()) if not session_id else session_id


def configure_llm(api_key: str = None, api_base_url: str = None):
    """Configure LiteLLM with API keys and settings."""
    load_dotenv()

    if api_key:
        litellm.api_key = api_key
    if api_base_url:
        litellm.api_base = api_base_url

    # Validate API key
    validate_api_key()

    # Initialize headers
    litellm.headers = {}


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
        
        # Get or create session ID
        session_id = get_session_id(session_id)

        # Send session ID in first chunk
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        response = await acompletion(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )

        async for chunk in response:
            if chunk and hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
    except InvalidAPIKeyError as e:
        error_msg = f"API Key Error: {str(e)}"
        msg_data = {'error': error_msg}
        yield f"data: {json.dumps(msg_data)}\n\n"
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
                    "Please summarize this conversation:\n\n"
                    + "\n".join([
                        f"{m['role']}: {m['content']}"
                        for m in messages
                    ])
                )
            }
        ]
        
        # Get or create session ID
        session_id = get_session_id(session_id)
        
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
        
        # Estimate tokens for before/after comparison
        tokens_before = sum(
            len(str(msg.get("content", ""))) // 4 for msg in messages
        )
        tokens_after = sum(
            len(str(msg.get("content", ""))) // 4 
            for msg in summarized_messages
        )
        
        return {
            "success": True,
            "summary": summary,
            "messages": summarized_messages,
            "token_reduction": {
                "before": tokens_before,
                "after": tokens_after
            },
            "session_id": session_id
        }
        
    except Exception as e:
        return {"error": f"Error during summarization: {str(e)}"}
