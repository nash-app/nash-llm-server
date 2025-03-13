from litellm import acompletion
import litellm
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
):
    """Stream responses from the LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Optional model override
        api_key: Optional API key override
        api_base_url: Optional API base URL override

    Yields:
        Direct chunks from the LiteLLM API
    """
    try:
        if not messages:
            messages = []

        # Configure LLM with provided credentials
        configure_llm(api_key, api_base_url)
        
        # Create the response stream with stop sequence
        response = await acompletion(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.3,
            stop=["</tool_call>"],  # Stop after our explicit marker
            extra_headers=litellm.headers
        )

        # Simply yield each chunk directly
        async for chunk in response:
            yield chunk
            
    except Exception as e:
        # Simple error handling - just print the error
        print(f"\nError in stream_llm_response: {type(e).__name__}: {str(e)}")
        print(f"Messages: {len(messages)} items")
        
        # Re-raise to let the caller handle it
        raise


async def summarize_conversation(
    messages: list,
    model: str,
    api_key: str = None,
    api_base_url: str = None,
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
            }
        }
        
    except Exception as e:
        return {"error": f"Error during summarization: {str(e)}"}
