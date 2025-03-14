import litellm
from dotenv import load_dotenv
from typing import Optional


class InvalidAPIKeyError(Exception):
    """Raised when an API key is invalid or missing."""
    pass


def validate_api_key(api_key: Optional[str] = None, model: str = None) -> None:
    """Validate that an API key is present and has the correct format.

    Args:
        api_key: The API key to validate
        model: The model being used, to determine validation requirements

    Raises:
        InvalidAPIKeyError: If the API key is missing or invalid
    """
    # Skip validation for Ollama models
    if model and model.startswith("ollama/"):
        return
        
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


def configure_llm(api_key: str = None, api_base_url: str = None, model: str = None):
    """Configure LiteLLM with API keys and settings."""
    load_dotenv()

    if api_key:
        litellm.api_key = api_key
    if api_base_url:
        litellm.api_base = api_base_url

    # Validate API key
    validate_api_key(api_key, model)

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
        configure_llm(api_key, api_base_url, model)

        # Create the response stream with stop sequence
        response = await litellm.acompletion(
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
