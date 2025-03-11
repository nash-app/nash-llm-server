from litellm import acompletion
import json
import os
import litellm
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4-turbo"


def configure_llm():
    """Configure LiteLLM with API keys and settings."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    litellm.api_key = api_key


async def stream_llm_response(messages: list = None, model: str = None):
    """Stream responses from the LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Optional model override

    Yields:
        SSE formatted JSON strings with content or error messages
    """
    response = None
    try:
        if not messages:
            messages = []

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
        yield "data: [DONE]\n\n"
        if response and hasattr(response, 'aclose'):
            await response.aclose()
