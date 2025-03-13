import asyncio
import json
import os
from dotenv import load_dotenv
from app.llm_handler import (
    stream_llm_response,
    summarize_conversation
)


def print_setup_instructions():
    """Print instructions for setting up API keys and configuration."""
    print("\n=== Nash LLM Test Script ===")
    print("\nThis script uses environment variables for API configuration.")
    print("To set up, copy .env.example to .env and configure your keys:")
    print("\n```bash")
    print("cp .env.example .env")
    print("```\n")
    print("Then edit .env with your configuration:")
    print("```")
    print("# Provider credentials")
    print("PROVIDER_API_KEY=sk-...      # Required - LLM API key")
    print("PROVIDER_API_BASE=...        # Required - API base URL (e.g. https://api.anthropic.com)")
    print("PROVIDER_MODEL=...           # Required - Model to use (e.g. claude-3-sonnet-20240229)")
    print("```")
    print("\nAll three environment variables are required.")


# Available models for each provider
OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-4-0125-preview",
    "gpt-4",
    "gpt-3.5-turbo"
]

ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
]


def get_model_choice(provider):
    """Get the user's choice of model for the selected provider."""
    models = OPENAI_MODELS if provider == 'openai' else ANTHROPIC_MODELS
    
    print(f"\nAvailable {provider.upper()} models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"\nChoose model (1-{len(models)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
        except ValueError:
            pass
        print("Invalid choice. Please enter a number between 1 and "
              f"{len(models)}.")


def get_provider_choice():
    """Get the user's choice of AI provider."""
    while True:
        choice = input(
            "\nChoose AI provider (1 for OpenAI, 2 for Anthropic): "
        ).strip()
        if choice in ['1', '2']:
            return 'openai' if choice == '1' else 'anthropic'
        print("Invalid choice. Please enter 1 or 2.")


def get_credentials():
    """Get API credentials from environment variables."""
    load_dotenv()
    
    # Get provider API key
    api_key = os.getenv("PROVIDER_API_KEY")
    if not api_key:
        print("\nError: No PROVIDER_API_KEY found in .env file")
        print("Please set PROVIDER_API_KEY in your .env file")
        return None, None, None
    
    # Get provider base URL
    api_base_url = os.getenv("PROVIDER_API_BASE")
    if not api_base_url:
        print("\nError: No PROVIDER_API_BASE found in .env file")
        print("Please set PROVIDER_API_BASE in your .env file")
        return None, None, None
    
    # Get provider model
    model = os.getenv("PROVIDER_MODEL")
    if not model:
        print("\nError: No PROVIDER_MODEL found in .env file")
        print("Please set PROVIDER_MODEL in your .env file")
        return None, None, None
    
    print("\nUsing configuration:")
    print(f"- API Key: {api_key[:6]}...")
    print(f"- Model: {model}")
    print(f"- API Base URL: {api_base_url}")
    
    return api_key, api_base_url, model


async def chat():
    api_key, api_base_url, model = get_credentials()
    if not api_key:
        return
    
    messages = []
    session_id = None
    
    try:
        while True:
            # Get user input
            prompt = "\nYou (type 'summarize' to test summarization, " \
                    "'quit' to exit): "
            user_input = input(prompt).strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if user_input.lower() == 'summarize':
                if len(messages) < 2:
                    print("Not enough messages to summarize yet.")
                    continue
                
                print("\nSummarizing conversation...")
                result = await summarize_conversation(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    session_id=session_id
                )
                
                if "error" in result:
                    print("Error:", result["error"])
                else:
                    print("\nSummary:", result["summary"])
                    print("\nToken reduction:", result["token_reduction"])
                    messages = result["messages"]
                    session_id = result["session_id"]
                continue
                
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Stream AI response
            print("\nAssistant: ", end="", flush=True)
            assistant_message = ""
            
            async for chunk in stream_llm_response(
                messages=messages,
                model=model,
                api_key=api_key,
                api_base_url=api_base_url,
                session_id=session_id
            ):
                if "error" in chunk:
                    error_data = json.loads(chunk.replace("data: ", ""))
                    print("\nError:", error_data["error"])
                    break
                
                if "session_id" in chunk:
                    session_data = json.loads(chunk.replace("data: ", ""))
                    session_id = session_data["session_id"]
                    continue
                
                if "warning" in chunk:
                    warning_data = json.loads(chunk.replace("data: ", ""))
                    print("\nWarning:", warning_data["warning"])
                    suggestions = warning_data["warning"]["suggestions"]
                    print("Suggestions:", "\n- ".join([""] + suggestions))
                    continue
                
                if "[DONE]" not in chunk:
                    content = json.loads(
                        chunk.replace("data: ", "")
                    ).get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        assistant_message += content
            
            # Add assistant response to history if we got one
            if assistant_message:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
            
            print(f"\n(Session ID: {session_id})")
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    
    print("\nChat ended. Final message count:", len(messages))


if __name__ == "__main__":
    print_setup_instructions()
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}") 
