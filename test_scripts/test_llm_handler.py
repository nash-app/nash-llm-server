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
    print("# OpenAI")
    print("OPENAI_API_KEY=sk-...        # Required for OpenAI models")
    print("OPENAI_API_BASE=...          # Optional, defaults to official API")
    print("\n# Anthropic")
    print("ANTHROPIC_API_KEY=sk-...     # Required for Anthropic models")
    print("ANTHROPIC_API_BASE=...       # Optional, defaults to official API")
    print("\n# Helicone (Optional)")
    print("HELICONE_API_KEY=sk-...      # Enable request tracking")
    print("```")
    print("\nYou'll be prompted to select a model after setup.")


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
    "claude-3-haiku-20240229",
    "claude-2.1"
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
    
    # Check for available API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    helicone_key = os.getenv("HELICONE_API_KEY")
    
    # Determine provider based on available keys
    if openai_key and anthropic_key:
        # Both keys available, let user choose
        provider = get_provider_choice()
        api_key = openai_key if provider == 'openai' else anthropic_key
    elif openai_key:
        provider = 'openai'
        api_key = openai_key
    elif anthropic_key:
        provider = 'anthropic'
        api_key = anthropic_key
    else:
        print("\nError: No API keys found in .env file")
        print("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return None, None, None
    
    # Set up base URL based on provider and Helicone
    if provider == 'openai':
        api_base_url = os.getenv("OPENAI_API_BASE")
        if helicone_key:
            api_base_url = "https://oai.helicone.ai/v1"
        elif not api_base_url:
            api_base_url = "https://api.openai.com/v1"
    else:  # anthropic
        api_base_url = os.getenv("ANTHROPIC_API_BASE")
        if helicone_key:
            api_base_url = "https://anthropic.helicone.ai"
        elif not api_base_url:
            api_base_url = "https://api.anthropic.com"
    
    # Get model choice from user
    model = get_model_choice(provider)
    
    print("\nUsing configuration:")
    print(f"- Provider: {provider.upper()}")
    print(f"- Model: {model}")
    print(f"- API Base URL: {api_base_url}")
    print(f"- Helicone: {'Enabled' if helicone_key else 'Disabled'}")
    
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
