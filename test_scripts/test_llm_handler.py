import asyncio
import json
import os
from dotenv import load_dotenv
from app.llm_handler import (
    stream_llm_response,
    summarize_conversation
)


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

PROVIDER_BASE_URLS = {
    'openai': {
        'direct': "https://api.openai.com/v1",
        'helicone': "https://oai.helicone.ai/v1"
    },
    'anthropic': {
        'direct': "https://api.anthropic.com",
        'helicone': "https://anthropic.helicone.ai"
    }
}


def get_provider_choice():
    """Get the user's choice of AI provider."""
    while True:
        choice = input(
            "\nChoose AI provider (1 for OpenAI, 2 for Anthropic): "
        ).strip()
        if choice in ['1', '2']:
            return 'openai' if choice == '1' else 'anthropic'
        print("Invalid choice. Please enter 1 or 2.")


def get_model_choice(provider):
    """Get the user's choice of model for the selected provider."""
    models = OPENAI_MODELS if provider == 'openai' else ANTHROPIC_MODELS
    
    print(f"\nAvailable {provider.upper()} models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(
                f"\nChoose model (1-{len(models)}): "
            ).strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
        except ValueError:
            pass
        print("Invalid choice. Please enter a number between "
              f"1 and {len(models)}.")


def get_helicone_choice():
    """Get user's choice whether to use Helicone."""
    while True:
        choice = input(
            "\nUse Helicone for request tracking? (y/n): "
        ).strip().lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print("Invalid choice. Please enter 'y' or 'n'.")


def get_credentials():
    """Get API credentials from environment or user input."""
    load_dotenv()
    
    # Get provider and model choice
    provider = get_provider_choice()
    model = get_model_choice(provider)
    use_helicone = get_helicone_choice()
    
    # Set up credentials based on provider
    if provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ").strip()
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("Enter your Anthropic API key: ").strip()
    
    # Set up base URL based on provider and Helicone choice
    routing = 'helicone' if use_helicone else 'direct'
    api_base_url = PROVIDER_BASE_URLS[provider][routing]
    
    if use_helicone:
        helicone_key = os.getenv("HELICONE_API_KEY")
        if not helicone_key:
            print("\nWarning: HELICONE_API_KEY not found in environment.")
            print("Helicone tracking will not work without an API key.")
    
    print(f"\nConfiguration:")
    print(f"- Provider: {provider.upper()}")
    print(f"- Model: {model}")
    print(f"- Base URL: {api_base_url}")
    print(f"- Helicone: {'Enabled' if use_helicone else 'Disabled'}")
    
    return provider, model, api_key, api_base_url


async def chat():
    provider, model, api_key, api_base_url = get_credentials()
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
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}") 
