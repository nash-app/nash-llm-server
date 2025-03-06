import requests
import json
import sys
import os
from typing import Generator, List, Dict, Optional
from dotenv import load_dotenv


# Provider-specific base URLs
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
HELICONE_OPENAI_BASE_URL = "https://oai.helicone.ai/v1"
HELICONE_ANTHROPIC_BASE_URL = "https://anthropic.helicone.ai/"


def check_api_key():
    if not os.path.exists(".env"):
        print("Error: .env file not found.")
        print("Please create one based on .env.example:")
        print("cp .env.example .env")
        print("Then edit .env with your API keys")
        sys.exit(1)
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: No API keys found in .env file")
        print("You will need to provide an API key in your requests")


class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.session_id: str = None
        self.api_key: Optional[str] = None
        self.api_base_url: Optional[str] = None
        self.model: Optional[str] = None

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be either user or assistant")
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def set_messages(self, messages: List[Dict[str, str]]):
        """Replace current messages with new ones."""
        self.messages = messages

    def set_api_key(self, api_key: str):
        """Set the API key to use for requests."""
        self.api_key = api_key

    def set_api_base_url(self, api_base_url: str):
        """Set the API base URL to use for requests."""
        self.api_base_url = api_base_url

    def set_model(self, model: str):
        """Set the model to use for requests."""
        self.model = model


def get_api_config() -> Dict[str, str]:
    """Get API configuration from environment variables."""
    load_dotenv()
    
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    helicone_key = os.getenv("HELICONE_API_KEY")
    
    # Determine which API to use based on available keys
    if helicone_key:
        if anthropic_key:
            return {
                "api_key": anthropic_key,
                "api_base_url": HELICONE_ANTHROPIC_BASE_URL,
                "model": "claude-3-opus-20240229"
            }
        elif openai_key:
            return {
                "api_key": openai_key,
                "api_base_url": HELICONE_OPENAI_BASE_URL,
                "model": "gpt-4-turbo-preview"
            }
    else:
        if anthropic_key:
            return {
                "api_key": anthropic_key,
                "api_base_url": ANTHROPIC_BASE_URL,
                "model": "claude-3-opus-20240229"
            }
        elif openai_key:
            return {
                "api_key": openai_key,
                "api_base_url": OPENAI_BASE_URL,
                "model": "gpt-4-turbo-preview"
            }
    
    return {}


def stream_response(
    messages: List[Dict[str, str]],
    model: str = None,
    session_id: str = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None
) -> Generator[str, None, None]:
    try:
        print("\n=== Stream Response Start ===")
        print(f"Input session_id: {session_id}")
        print(f"Message count: {len(messages)}")
        if messages:
            print(f"First message role: {messages[0]['role']}")
            print(f"Last message role: {messages[-1]['role']}")
        
        payload = {
            "messages": messages,
            "session_id": session_id,
            "model": model,
            "api_key": api_key,
            "api_base_url": api_base_url
        }
            
        print("\nSending request to server...")
        response = requests.post(
            "http://localhost:8001/v1/chat/completions/stream",
            json=payload,
            stream=True,
        )
        
        if response.status_code != 200:
            error_msg = (
                f"Error: Server returned status code {response.status_code}"
            )
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg += f"\nDetails: {error_data['detail']}"
            except json.JSONDecodeError:
                pass
            print(error_msg)
            return
        
        print("\nProcessing server response...")
        full_response = ""
        first_session_id = None
        last_session_id = None
        chunk_count = 0
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    chunk_count += 1
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        print("\nReceived [DONE] marker")
                        break
                    try:
                        parsed = json.loads(data)
                        if "error" in parsed:
                            print("\nERROR")
                            print(f"Error content: {parsed['error']}")
                            return
                        if "warning" in parsed:
                            print("\nWARNING")
                            warning = parsed["warning"]
                            print(f"\n⚠️  {warning['warning']}")
                            print("\nSuggestions:")
                            for i, suggestion in enumerate(
                                warning["suggestions"], 1
                            ):
                                print(f"{i}. {suggestion}")
                            
                            details = warning["details"]
                            limits = details["limits"]
                            print("\nDetails:")
                            msg_count = details["message_count"]
                            max_msgs = limits["max_messages"]
                            print(f"- Messages: {msg_count}/{max_msgs}")
                            
                            est_tokens = details["estimated_tokens"]
                            max_tokens = limits["max_tokens"]
                            print(f"- Est. Tokens: {est_tokens}/{max_tokens}")
                            
                            print(
                                "\nTip: Use 'summarize' to condense the "
                                "conversation while keeping context."
                            )
                            return None, first_session_id
                        
                        if "session_id" in parsed:
                            session_id = parsed["session_id"]
                            if first_session_id is None:
                                first_session_id = session_id
                                print(f"\nGot first session_id: {session_id}")
                                # Yield session ID tuple immediately
                                yield "", first_session_id
                            last_session_id = session_id
                            continue
                            
                        if "content" in parsed:
                            content = parsed["content"]
                            full_response += content
                            yield content
                        else:
                            print(f"\nUnknown chunk type: {parsed}")
                    except json.JSONDecodeError:
                        print("\nParse error - Invalid JSON")
                        continue
        
        print("\n=== Stream Response Summary ===")
        print(f"Total chunks: {chunk_count}")
        if first_session_id != last_session_id:
            print("\n⚠️  WARNING: Session ID mismatch")
            print(f"First: {first_session_id}")
            print(f"Last: {last_session_id}")
            return None, None
            
        return full_response, first_session_id
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the LLM server.")
        print("Make sure to start it first with: poetry run llm_server")
        return None, None
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None, None


def summarize_conversation(
    messages: List[Dict[str, str]],
    model: str = None,
    session_id: str = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None
) -> Dict:
    """Request conversation summarization from the server."""
    try:
        payload = {
            "messages": messages,
            "model": model,
            "session_id": session_id,
            "api_key": api_key,
            "api_base_url": api_base_url
        }
            
        response = requests.post(
            "http://localhost:8001/v1/chat/summarize",
            json=payload
        )
        
        if response.status_code != 200:
            return {"error": f"Server error: {response.status_code}"}
        
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "Could not connect to server. "
            "Is it running?"
        }
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


def print_summarization_result(result: Dict) -> bool:
    """Print the summarization result in a user-friendly format."""
    if "error" in result:
        print(f"\n❌ Summarization failed: {result['error']}")
        return False
    
    if not result.get("success"):
        print("\n❌ Summarization failed: Unknown error")
        return False
    
    print("\n✨ Conversation summarized successfully!")
    print("\nSummary:")
    print("-" * 40)
    print(result["summary"])
    print("-" * 40)
    
    reduction = result["token_reduction"]
    saved = reduction["before"] - reduction["after"]
    percent = (
        (saved / reduction["before"]) * 100 
        if reduction["before"] > 0 else 0
    )
    
    print(f"\nToken reduction: {saved} ({percent:.1f}%)")
    print(f"- Before: {reduction['before']}")
    print(f"- After: {reduction['after']}")
    
    return True


def chat_loop():
    check_api_key()
    conversation = Conversation()
    first_message = True
    message_count = 0
    
    # Get API configuration from environment
    api_config = get_api_config()
    if api_config:
        conversation.set_api_key(api_config["api_key"])
        conversation.set_api_base_url(api_config["api_base_url"])
        conversation.set_model(api_config["model"])
    
    print("\n=== Chat Session Started ===")
    print("Commands:")
    print("- 'exit': End the conversation")
    print("- 'summarize': Summarize conversation to reduce length")
    print("- 'set-api-key <key>': Set API key for requests")
    print("- 'set-base-url <url>': Set API base URL for requests")
    print("- 'set-model <model>': Set model to use for requests")
    print("-" * 60)
    
    while True:
        try:
            message_count += 1
            print(f"\n=== Message {message_count} ===")
            if conversation.session_id:
                print(f"Session: {conversation.session_id[:8]}...")
            else:
                print("Session: None")
            if conversation.api_key:
                print(f"API Key: {conversation.api_key[:6]}...")
            else:
                print("API Key: Using environment variable")
            if conversation.api_base_url:
                print(f"API Base URL: {conversation.api_base_url}")
            if conversation.model:
                print(f"Model: {conversation.model}")
            
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'summarize':
                print("\n=== Summarizing Conversation ===")
                result = summarize_conversation(
                    conversation.get_messages(),
                    conversation.model,
                    conversation.session_id,
                    conversation.api_key,
                    conversation.api_base_url
                )
                if print_summarization_result(result):
                    conversation.set_messages(result["messages"])
                    if "session_id" in result:
                        conversation.session_id = result["session_id"]
                continue
            
            if user_input.lower().startswith('set-api-key '):
                api_key = user_input[11:].strip()
                if api_key:
                    conversation.set_api_key(api_key)
                    print(f"\nAPI key set: {api_key[:6]}...")
                else:
                    print("\nError: Please provide an API key")
                continue
            
            if user_input.lower().startswith('set-base-url '):
                api_base_url = user_input[12:].strip()
                if api_base_url:
                    conversation.set_api_base_url(api_base_url)
                    print(f"\nAPI base URL set: {api_base_url}")
                else:
                    print("\nError: Please provide an API base URL")
                continue
            
            if user_input.lower().startswith('set-model '):
                model = user_input[9:].strip()
                if model:
                    conversation.set_model(model)
                    print(f"\nModel set: {model}")
                else:
                    print("\nError: Please provide a model name")
                continue
            
            # Add user message to history
            conversation.add_message("user", user_input)
            
            # Only send session_id after first message
            current_session_id = (
                None if first_message else conversation.session_id
            )
            
            print("\nAssistant:", end=" ", flush=True)
            
            # Process the stream response
            response_text = ""
            response_chunks = []
            
            # Collect all chunks from the stream
            got_session_id = False
            for chunk in stream_response(
                conversation.get_messages(),
                conversation.model,
                current_session_id,
                conversation.api_key,
                conversation.api_base_url
            ):
                if isinstance(chunk, tuple):
                    # This is the final response tuple
                    response_text, new_session_id = chunk
                    if new_session_id and not got_session_id:
                        # Update session ID if we haven't already
                        conversation.session_id = new_session_id
                        first_message = False
                        got_session_id = True
                else:
                    # This is a content chunk
                    print(chunk, end="", flush=True)
                    response_chunks.append(chunk)
                    response_text += chunk
            print()
            
            # Add assistant's response to history if we got one
            if response_text:
                conversation.add_message("assistant", response_text)
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except EOFError:
            print("\nExiting chat...")
            break


def main():
    chat_loop()


if __name__ == "__main__":
    main() 