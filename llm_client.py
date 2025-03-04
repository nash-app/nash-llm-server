import requests
import json
import sys
import os
from typing import Generator
from dotenv import load_dotenv


def check_api_key():
    if not os.path.exists(".env"):
        print("Error: .env file not found.")
        print("Please create one based on .env.example:")
        print("cp .env.example .env")
        print("Then edit .env with your OpenAI API key")
        sys.exit(1)
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env file")
        print("Please edit .env and add your OpenAI API key")
        sys.exit(1)


def stream_response(
    prompt: str,
    model: str = None
) -> Generator[str, None, None]:
    try:
        payload = {"prompt": prompt}
        if model:
            payload["model"] = model
            
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
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        if "error" in parsed:
                            print(f"\nError: {parsed['error']}")
                            return
                        content = parsed.get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
                        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the LLM server.")
        print("Make sure to start it first with: poetry run llm_server")
        return


def main():
    check_api_key()
    
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("Enter your prompt (Ctrl+D or Ctrl+Z to submit):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: Empty prompt")
        return

    print("\nResponse:")
    response_received = False
    for chunk in stream_response(prompt):
        response_received = True
        print(chunk, end="", flush=True)
    
    if response_received:
        print("\n")


if __name__ == "__main__":
    main() 