import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response


async def chat():
    configure_llm()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Stream AI response
            print("\nAssistant: ", end="", flush=True)
            assistant_message = ""
            
            async for chunk in stream_llm_response(messages):
                if "error" in chunk:
                    print("\nError:", chunk)
                    break
                    
                if "[DONE]" not in chunk:
                    content = json.loads(
                        chunk.replace("data: ", "")
                    ).get("content", "")
                    print(content, end="", flush=True)
                    assistant_message += content
            
            # Add assistant response to history
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
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
