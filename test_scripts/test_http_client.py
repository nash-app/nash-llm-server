import requests
import json
import sys
from typing import Generator, List, Dict, Optional
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.tool_parser import parse_tool_call, format_tool_result


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


# Provider and model selection now handled entirely through environment variables


# Using get_api_credentials imported from api_credentials.py


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
            "http://localhost:6274/v1/chat/completions/stream",
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
                            continue  # Continue to next chunk after handling content
                        
                        # Handle tool call events
                        elif "tool_call" in parsed:
                            tool_call = parsed["tool_call"]
                            print("\n\nTOOL CALL DETECTED")
                            print(f"Tool: {tool_call.get('tool_name', 'Unknown tool')}")
                            
                            # Call the tool via the MCP API
                            tool_name = tool_call.get('tool_name')
                            arguments = tool_call.get('arguments', {})
                            
                            if tool_name:
                                print(f"Calling tool: {tool_name}")
                                try:
                                    # Call the tool via the server API
                                    tool_result = call_mcp_tool(tool_name, arguments)
                                    
                                    # Add the tool result to the conversation
                                    result_message = f"Tool result: {tool_result}"
                                    full_response += f"\n{result_message}"
                                    yield f"\n\nTOOL RESULT: {tool_result}\n\n"
                                except Exception as e:
                                    print(f"Error calling tool: {str(e)}")
                            continue  # Continue to next chunk after handling tool call
                        
                        # Handle tool result events
                        elif "tool_result" in parsed:
                            tool_result = parsed["tool_result"]
                            print(f"\n\nTOOL RESULT: {tool_result}\n\n")
                            yield f"\n{tool_result}\n"
                            continue  # Continue to next chunk after handling tool result
                        
                        # Handle session ID events - already handled above
                        elif "session_id" in parsed:
                            # Skip since we already handled this above
                            continue
                            
                        # Unknown chunk type - debug only
                        else:
                            print(f"\nDEBUG - Unknown chunk type: {parsed}")
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

def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call an MCP tool through the server API."""
    try:
        payload = {
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        response = requests.post(
            "http://localhost:6274/v1/mcp/call_tool",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error calling tool: Server returned status code {response.status_code}"
        
        result = response.json()
        if "result" in result:
            return str(result["result"])
        else:
            return str(result)
    except Exception as e:
        return f"Error calling tool: {str(e)}"


# Using print_credentials_info imported from api_credentials.py

def chat_loop():
    conversation = Conversation()
    first_message = True
    message_count = 0
    
    # Get API configuration from environment
    try:
        api_key, api_base_url, model = get_api_credentials()
        conversation.set_api_key(api_key)
        conversation.set_api_base_url(api_base_url)
        conversation.set_model(model)
        # Print credentials info
        print_credentials_info(api_key, api_base_url, model)
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set all required environment variables and try again.")
        sys.exit(1)
    
    print("\n=== Chat Session Started ===")
    print("Commands:")
    print("- 'exit': End the conversation")
    print("- 'list-tools': List available MCP tools")
    print("-" * 60)
    
    while True:
        try:
            message_count += 1
            print(f"\n=== Message {message_count} ===")
            if conversation.session_id:
                print(f"Session: {conversation.session_id[:8]}...")
            else:
                print("Session: None")
            
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'list-tools':
                try:
                    response = requests.post(
                        "http://localhost:6274/v1/mcp/list_tools"
                    )
                    if response.status_code == 200:
                        result = response.json()
                        tools = result.get("tools", {})
                        print("\n=== Available MCP Tools ===")
                        if hasattr(tools, 'tools'):
                            for i, tool in enumerate(tools.tools, 1):
                                print(f"{i}. {tool.name}: {tool.description}")
                        else:
                            # Try to print tools directly from the response
                            print(json.dumps(tools, indent=2))
                    else:
                        print(f"Error listing tools: {response.status_code}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                continue
            
            # Configuration changes now handled through environment variables
            
            # Add user message to history
            conversation.add_message("user", user_input)
            
            # Enter a loop to handle multiple tool calls and responses
            while True:
                # Only send session_id after first message
                current_session_id = (
                    None if first_message else conversation.session_id
                )
                
                print("\nAssistant:", end=" ", flush=True)
                
                # Process the stream response
                response_text = ""
                received_tool_result = False
                tool_result_content = ""
                
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
                        if chunk.startswith("\n\nTOOL RESULT:"):
                            received_tool_result = True
                            tool_result_content = chunk
                        else:
                            print(chunk, end="", flush=True)
                            response_text += chunk
                print()
                
                # Save the assistant's response to history
                if response_text:
                    conversation.add_message("assistant", response_text)
                
                # Check if response contains a tool call pattern
                if "<tool_call>" in response_text:
                    print("\nTool call detected in response. Executing...")
                    
                    # Add closing tag if it's missing (similar to test_chat_with_mcp.py)
                    if "</tool_call>" not in response_text:
                        message_with_closing_tag = response_text + "</tool_call>"
                        print("Adding missing closing tag to tool call")
                    else:
                        message_with_closing_tag = response_text
                    
                    # Use the shared parser to extract tool call information
                    parsed = parse_tool_call(message_with_closing_tag)
                    
                    if parsed['tool_call_found']:
                        tool_name = parsed['tool_name']
                        arguments = parsed['arguments']
                        
                        print(f"Calling tool: {tool_name}")
                        
                        # Call the tool via the server API
                        tool_result = call_mcp_tool(tool_name, arguments)
                        
                        # Format the result using the shared formatter
                        formatted_result = format_tool_result(tool_result)
                        
                        # Display the result
                        print(f"\nTool result: {tool_result}")
                        
                        # Persist the end tag in the assistant message
                        if "</tool_call>" not in response_text:
                            conversation.messages[-1]['content'] += "</tool_call>"
                        
                        # Add the tool result as an assistant message
                        conversation.add_message("assistant", f"Tool result: {formatted_result}")
                        
                        # Continue the loop to get another response from the LLM
                        print("\nTOOL CALL --------------------------------------------------------")
                        print(formatted_result)
                        print("END CALL --------------------------------------------------------")
                        
                        # Continue to next iteration (get another assistant response)
                        continue
                    elif parsed['error']:
                        print(f"Error processing tool call: {parsed['error']}")
                    else:
                        print("Could not parse tool call")
                
                # If we got a tool result from streaming, add it as an assistant message
                if received_tool_result:
                    conversation.add_message("assistant", tool_result_content)
                
                # No tool call detected, exit the loop
                break
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except EOFError:
            print("\nExiting chat...")
            break


if __name__ == "__main__":
    chat_loop() 
