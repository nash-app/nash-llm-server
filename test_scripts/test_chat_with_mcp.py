import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt
from test_scripts.api_credentials import get_api_credentials, print_credentials_info


async def chat():
    # Get API credentials from environment
    api_key, api_base_url, model = get_api_credentials()
    
    # Configure LLM with credentials
    configure_llm(api_key=api_key, api_base_url=api_base_url)
    
    # Print credentials info
    print_credentials_info(api_key, api_base_url, model)
    
    messages = []
    
    # Initialize MCP and get tool definitions
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    tools = await mcp.list_tools()

    system_prompt = get_system_prompt(tools)

    messages.append({
        "role": "system",
        "content": system_prompt
    })

    try:
        while True:
            # Define ANSI escape codes for bold text
            bold_start = "\033[1m"
            bold_end = "\033[0m"
            
            # Print current message history
            print(f"\n{bold_start}Current Messages{bold_end} " + "-" * 65)
            for i, message in enumerate(messages):
                role = message['role']
                content = message['content']
                # Truncate long content to first 200 chars with ellipsis
                if len(content) > 200:
                    content = content[:197] + "..."
                # Add extra formatting for better readability
                print(f"\n{bold_start}{i+1}. {role}{bold_end}:")
                # Indent the content
                for line in content.split('\n'):
                    print(f"   {line}")
                # Add a separator between messages except after the last one
                if i < len(messages) - 1:
                    print("   " + "-" * 50)
            
            # Get user input
            print(f"\n{bold_start}User{bold_end} " + "-" * 70)
            user_input = input("").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Stream AI response
            print(f"\n{bold_start}Assistant{bold_end} " + "-" * 70)
            assistant_message = ""
            
            async for chunk in stream_llm_response(
                messages=messages,
                model=model,
                api_key=api_key,
                api_base_url=api_base_url
            ):
                # Process the raw LiteLLM chunk
                if hasattr(chunk, 'choices') and chunk.choices:
                    # Extract the content from the choices
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        print(content, end="", flush=True)
                        assistant_message += content
            
            # Add assistant response to history and check for function calls
            if assistant_message:
                # Add the original assistant message to the history first
                messages.append({
                    "role": "assistant", 
                    "content": assistant_message
                })
                
                # Check for function call tag
                if "<fnc_call>" in assistant_message:
                    start_tag = "<fnc_call>"
                    end_tag = "</fnc_call>"
                    start_idx = assistant_message.find(start_tag) + len(start_tag)
                    end_idx = assistant_message.find(end_tag)
                    
                    if start_idx > -1 and end_idx > -1:
                        json_str = assistant_message[start_idx:end_idx].strip()
                        try:
                            # Parse the function call JSON
                            function_call = json.loads(json_str)
                            
                            # Check if we have a direct function call or a list
                            if isinstance(function_call, list):
                                # Handle the first function call in the list
                                call = function_call[0]
                                function = call.get("function", {})
                            else:
                                # Direct function call object
                                function = function_call.get("function", {})
                            
                            if not isinstance(function, dict):
                                raise ValueError(f"Expected function to be a dict, got: {type(function)}")
                                
                            tool_name = function.get("name")
                            arguments = function.get("arguments", {})
                            
                            print(f"\n{bold_start}Tool Call{bold_end} " + "-" * 70)
                            
                            if tool_name:
                                # Execute the tool
                                tool_result = await mcp.call_tool(
                                    tool_name,
                                    arguments=arguments
                                )
                                print(f"\nTool result: {tool_result}")
                                
                                # Extract the text content from the tool result
                                result_text = ""
                                
                                # Try to extract text from the content field if available
                                if hasattr(tool_result, 'content') and tool_result.content:
                                    # If it's a list of content items
                                    if isinstance(tool_result.content, list):
                                        for content_item in tool_result.content:
                                            if hasattr(content_item, 'text'):
                                                result_text += content_item.text
                                    # If it's a single content item
                                    elif hasattr(tool_result.content, 'text'):
                                        result_text = tool_result.content.text
                                
                                # If we couldn't extract text content, fall back to string representation
                                if not result_text:
                                    result_text = str(tool_result)
                                
                                # Format the result according to the specified format
                                if hasattr(tool_result, 'isError') and tool_result.isError:
                                    formatted_result = f"<fnc_results>\n<error>{result_text}</error>\n</fnc_results>"
                                else:
                                    formatted_result = f"<fnc_results>\n{result_text}\n</fnc_results>"
                                
                                # Add anti-hallucination instructions
                                instruction = """IMPORTANT: 
1. Process the following tool results but DO NOT repeat the raw data verbatim in your response.
2. Analyze the information and provide a helpful, concise summary in a user-friendly format.
3. NEVER fabricate additional function calls or pretend to be making further API requests.
4. DO NOT hallucinate responses or create fictional scenarios based on the results.
5. If the tool returns an error or insufficient information, acknowledge this honestly rather than making up data."""

                                # Add the tool result as a system message
                                messages.append({
                                    "role": "system", 
                                    "content": formatted_result
                                })
                                
                                # Add a system message instructing the assistant to continue solving the problem
                                messages.append({
                                    "role": "system", 
                                    "content": "Continue solving the user's request autonomously based on these tool results. If the results indicate an error or unexpected outcome, fix your approach and try again."
                                })
                                
                                # Get LLM's response to the tool result
                                print(f"\n{bold_start}Assistant{bold_end} " + "-" * 70)
                                assistant_response = ""
                                async for chunk in stream_llm_response(
                                    messages=messages,
                                    model=model,
                                    api_key=api_key,
                                    api_base_url=api_base_url
                                ):
                                    # Process the raw LiteLLM chunk
                                    if hasattr(chunk, 'choices') and chunk.choices:
                                        # Extract the content from the choices
                                        delta = chunk.choices[0].delta
                                        if hasattr(delta, 'content') and delta.content:
                                            content = delta.content
                                            # Skip printing if assistant is repeating function results
                                            skip_print = False
                                            if content.strip().startswith("<fnc_results>") or content.strip().startswith("<fnc_results>"):
                                                skip_print = True
                                                
                                            if not skip_print:
                                                print(content, end="", flush=True)
                                            assistant_response += content
                                
                                # Add the tool response to message history
                                if assistant_response:
                                    # Update the assistant's response in the message history
                                    messages.append({
                                        "role": "assistant", 
                                        "content": assistant_response
                                    })
                                
                        except json.JSONDecodeError as e:
                            print(f"\nError parsing function data: {e}")
                        except Exception as e:
                            print(f"\nError executing function: {e}")
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    finally:
        # Clean up MCP
        await mcp.close()
    
    print("\nChat ended.")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
        asyncio.run(MCPHandler.get_instance().close())
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        asyncio.run(MCPHandler.get_instance().close())
