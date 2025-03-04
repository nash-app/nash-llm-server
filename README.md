# nash-llm-server

A local server for LLM interactions that prioritizes:

- **Control**: Your data stays on your machine
- **Simplicity**: Minimal setup, clear interface
- **Reliability**: Proper token management and error handling
- **Flexibility**: Works with various LLM providers via LiteLLM

## Architecture

The server is intentionally stateless - it maintains no conversation history. Instead, clients pass their full conversation history with each request. This design:

- Enables multiple clients without server-side session management
- Gives clients complete control over their conversation history
- Simplifies horizontal scaling and error recovery
- Allows conversations to survive server restarts

## Components

### Server (`app/llm_server.py`)

Runs on `localhost:8001` with two endpoints:

- `POST /v1/chat/completions/stream`: Main chat endpoint with streaming responses
- `POST /v1/chat/summarize`: Summarizes long conversations

Features:

- Handles token limits and conversation length
- Uses LiteLLM for model compatibility
- Integrates with Helicone for request tracking

### Example Client (`client_example.py`)

- Interactive chat interface with streaming responses
- Maintains conversation history (user and assistant messages)
- Two modes:
  1. Single prompt: `client_example "Your question here"`
  2. Interactive chat: `client_example`
- Commands: `exit`, `summarize`

## Getting Started

```bash
poetry install
poetry run llm_server
```

And then another terminal tab:

```bash
poetry run client_example "Write me 5 paragraphs about the sky"
```

OR

```bash
poetry run client_example
```

## Implementing Alternative Clients

While this repository includes a Python client for demonstration purposes, the server is designed to work with any client implementation. Here's what you need to know when implementing your own client:

### Session Handling

The server uses Helicone for request tracking and analytics. Session handling is managed entirely by the server to ensure consistency:

1. **Initial Request**:

   - Client does not need to generate or provide a session ID
   - Server generates a new session ID if none provided
   - Server sends session ID in two places:
     1. First chunk: `{"session_id": "uuid-here"}`
     2. Last chunk before `[DONE]`: `{"session_id": "uuid-here"}`

2. **Subsequent Requests**:

   - Client should store and reuse the session ID from previous responses
   - Pass the same session ID in subsequent requests to maintain conversation continuity
   - Server will validate and return the same session ID

3. **Response Protocol**:

   ```
   data: {"session_id": "uuid-here"}
   data: {"content": "first part of response"}
   data: {"content": "more response text"}
   ...
   data: {"session_id": "uuid-here"}
   data: [DONE]
   ```

4. **Error Handling**:
   - If connection drops, client can retry with last known session ID
   - Server preserves session ID even when returning warnings/errors
   - Summarization endpoint also maintains session continuity

### Best Practices for Client Implementation

1. **Session Management**:

   - Store session ID from first response chunk
   - Verify it matches final chunk's session ID
   - Reuse session ID for all requests in the conversation
   - Pass session ID in both chat and summarize requests

2. **Stream Processing**:

   - Handle different chunk types appropriately:
     - `session_id`: Update stored session ID
     - `content`: Accumulate response text
     - `warning`: Handle length/token warnings
     - `error`: Handle error messages
   - Always process until `[DONE]` marker

3. **Error Recovery**:
   - Keep last known session ID for reconnection
   - Handle connection drops gracefully
   - Maintain message history on client side

### Endpoints

The server exposes two endpoints at `localhost:8001`:

1. `POST /v1/chat/completions/stream`

   - Request body: `{"messages": [], "model": "optional", "session_id": "optional"}`
   - Returns: Server-sent events stream

2. `POST /v1/chat/summarize`
   - Request body: `{"messages": [], "session_id": "optional"}`
   - Returns: JSON with summary and new session ID

### Best Practices

1. Always capture and reuse the session ID returned by the server
2. Handle server-sent events appropriately for streaming responses
3. Maintain message history on the client side (server is stateless)
4. Implement summarization when conversation length exceeds limits
5. Handle warnings and errors returned in the response format
