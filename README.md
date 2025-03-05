# nash-llm-server

A local server for LLM interactions with proper session handling and streaming responses.

## Quick Start

```bash
poetry install
poetry run llm_server  # Terminal 1
poetry run client_example  # Terminal 2
```

## Architecture

- **Stateless Server**: No conversation history stored server-side
- **Client-Side State**: Clients maintain message history and session IDs
- **Streaming**: Server-sent events with proper session handling

## Session ID Flow

1. **First Message**

   - Client sends request without session ID
   - Server generates new ID and sends it in first chunk
   - Client stores ID for future requests

2. **Subsequent Messages**
   - Client includes stored session ID
   - Server validates and maintains session continuity
   - Same ID returned in response

## Response Format

```
data: {"session_id": "uuid-here"}  # First chunk
data: {"content": "response text"}  # Content chunks
data: {"session_id": "uuid-here"}  # Last chunk
data: [DONE]
```

## Endpoints

1. `POST /v1/chat/completions/stream`

   ```json
   {
     "messages": [],
     "session_id": "optional",
     "model": "optional"
   }
   ```

2. `POST /v1/chat/summarize`
   ```json
   {
     "messages": [],
     "session_id": "optional"
   }
   ```

## Client Implementation Tips

1. **Session Management**

   - Store session ID from first response chunk
   - Verify against final chunk's ID
   - Pass ID in all subsequent requests

2. **Error Recovery**
   - Keep last known session ID
   - Can retry with same ID if connection drops
   - Server preserves ID even during errors

See `client_example.py` for a complete implementation.
