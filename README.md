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

## API Endpoints

### 1. Stream Chat Completions

`POST /v1/chat/completions/stream`

Stream chat completions from the LLM with server-sent events.

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "model": "gpt-4-turbo-preview", // Required
  "api_key": "sk-...", // Required
  "api_base_url": "https://api.openai.com/v1", // Required
  "session_id": "optional-uuid" // Optional
}
```

#### Response Format

```
data: {"session_id": "uuid-here"}  # First chunk
data: {"content": "response text"}  # Content chunks
data: {"session_id": "uuid-here"}  # Last chunk
data: [DONE]
```

### 2. Summarize Conversation

`POST /v1/chat/summarize`

Summarize a conversation to reduce token count while preserving context.

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    }
  ],
  "model": "gpt-4-turbo-preview", // Required
  "api_key": "sk-...", // Required
  "api_base_url": "https://api.openai.com/v1", // Required
  "session_id": "optional-uuid" // Optional
}
```

#### Response Format

```json
{
  "success": true,
  "summary": "Conversation summary text",
  "messages": [
    {
      "role": "assistant",
      "content": "Previous conversation summary:\n[summary text]"
    }
  ],
  "token_reduction": {
    "before": 100,
    "after": 50
  },
  "session_id": "uuid-here"
}
```

## Provider Support

The server supports multiple LLM providers through their respective base URLs:

- OpenAI: `https://api.openai.com/v1`
- Anthropic: `https://api.anthropic.com`
- Helicone OpenAI: `https://oai.helicone.ai/v1`
- Helicone Anthropic: `https://anthropic.helicone.ai/`

## Client Implementation Tips

1. **Session Management**

   - Store session ID from first response chunk
   - Verify against final chunk's ID
   - Pass ID in all subsequent requests

2. **Error Recovery**

   - Keep last known session ID
   - Can retry with same ID if connection drops
   - Server preserves ID even during errors

3. **API Configuration**
   - Always provide model, api_key, and api_base_url
   - Use appropriate base URL for your provider
   - Consider using Helicone for observability

See `client_example.py` for a complete implementation.
