"""System prompts used by the LLM server."""

CHAT_SYSTEM_PROMPT = "You are a helpful AI assistant."

SUMMARIZE_SYSTEM_PROMPT = """
Summarize the key points of this conversation while preserving important 
context. Focus on maintaining:
1. Essential information exchanged
2. Important decisions or conclusions
3. Current context needed for continuation
Be concise but ensure no critical details are lost.
"""
