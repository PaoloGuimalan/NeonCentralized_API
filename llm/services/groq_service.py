"""Groq, through the shared chat loop.

Groq's client mirrors OpenAI's `chat.completions.create` surface closely
enough - including `tools` / `tool_calls` and streaming deltas - that the only
provider-specific thing left here is how the client is built.
"""

from groq import Groq

from .base import ChatCompletionService


class GroqService(ChatCompletionService):

    def build_client(self, api_key):
        return Groq(api_key=api_key)
