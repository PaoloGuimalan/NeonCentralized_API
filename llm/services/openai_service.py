"""OpenAI, through the shared chat loop."""

from openai import OpenAI

from .base import ChatCompletionService


class OpenAIService(ChatCompletionService):

    def build_client(self, api_key):
        return OpenAI(api_key=api_key)
