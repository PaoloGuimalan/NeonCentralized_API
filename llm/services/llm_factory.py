from .groq_service import GroqService
from .openai_service import OpenAIService


class LLMFactory:

    @classmethod
    def create(cls, service, api_key, model):
        if service == "Groq":
            return GroqService(api_key, model)
        elif service == "OpenAI":
            return OpenAIService(api_key, model)
        else:
            return None
