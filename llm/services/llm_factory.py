from .groq_service import GroqService


class LLMFactory:

    @classmethod
    def create(cls, service, api_key, model):
        if service == "Groq":
            return GroqService(api_key, model)
        else:
            return None
