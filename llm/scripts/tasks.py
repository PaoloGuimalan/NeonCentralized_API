from celery import shared_task
from messenger.models import Message
from ..services.rag import CustomerServiceRAG


@shared_task
def index_chat_message_task(message_id, open_ai_api_key):
    print("Init Delay Index")
    message = Message.objects.get(message_id=message_id)
    rag = CustomerServiceRAG()

    sender_id = message.sender.id if message.sender else "ai_reply"

    rag.index_chat_message(
        sender_id,
        message.conversation.conversation_id,
        message.message_type,
        message.content,
        open_ai_api_key,
    )
