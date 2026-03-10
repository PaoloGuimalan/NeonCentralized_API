import pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neon import settings
import uuid
from datetime import datetime
import time


class CustomerServiceRAG:
    def __init__(self):
        self.model = SentenceTransformer(settings.RAG["EMBEDDING_MODEL"])
        self.dimension = int(settings.RAG["DIMENSION"])
        self.index_name = settings.RAG["PINECONE_INDEX"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(settings.RAG["CHUNK_SIZE"]), chunk_overlap=200
        )
        self.pc = None

        try:
            self.ensure_index()
        except Exception as e:
            print(f"⚠️ RAG init warning: {e}")
            # Index will be created on first use

    def ensure_index(self):
        """Create index if doesn't exist"""
        self.pc = pinecone.Pinecone(api_key=settings.RAG["PINECONE_API_KEY"])
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(30)  # Wait for creation

        self.index = self.pc.Index(self.index_name)

    def bulk_index_docs(self, documents):
        """Index company documents (run once)"""
        all_vectors = []
        for doc in documents:
            chunks = self.splitter.split_text(doc)
            vectors = self.model.encode(chunks)

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                all_vectors.append(
                    {
                        "id": f"doc_{uuid.uuid4()}",
                        "values": vector.tolist(),
                        "metadata": {
                            "type": "doc",
                            "text": chunk,
                            "source": doc,
                            "chunk_id": i,
                        },
                    }
                )

        # Bulk upsert
        for i in range(0, len(all_vectors), 500):
            batch = all_vectors[i : i + 500]
            self.index.upsert(vectors=batch)
        print(f"✅ Indexed {len(all_vectors)} chunks")

    def index_chat_message(self, user_id, conversationID, type, message):
        print("Processing")
        """Index user chat (async)"""
        vector = self.model.encode([message])[0].tolist()
        self.index.upsert(
            [
                {
                    "id": f"chat_{user_id}_{uuid.uuid4()}",
                    "values": vector,
                    "metadata": {
                        "type": "chat",
                        "user_id": str(user_id),
                        "conversation_id": str(conversationID),
                        "type": type,
                        "text": message,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            ]
        )

    def retrieve(self, query, conversationID, top_k):
        if self.index is None:
            self.ensure_index()

        """Core retrieval - docs + user history"""
        query_vec = self.model.encode([query])[0].tolist()

        filters = {"type": "doc"}
        if conversationID:
            filters = {
                "$or": [
                    {"conversation_id": str(conversationID)},
                ]
            }

        results = self.index.query(
            vector=query_vec, top_k=top_k, filter=filters, include_metadata=True
        )

        return [match["metadata"] for match in results["matches"]]
