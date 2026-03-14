import uuid
from datetime import datetime
from openai import OpenAI
import pinecone
from pinecone import ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neon import settings
import time


class CustomerServiceRAG:
    def __init__(self):
        # Static config only
        self.dimension = 1536
        self.index_name = settings.RAG["PINECONE_INDEX"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(settings.RAG.get("CHUNK_SIZE", 2000)), chunk_overlap=200
        )

        self.pc = pinecone.Pinecone(api_key=settings.RAG["PINECONE_API_KEY"])
        self.ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _get_index(self):
        """Helper to get Pinecone index using YOUR system key"""
        pc = pinecone.Pinecone(api_key=settings.RAG["PINECONE_API_KEY"])
        return pc.Index(self.index_name)

    def ensure_index(self):
        """Check if index exists; if not, create and wait until ready"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"🚀 Creating new Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            # Wait for Pinecone to report 'Ready' status
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("⏳ Waiting for index to initialize...")
                time.sleep(5)

            # Small extra buffer for DNS propagation
            time.sleep(10)
            print(f"✅ Index {self.index_name} is ready.")

    def get_embedding(self, text, api_key):
        """Fetches embedding using the USER'S OpenAI key"""
        client = OpenAI(api_key=api_key)
        # Note the [0] index to get the first result from the list
        response = client.embeddings.create(
            input=[text.replace("\n", " ")], model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def index_chat_message(
        self, user_id, conversationID, type, message, user_openai_key
    ):
        """Indexes a single chat message using user's key"""
        vector = self.get_embedding(message, user_openai_key)
        index = self._get_index()

        index.upsert(
            [
                {
                    "id": f"chat_{uuid.uuid4()}",
                    "values": vector,
                    "metadata": {
                        "type": type,
                        "user_id": str(user_id),
                        "conversation_id": str(conversationID),
                        "msg_type": type,
                        "text": message,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            ]
        )

    def bulk_index_docs(self, documents, user_openai_key):
        """Indexes company docs using user's key"""
        client = OpenAI(api_key=user_openai_key)
        index = self._get_index()
        all_vectors = []

        for doc in documents:
            chunks = self.splitter.split_text(doc)

            # Batch call for all chunks in a document (Efficient)
            response = client.embeddings.create(
                input=chunks, model="text-embedding-3-small"
            )

            for i, (chunk, enc) in enumerate(zip(chunks, response.data)):
                all_vectors.append(
                    {
                        "id": f"doc_{uuid.uuid4()}",
                        "values": enc.embedding,
                        "metadata": {
                            "type": "doc",
                            "text": chunk,
                            "source": doc[:100],
                            "chunk_id": i,
                        },
                    }
                )

        # Batch upsert to Pinecone
        for i in range(0, len(all_vectors), 100):
            index.upsert(vectors=all_vectors[i : i + 100])

    def retrieve(self, query, conversationID, user_openai_key, top_k=5):
        """Retrieves context using user's key for the query embedding"""
        query_vec = self.get_embedding(query, user_openai_key)
        index = self._get_index()

        filters = {"$or": [{"type": "doc"}, {"conversation_id": str(conversationID)}]}

        results = index.query(
            vector=query_vec, top_k=30, filter=filters, include_metadata=True
        )
        reranked = self.pc.inference.rerank(
            model=settings.RAG["RERANKER_MODEL"],
            query=query,
            documents=[match["metadata"] for match in results["matches"]],
            top_n=top_k,
        )

        documents = [item.document for item in reranked.data]

        return documents
