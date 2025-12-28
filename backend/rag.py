import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "../qdrant_storage")

# Create ONE global client instance
# qdrant_client = QdrantClient(path=DB_DIR)
# From this: qdrant_client = QdrantClient(path=DB_DIR)
# To this:
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

def retrieve(query, k=10):
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == "hiring_assistant" for c in collections):
            return {"context": "", "sources": []}

        vectorstore = QdrantVectorStore(
            client=qdrant_client, # Use the shared client
            collection_name="hiring_assistant",
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        results = vectorstore.similarity_search(query, k=k)
        context = "\n---\n".join([d.page_content for d in results])
        sources = list(set([d.metadata.get("source", "Unknown") for d in results]))
        return {"context": context, "sources": sources}
    except Exception as e:
        return {"error": str(e), "context": "", "sources": []}