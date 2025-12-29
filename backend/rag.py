import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv

load_dotenv()

# Global client and dense embeddings are fine (they don't download large local files)
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- REMOVED sparse_embeddings from here ---

def retrieve(query, k=10):
    try:
        # 1. Lazy Load the sparse embeddings INSIDE the function
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        collections = qdrant_client.get_collections().collections
        if not any(c.name == "hiring_assistant" for c in collections):
            return {"context": "", "sources": []}

        # 2. Use the local sparse_embeddings here
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name="hiring_assistant",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        results = vectorstore.similarity_search(query, k=k)
        context = "\n---\n".join([d.page_content for d in results])
        sources = list(set([d.metadata.get("source", "Unknown") for d in results]))
        return {"context": context, "sources": sources}
    except Exception as e:
        return {"error": str(e), "context": "", "sources": []}
