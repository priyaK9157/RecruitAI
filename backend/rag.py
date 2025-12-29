import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
except Exception as e:
    print(f"Warning: Could not pre-load sparse embeddings: {e}")
    sparse_embeddings = None

def retrieve(query, k=5): # Reduced k to 5 to save tokens/memory on Render
    try:
        # Check if collection even exists before searching
        collections = qdrant_client.get_collections().collections
        if not any(c.name == "hiring_assistant" for c in collections):
            return {"context": "", "sources": [], "error": "Knowledge base is empty. Please upload files."}

        # Initialize VectorStore
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name="hiring_assistant",
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        # Perform Search
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            return {"context": "", "sources": []}

        context = "\n---\n".join([d.page_content for d in results])
        sources = list(set([d.metadata.get("source", "Unknown") for d in results]))
        
        return {"context": context, "sources": sources}
        
    except Exception as e:
        # Return the actual error to the frontend for debugging
        return {"error": str(e), "context": "", "sources": []}
