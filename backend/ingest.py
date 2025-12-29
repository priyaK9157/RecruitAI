import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import qdrant_client, dense_embeddings

# Use Absolute Paths for Render
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "hiring_assistant")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def ingest_folder(folder_path: str = None, doc_type: str = "resume"):
    path_to_process = folder_path if folder_path else DEFAULT_DATA_DIR
    print(f"🛠️ Starting ingestion from: {path_to_process}")
    
    # Lazy load sparse model to save RAM
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    if not os.path.exists(path_to_process):
        os.makedirs(path_to_process, exist_ok=True)
        return
    
    all_docs = []
    files = [f for f in os.listdir(path_to_process) if f.endswith((".pdf", ".txt"))]
    print(f"📂 Files found: {files}")
    
    for file in files:
        file_path = os.path.join(path_to_process, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata = {"source": file, "type": doc_type}
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    if all_docs:
        chunks = splitter.split_documents(all_docs)
        collection_name = "hiring_assistant"
        
        try:
            # ✅ FIX 2: Correct initialization for Hybrid Search
            # We use the existing client object from rag.py
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=dense_embeddings, # Dense
                client=qdrant_client,       # Our pre-configured client
                collection_name=collection_name,
                sparse_embedding=sparse_embeddings, # Sparse
                sparse_vector_name="langchain-sparse",
                retrieval_mode=RetrievalMode.HYBRID,
            )
            print(f"✅ Indexed {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"❌ Qdrant Indexing Failed: {e}")
            # If 'client' still fails, it's a version conflict. 
            # Check your requirements.txt for langchain-qdrant==0.2.0
