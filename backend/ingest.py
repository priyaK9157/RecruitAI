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
    # Use default path if none provided
    path_to_process = folder_path if folder_path else DEFAULT_DATA_DIR
    
    print(f"🛠️ Starting ingestion from: {path_to_process}")
    
    # 1. Initialize sparse model (Lazy load inside function to save RAM on start)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    if not os.path.exists(path_to_process):
        os.makedirs(path_to_process, exist_ok=True)
        print(f"⚠️ Folder was missing, created it: {path_to_process}")
        return
    
    all_docs = []
    files = [f for f in os.listdir(path_to_process) if f.endswith((".pdf", ".txt"))]
    print(f"📂 Files found in directory: {files}")
    
    for file in files:
        file_path = os.path.join(path_to_process, file)
        try:
            # Better PDF Loading for Render
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                
            loaded_docs = loader.load()
            
            if not loaded_docs or all(not d.page_content.strip() for d in loaded_docs):
                print(f"⚠️ Warning: {file} is empty or unreadable (scanned?).")
                continue
                
            for doc in loaded_docs:
                doc.metadata = {"source": file, "type": doc_type}
            all_docs.extend(loaded_docs)
            
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    if all_docs:
        chunks = splitter.split_documents(all_docs)
        collection_name = "hiring_assistant"
        
        try:
            # Sync with rag.py configuration
            QdrantVectorStore.from_documents(
                chunks,
                dense_embeddings,
                client=qdrant_client,
                collection_name=collection_name,
                sparse_embedding=sparse_embeddings,
                sparse_vector_name="langchain-sparse", # Must match retrieve()
                retrieval_mode=RetrievalMode.HYBRID,
            )
            print(f"✅ Indexed {len(chunks)} chunks into Qdrant.")
            
        except Exception as e:
            print(f"❌ Qdrant Indexing Failed: {e}")
    else:
        print("❌ Ingestion aborted: No readable documents found.")
