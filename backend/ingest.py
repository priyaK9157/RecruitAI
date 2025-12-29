import os
import shutil
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Using Absolute Import (No dots)
from rag import qdrant_client, dense_embeddings

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "hiring_assistant")

# Configure text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def ingest_folder(folder_path: str = None, doc_type: str = "resume"):
    """
    Reads files from a folder, splits them into chunks, and indexes them into Qdrant.
    Cleans up the folder after processing if it's a temporary directory.
    """
    path_to_process = folder_path if folder_path else DEFAULT_DATA_DIR
    print(f"🛠️ Starting ingestion from: {path_to_process}")
    
    # 1. Initialize sparse model (Lazy load inside function to save RAM on start)
    # This model helps with keyword matching (e.g., "Python", "Docker")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    try:
        if not os.path.exists(path_to_process):
            print(f"⚠️ Folder not found: {path_to_process}")
            return
        
        all_docs = []
        files = [f for f in os.listdir(path_to_process) if f.endswith((".pdf", ".txt"))]
        print(f"📂 Files found: {files}")
        
        # 2. Load Documents
        for file in files:
            file_path = os.path.join(path_to_process, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                    
                loaded_docs = loader.load()
                
                # Add metadata for better filtering/tracking
                for doc in loaded_docs:
                    doc.metadata = {"source": file, "type": doc_type}
                
                all_docs.extend(loaded_docs)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

        # 3. Process and Index
        if all_docs:
            chunks = splitter.split_documents(all_docs)
            collection_name = "hiring_assistant"
            
            print(f"🧠 Chunking complete. Indexing {len(chunks)} chunks...")
            
            # Hybrid Search combines Vector (semantic) + BM25 (keyword) search
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=dense_embeddings,
                client=qdrant_client,
                collection_name=collection_name,
                sparse_embedding=sparse_embeddings,
                sparse_vector_name="langchain-sparse",
                retrieval_mode=RetrievalMode.HYBRID,
            )
            print(f"✅ Successfully indexed into Qdrant.")
        else:
            print("⚠️ No readable documents found to index.")

    except Exception as e:
        print(f"❌ Critical Ingestion Error: {e}")

    finally:
        # 4. Mandatory Cleanup
        # If the path is a temporary directory created by api.py, delete it
        if folder_path and "/tmp/" in folder_path:
            try:
                shutil.rmtree(folder_path)
                print(f"🧹 Cleaned up temporary directory: {folder_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup failed for {folder_path}: {cleanup_error}")

# This allows you to run ingestion manually for local testing
if __name__ == "__main__":
    ingest_folder()
