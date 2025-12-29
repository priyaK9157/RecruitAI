import os
import shutil
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import shared embeddings from your rag.py
from rag import dense_embeddings

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "hiring_assistant")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def ingest_folder(folder_path: str = None, doc_type: str = "resume"):
    path_to_process = folder_path if folder_path else DEFAULT_DATA_DIR
    print(f"🛠️ Starting ingestion from: {path_to_process}")
    
    # Initialize sparse model for keyword matching
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    try:
        if not os.path.exists(path_to_process):
            print(f"⚠️ Folder not found: {path_to_process}")
            return
        
        all_docs = []
        files = [f for f in os.listdir(path_to_process) if f.endswith((".pdf", ".txt"))]
        
        for file in files:
            file_path = os.path.join(path_to_process, file)
            try:
                loader = PyPDFLoader(file_path) if file.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata = {"source": file, "type": doc_type}
                all_docs.extend(loaded_docs)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

        if all_docs:
            chunks = splitter.split_documents(all_docs)
            collection_name = "hiring_assistant"
            print(f"🧠 Chunking complete. Indexing {len(chunks)} chunks...")
            
            # ✅ THE FIX: Pass connection details instead of the 'client' object
            # This allows the factory method to correctly set up Hybrid Search
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=dense_embeddings,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=True,
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
        # Cleanup temporary files
        if folder_path and "/tmp/" in folder_path:
            try:
                shutil.rmtree(folder_path)
                print(f"🧹 Cleaned up temporary directory: {folder_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup failed: {cleanup_error}")

if __name__ == "__main__":
    ingest_folder()
