import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .rag import qdrant_client, dense_embeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def ingest_folder(folder_path: str, doc_type: str):
    # 1. Initialize sparse model
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    all_docs = []
    files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".txt"))]
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            loader = PyPDFLoader(file_path) if file.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            
            # Verify if text was actually found
            if not loaded_docs or all(not d.page_content.strip() for d in loaded_docs):
                print(f"⚠️ Warning: No text found in {file}. It might be a scanned image.")
                continue
                
            for doc in loaded_docs:
                doc.metadata = {"source": file, "type": doc_type}
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    if all_docs:
        chunks = splitter.split_documents(all_docs)
        collection_name = "hiring_assistant" # <--- ENSURE THIS MATCHES RAG.PY
        
        try:
            # 2. Use from_documents for a cleaner, unified ingestion
            # This handles the collection creation and sparse/dense config automatically
            QdrantVectorStore.from_documents(
                chunks,
                dense_embeddings,
                client=qdrant_client,
                collection_name=collection_name,
                sparse_embedding=sparse_embeddings,
                sparse_vector_name="langchain-sparse",
                retrieval_mode=RetrievalMode.HYBRID,
            )
            print(f"✅ Successfully indexed {len(chunks)} chunks into {collection_name}.")
            
        except Exception as e:
            print(f"❌ Error during Qdrant indexing: {e}")
    else:
        print("No readable documents were found to index.")
