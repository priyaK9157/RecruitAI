import os
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import shared components from rag.py to avoid locking issues
from rag import qdrant_client, embeddings, sparse_embeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

def ingest_folder(folder_path: str, doc_type: str):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    # --- FIX 1: Initialize documents list at the start ---
    all_docs = []
    
    files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".txt"))]
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            loader = PyPDFLoader(file_path) if file.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata = {"source": file, "type": doc_type}
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if all_docs:
        chunks = splitter.split_documents(all_docs)
        collection_name = "hiring_assistant"
        
        # --- FIX 2: Ensure collection exists with LangChain-compatible naming ---
        try:
            collections = qdrant_client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                print(f"Creating collection: {collection_name}")
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=1536, # Size for OpenAI text-embedding-3-small
                        distance=rest.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "langchain-sparse": rest.SparseVectorParams()
                    }
                )

            # --- FIX 3: Initialize with correct sparse_vector_name ---
            vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                sparse_vector_name="langchain-sparse",
                retrieval_mode=RetrievalMode.HYBRID
            )
            
            vectorstore.add_documents(chunks)
            print(f"✅ Successfully indexed {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"❌ Error during Qdrant indexing: {e}")
    else:
        print("No documents were found to index.")