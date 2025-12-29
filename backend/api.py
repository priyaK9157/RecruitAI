import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 1. Import this
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
app = FastAPI()

# --- 2. ADD THIS CORS SECTION ---
origins = [
    "https://recruitai-xguc3nypm6ujpcluzm8dhy.streamlit.app",
    "http://localhost:8501", # Useful for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------

class Question(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "alive", "message": "Backend is running!"}

@app.post("/upload")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    # Lazy Import inside the function
    from .ingest import ingest_folder
    
    upload_path = "../data/hiring_assistant"
    os.makedirs(upload_path, exist_ok=True)
    
    try:
        for file in files:
            file_location = os.path.join(upload_path, file.filename)
            with open(file_location, "wb+") as f_obj:
                shutil.copyfileobj(file.file, f_obj)
        
        background_tasks.add_task(ingest_folder, upload_path, "hiring_assistant")
        return {"message": "Files uploaded. Indexing started in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(question: Question):
    # Lazy Import inside the function
    from .rag import retrieve
    
    # 1. Retrieve the top context chunks
    data = retrieve(question.query)
    context = data.get("context", "")
    
    if not context.strip():
        return {"answer": "I'm sorry, I couldn't find any relevant information in the uploaded resumes.", "sources": []}

    system_instruction = (
        "You are an Expert Technical Recruiter assistant. "
        "Your goal is to answer questions using ONLY the provided context. "
        "Be very specific. If a candidate mentions a specific tool (like FastAPI, React, or SQL), "
        "mention it by name. Always identify candidates by their full names. "
        "If the context contains multiple resumes, compare them clearly."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"CONTEXT FROM RESUMES:\n{context}\n\nUSER QUERY: {question.query}"}
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content, 
            "sources": data["sources"]
        }
    except Exception as e:
        return {"answer": f"Error during generation: {str(e)}", "sources": []}
