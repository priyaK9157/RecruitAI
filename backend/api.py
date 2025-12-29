import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
app = FastAPI()

origins = [
    "https://recruitai-xguc3nypm6ujpcluzm8dhy.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "alive", "message": "Backend is running!"}

@app.post("/upload")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    from ingest import ingest_folder
    
    # Create a unique temp directory for this upload batch
    # This prevents files from disappearing or mixing between users
    temp_dir = tempfile.mkdtemp()
    
    try:
        for file in files:
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb+") as f_obj:
                shutil.copyfileobj(file.file, f_obj)
        
        background_tasks.add_task(ingest_folder, temp_dir, "resume")
        
        return {"message": f"Successfully received {len(files)} files. Indexing in progress."}
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(question: Question):
    from rag import retrieve
    
    data = retrieve(question.query)
    context = data.get("context", "")
    
    if not context.strip():
        # Helpful for debugging Render issues
        return {
            "answer": "I couldn't find any information in the database. Please try uploading your resumes again.", 
            "sources": []
        }

    system_instruction = (
        "You are an Expert Technical Recruiter assistant. "
        "Answer using ONLY the provided context. Identify candidates by full name."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY: {question.query}"}
            ],
            temperature=0
        )
        return {"answer": response.choices[0].message.content, "sources": data["sources"]}
    except Exception as e:
        return {"answer": f"Generation Error: {str(e)}", "sources": []}
