# RecruitAI

**RecruitAI** is an AI-powered recruitment assistant that helps you analyze resumes and answer queries about candidates efficiently. It combines a **Streamlit frontend** with a **FastAPI backend** and leverages **OpenAI's GPT models** for smart candidate insights.

---

## Features

- Upload multiple resumes (PDF, TXT) and add them to a searchable knowledge base.
- Ask questions about candidates, e.g., "Who has the most Python experience?" or "Summarize Jane’s profile."
- View answers along with sources from uploaded documents.
- Clean, hand-coded, modern UI with a sidebar workspace and chat interface.
- Reset chat and manage uploaded files easily.

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database/Indexing**: Custom ingestion pipeline for documents (`ingest.py`) and retrieval logic (`rag.py`)
- **AI Model**: OpenAI GPT-4o-mini for contextual chat
- **Environment**: `.env` for sensitive config (e.g., `BACKEND_URL`)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/priyaK9157/RecruitAI.git
   cd RecruitAI
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies for backend and frontend**

   ```bash
   cd backend
   pip install -r requirements.txt
   cd ../frontend
   pip install -r requirements.txt
   cd ..
   ```

4. **Configure environment variables**  
   Create a `.env` file in the **root of the project**:

   ```env
   BACKEND_URL=http://localhost:8000
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=https://your-project-name-xxxxx.qdrant.cloud
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

> **Note:** Both frontend and backend read these variables, so keep `.env` in the root.

---

## Usage

### 1. Start the backend

```bash
cd backend
uvicorn app:app --reload
```

- The backend exposes two main endpoints:
  * `/upload` → Upload resumes to the knowledge base
  * `/ask` → Query candidate information

### 2. Start the frontend

Open a **new terminal**:

```bash
cd frontend
streamlit run frontend_app.py
```

- Upload resumes via the sidebar.
- Use the chat interface to ask questions about the uploaded candidates.
- Responses include AI-generated insights and reference sources.

> **Tip:** Keep both terminals open while running so the frontend can communicate with the backend.

---

## Project Structure

```
recruitai/
│
├─ backend/
│   ├─ app.py            # FastAPI backend
│   ├─ ingest.py         # Resume ingestion logic
│   ├─ rag.py            # Retrieval-Augmented Generation logic
│   └─ requirements.txt  # Backend dependencies
│
├─ frontend/
│   ├─ frontend_app.py   # Streamlit frontend
│   └─ requirements.txt  # Frontend dependencies
│
├─ .env                  # Environment variables
└─ README.md
```

---

## Example Usage

1. Upload resumes (`.pdf` or `.txt`) via the sidebar.
2. Ask questions like:
   - "Who has 5+ years of Python experience?"
   - "Summarize John's resume."
3. Receive AI-generated answers with document sources.

---

## Notes

- The frontend assumes the backend is running at the URL specified in `BACKEND_URL`.
- Ensure your OpenAI API key is set for AI-powered responses.
- Both frontend and backend must be running simultaneously for full functionality.
