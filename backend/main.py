from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from rag_pipeline import RAGPipeline
from typing import List, Union
import os
from pathlib import Path
from dotenv import load_dotenv

# -- Set up and Configuration --
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# -- Set up directory for uploaded PDFs --
UPLOAD_DIR = "uploads"

# -- Initialize FastAPI app --
app = FastAPI()

# -- Allow frontend to access the backend --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()

# Endpoint to upload PDFs (one or many)
@app.post("/upload_pdf/")
async def upload_pdf(files: Union[List[UploadFile], UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if isinstance(files, UploadFile):
        files = [files]

    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")

    saved_files = []
    total_chunks = 0

    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)

        # Stream-write file (efficient for large PDFs)
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB at a time
                f.write(chunk)

        saved_files.append(save_path)

        # Extract and index the text
        text = rag_pipeline.extract_text_from_pdf(save_path)
        chunks = rag_pipeline.build_vector_store(text)
        total_chunks += chunks

    return {
        "message": f"Uploaded {len(files)} PDF{'s' if len(files) > 1 else ''} successfully!",
        "saved_files": saved_files,
        "total_chunks": total_chunks
    }

# Endpoint to ask a question
@app.post("/ask/")
async def ask_question(question: str = Form(...), api_key: str = Form(None)):
    # Allow runtime override of Google API key
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    try:
        answer = rag_pipeline.query(question)
        return answer
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Endpoint to get uploaded PDF
@app.get("/get_pdf/{pdf_name}")
async def get_pdf(pdf_name: str):
    file_path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path=file_path, filename=pdf_name, media_type="application/pdf")

# Endpoint to root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini RAG Backend API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)