from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag_pipeline import RAGPipeline
from typing import List, Union
import arxiv
import os
import json
import requests
import datetime
from pydantic import BaseModel


# -- Set up and Configuration --
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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

# -- Initialize instances and client --
rag_pipeline = RAGPipeline()
arxiv_client = arxiv.Client()

# --  Date Model For Response --
class Paper(BaseModel):
    title: str
    date: str
    authors: List[str]
    categories: List[str]
    pdf: str
    url: str
    summary: str

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

# Endpoint to ask a question and get a streaming response
@app.post("/ask/")
async def ask_question(question: str = Form(...), api_key: str = Form(None)):
    # Allow runtime override of Google API key
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    try:
        # Stream the response
        def iterfile():
            for chunk in rag_pipeline.stream_query(question):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(iterfile(), media_type="application/x-ndjson")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Endpoint to search arxiv research paper
@app.get("/search/", response_model=List[Paper])
async def search_papers(query: str, category: str = "", sort_by: str=  "relevance", days_back: int = 0, limit: int = 5):
    search_parts = [f"{query}"] if query else []

    if category:
        search_parts.append(f"cat:{category}")
    
    search_query = " AND ".join(search_parts) if search_parts else "all:all"

    # Handle Date Filter
    if days_back > 0:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        date_str = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO 209912312359]"
        search_query = f"({search_query}) AND {date_str}"
    
    criterion = (arxiv.SortCriterion.Relevance if sort_by == "relevance" 
                 else arxiv.SortCriterion.SubmittedDate)
    
    search = arxiv.Search(
        query=search_query,
        max_results=limit,
        sort_by=criterion,
        sort_order=arxiv.SortOrder.Descending
    )

    results = list(arxiv_client.results(search))
    return [
        Paper(
            title=r.title,
            date=r.published.strftime("%Y-%m-%d"),
            authors=[a.name for a in r.authors],
            categories=r.categories,
            pdf=r.pdf_url,
            url=r.entry_id,
            summary=r.summary,
        ) for r in results
    ]

# Endpoint to ingest PDF from URL
@app.post("/ingest-from-url/")
async def ingest_from_url(url: str, title: str):
    try:
        # Clean filename from title
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{clean_title.replace(' ', '_')}.pdf"
        save_path = os.path.join(UPLOAD_DIR, filename)

        # Download the PDF from ArXiv (or any URL)
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    f.write(chunk)

        text = rag_pipeline.extract_text_from_pdf(save_path)
        num_chunks = rag_pipeline.build_vector_store(text)

        return {
            "status": "success",
            "message": f"Successfully ingested '{title}'",
            "saved_file": save_path,
            "total_chunks": num_chunks
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# Endpoint to root
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Document Chatbot Backend API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
