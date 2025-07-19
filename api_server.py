from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List
import shutil
import os
from datetime import datetime
import sys

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
# Directly import QuillAILLM and RetrievalAugmentor for type hinting
from quillai_llm import QuillAILLM
from retrieval import RetrievalAugmentor
from knowledgebase import KnowledgeBaseManager

# Import functions from app.py
from app import process_api_query, initialize_api_components

# Import the new caching system
from response_cache import DualResponseCache

# Initialize FastAPI app
app = FastAPI(
    title="QuillAI Enhanced Academic Assistant API",
    description="Intelligent academic assistant with dual output system for Flutter integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated Pydantic models for Flutter compatibility
class QueryRequest(BaseModel):
    prompt: str  # Frontend sends "prompt" instead of "query"
    mode: int    # Frontend sends 0 (learning) or 1 (question)
    marks: int   # Frontend sends 0, 2, 5, or 10

    @validator('mode')
    def validate_mode(cls, v):
        if v not in [0, 1]:
            raise ValueError('Mode must be 0 (learning) or 1 (question)')
        return v

    @validator('marks')
    def validate_marks(cls, v):
        if v not in [0, 2, 5, 10]:
            raise ValueError('Marks must be 0, 2, 5, or 10')
        return v

    def to_internal_format(self):
        """Convert frontend format to internal format"""
        return {
            'query': self.prompt,  # Map prompt to query
            'mode': 'learning' if self.mode == 0 else 'question',  # Convert int to string
            'marks': self.marks if self.marks > 0 else None  # Convert 0 to None for learning mode
        }

# Pydantic models for additional endpoints
class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    error_message: Optional[str] = None

class SystemStatsResponse(BaseModel):
    knowledge_base: dict
    retrieval_index: dict
    timestamp: str

# Global instances for LLM, Retrieval System, and Cache
llm_instance: Optional[QuillAILLM] = None
retrieval_instance: Optional[RetrievalAugmentor] = None
cache_instance: Optional[DualResponseCache] = None # NEW: Cache instance

@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    global llm_instance, retrieval_instance, cache_instance # Declare global to modify them
    try:
        print("🚀 Starting QuillAI API Server...")
        print("🤖 Initializing AI components (this may take a moment)...")

        # Initialize components once and store them globally
        llm_instance, retrieval_instance = initialize_api_components()

        # NEW: Initialize cache
        cache_instance = DualResponseCache()

        print("✅ QuillAI API Server ready!")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔍 ReDoc Documentation: http://localhost:8000/redoc")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # Re-raise the exception to prevent the server from starting in a broken state
        raise

# Main query endpoint - the heart of your API
@app.post("/query", response_model=dict)
async def process_query(request: QueryRequest):
    """
    Process academic queries and return dual outputs (LLM + Custom).

    This is the main endpoint your Flutter app will call.
    """
    # Ensure components are initialized before processing requests
    if llm_instance is None or retrieval_instance is None or cache_instance is None:
        raise HTTPException(status_code=503, detail="AI components not initialized. Server is still starting up or failed to initialize.")

    from types import SimpleNamespace
    internal_query_obj = SimpleNamespace(**request.to_internal_format())
    query_text = internal_query_obj.query
    mode_str = internal_query_obj.mode
    marks_val = internal_query_obj.marks

    # NEW: Check cache first
    cached_response = cache_instance.get(query_text, mode_str, marks_val)
    if cached_response:
        print(f"Cache hit for query: {query_text[:50]}...")
        return cached_response

    try:
        print(f"📝 Processing query: {query_text[:50]}...")

        # Validate request
        if not query_text or not query_text.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Process the query, passing the initialized instances
        # process_api_query now returns the full dual response dict
        result = process_api_query(internal_query_obj, llm_instance, retrieval_instance)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error_message"])

        # Remap output keys to match frontend expectations
        frontend_response = {
            "success": result["success"],
            "dialogpt_output": result.get("llm_output", ""), # Renamed for consistency
            "custom_llm": result.get("custom_output", ""),
            "intent": result.get("intent", ""),
            "domain": result.get("domain", ""),
            "topics": result.get("topics", []),
            "context_used": result.get("context_used", False),
            "generation_time": result.get("generation_time", 0),
            "word_count": result.get("word_count", 0),
            "timestamp": result.get("timestamp", "")
        }

        # NEW: Cache the successful response
        cache_instance.set(query_text, mode_str, marks_val, frontend_response)

        # Log successful request
        print(f"Successful query: {query_text[:100]}, Intent: {result.get('intent', 'unknown')}, Domain: {result.get('domain', 'unknown')}")

        return frontend_response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# PDF upload endpoint
@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_ocr: bool = False,
    language: str = "eng"):
    """
    Upload and process PDF files for the knowledge base.
    """
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            return PDFUploadResponse(
                success=False,
                error_message="File must be a PDF"
            )

        # Save uploaded file to /tmp
        upload_dir = "/tmp/uploaded_pdfs" # Changed to /tmp
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF in background
        background_tasks.add_task(
            process_uploaded_pdf,
            file_path,
            file.filename,
            use_ocr,
            language
        )

        return PDFUploadResponse(
            success=True,
            message=f"PDF '{file.filename}' uploaded successfully and is being processed",
            filename=file.filename
        )

    except Exception as e:
        return PDFUploadResponse(
            success=False,
            error_message=f"Upload failed: {str(e)}"
        )

def process_uploaded_pdf(file_path: str, filename: str, use_ocr: bool, language: str):
    """Background task to process uploaded PDF."""
    try:
        # Initialize knowledge base with /tmp storage
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/textbooks") # Changed to /tmp

        # Add PDF to knowledge base
        kb_manager.add_pdf(file_path, force_ocr=use_ocr, language=language)

        # Build search index with /tmp paths (defaults in retrieval.py)
        retrieval_system = RetrievalAugmentor(chunk_size=400, chunk_overlap=50)
        retrieval_system.build_or_update_index_from_pdf(
            file_path,
            source_name=filename,
            force_rebuild=True
        )

        # Clean up uploaded file
        os.remove(file_path)

        print(f"Successfully processed uploaded PDF: {filename}")

    except Exception as e:
        print(f"Failed to process uploaded PDF {filename}: {e}")

# System statistics endpoint
@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """
    Get comprehensive system statistics.
    """
    try:
        # Knowledge base stats with /tmp storage
        kb_manager = KnowledgeBaseManager(storage_dir="/tmp/textbooks") # Changed to /tmp
        kb_stats = kb_manager.get_storage_stats()

        # Retrieval index stats (defaults in retrieval.py)
        retrieval_system = RetrievalAugmentor(chunk_size=400, chunk_overlap=50)
        index_stats = retrieval_system.get_index_stats()

        return SystemStatsResponse(
            knowledge_base=kb_stats,
            retrieval_index=index_stats,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "QuillAI API",
        "version": "1.0.0"
    }

# Search endpoint for debugging
@app.get("/search")
async def search_knowledge_base(query: str, max_results: int = 5):
    """
    Search the knowledge base for debugging purposes.
    """
    try:
        # Retrieval system (defaults in retrieval.py)
        retrieval_system = RetrievalAugmentor(chunk_size=400, chunk_overlap=50)
        results = retrieval_system.search_chunks(query, max_results=max_results)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /docs",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "Please try again later or contact support",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "QuillAI Enhanced Academic Assistant API",
        "version": "1.0.0",
        "description": "Intelligent academic assistant with dual output system",
        "endpoints": {
            "main_query": "/query",
            "upload_pdf": "/upload-pdf",
            "statistics": "/stats",
            "health": "/health",
            "search": "/search",
            "documentation": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)