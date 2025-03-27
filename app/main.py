from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .document_processor import DocumentProcessor
from .database import VectorDatabase
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Processing Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get database connection string from environment
db_connection_string = os.getenv("DATABASE_URL")

# Initialize components
document_processor = DocumentProcessor(db_connection_string)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await document_processor.initialize()

@app.post("/api/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    Process an uploaded document, generate summary and store embeddings.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Process document with metadata
        doc_id, summary = await document_processor.process_document(
            content=content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type
            }
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_documents(query: str, limit: int = 5):
    """
    Search for similar documents using a text query.
    """
    try:
        results = await document_processor.search_similar(query, limit)
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 