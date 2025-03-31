import aio_pika
import asyncio
import aiohttp
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

async def fetch_document(document_id: str) -> tuple[bytes, str]:
    """Fetch document content and content type from the download document API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{os.getenv('API_URL')}/api/v1/document/{document_id}",
            headers={"Authorization": f"Bearer {os.getenv('SERVICE_TOKEN')}"}
        ) as response:
            if response.status == 200:
                content = await response.read()
                content_type = response.headers.get('content-type')
                return content, content_type
            else:
                raise HTTPException(status_code=response.status, detail="Failed to fetch document")

async def process_message(message):
    """Process a message received from RabbitMQ."""
    async with message.process():
        # Decode the message body
        body = message.body.decode()
        logger.info(f"Received message: {body}")
        # Extract document_id from the message
        document_id = body  # Assuming the message body is the document_id
        # Fetch the document content and content type
        content, content_type = await fetch_document(document_id)
        # Process the document
        doc_id, summary = await document_processor.process_document(
            content=content,
            metadata={"document_id": document_id},
            document_id=document_id,
            content_type=content_type
        )
        logger.info(f"Processed document with ID: {doc_id}")

async def start_rabbitmq_listener():
    """Start the RabbitMQ listener."""
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(os.getenv("RABBITMQ_URL",))
    channel = await connection.channel()
    queue = await channel.declare_queue("document_uploaded", durable=True)

    # Start consuming messages
    await queue.consume(process_message)
    logger.info("Started listening to RabbitMQ queue.")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await document_processor.initialize()
    # Start the RabbitMQ listener
    asyncio.create_task(start_rabbitmq_listener())

@app.post("/api/process-document")
async def process_document(document_id: str):
    """
    Process a document by fetching it from the download document API.
    
    Args:
        document_id: ID of the document to process
    """
    try:
        # Fetch the document content and content type
        content, content_type = await fetch_document(document_id)
        
        # Process document with metadata
        doc_id, summary = await document_processor.process_document(
            content=content,
            metadata={"document_id": document_id},
            document_id=document_id,
            content_type=content_type
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document directly.
    
    Args:
        file: The document file to process
    """
    try:
        content = await file.read()
        content_type = file.content_type
        
        # Process document with metadata
        doc_id, summary = await document_processor.process_document(
            content=content,
            metadata={"filename": file.filename},
            content_type=content_type
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded document: {str(e)}")
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