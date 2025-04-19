import aio_pika
import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .document_processor import DocumentProcessor
from .database import VectorDatabase
import logging
import os
from dotenv import load_dotenv
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenTelemetry
resource = Resource.create({
    "service.name": "document-processing-service",
    "service.version": "1.0.0",
    "deployment.environment": os.getenv("ENVIRONMENT", "development")
})

# Set up the tracer provider
trace.set_tracer_provider(TracerProvider(resource=resource))

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True
)

# Add the exporter to the tracer provider
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

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

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

async def fetch_document(document_id: str) -> tuple[bytes, str]:
    """Fetch document content and content type from the download document API."""
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("fetch_document") as span:
        # Create headers with trace context
        headers = {
            "Authorization": f"Bearer {os.getenv('SERVICE_TOKEN')}"
        }
        # Inject the current span context into the headers
        propagator = TraceContextTextMapPropagator()
        propagator.inject(headers)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{os.getenv('API_URL')}/api/v1/document/{document_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    content = await response.read()
                    content_type = response.headers.get('content-type')
                    return content, content_type
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch document")

async def process_message(message):
    """Process a message received from RabbitMQ."""
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("process_message", kind=SpanKind.CONSUMER) as span:
        try:
            # Decode the message body
            body = message.body.decode()
            logger.info(f"Received message: {body}")
            span.set_attribute("message.body", body)
            
            # Extract document_id from the message
            document_id = body  # Assuming the message body is the document_id
            span.set_attribute("document_id", document_id)
            
            # Update document status to processing
            await document_processor.update_document_status(document_id, "processing")
            
            # Fetch the document content and content type
            content, content_type = await fetch_document(document_id)
            span.set_attribute("content_type", content_type)
            
            # Process the document
            doc_id, summary = await document_processor.process_document(
                content=content,
                metadata={"document_id": document_id},
                document_id=document_id,
                content_type=content_type
            )
            span.set_attribute("processed_doc_id", doc_id)
            logger.info(f"Processed document with ID: {doc_id}")
            
            # Update document status to processed
            await document_processor.update_document_status(document_id, "processed")
            
            # Acknowledge the message after successful processing
            await message.ack()
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            # Update document status to failed
            await document_processor.update_document_status(document_id, "failed", str(e))
            
            # Reject the message and requeue it
            await message.nack(requeue=True)
            logger.info(f"Message requeued for document_id: {document_id}")
            
            # Optionally, you can add a delay before retrying
            # await asyncio.sleep(5)  # 5 second delay before retry

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
async def process_document(document_id: str, request: Request):
    """
    Process a document by fetching it from the download document API.
    
    Args:
        document_id: ID of the document to process
        request: FastAPI request object for trace context
    """
    try:
        # Extract trace context from request headers
        carrier = dict(request.headers)
        ctx = TraceContextTextMapPropagator().extract(carrier)
        
        # Create a new span with the extracted context
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("process_document", context=ctx, kind=SpanKind.SERVER) as span:
            span.set_attribute("document_id", document_id)
            
            # Fetch the document content and content type
            content, content_type = await fetch_document(document_id)
            
            # Process document with metadata
            doc_id, summary = await document_processor.process_document(
                content=content,
                metadata={"document_id": document_id},
                document_id=document_id,
                content_type=content_type
            )
            
            span.set_attribute("processed_doc_id", doc_id)
            
            return {
                "status": "success",
                "document_id": doc_id,
                "summary": summary,
                "trace_id": format(span.get_span_context().trace_id, "032x")
            }
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    """
    Upload and process a document directly.
    
    Args:
        file: The document file to process
        request: FastAPI request object for trace context
    """
    try:
        # Extract trace context from request headers
        carrier = dict(request.headers) if request else {}
        ctx = TraceContextTextMapPropagator().extract(carrier)
        
        # Create a new span with the extracted context
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("upload_document", context=ctx, kind=SpanKind.SERVER) as span:
            span.set_attribute("filename", file.filename)
            
            content = await file.read()
            content_type = file.content_type
            
            # Process document with metadata
            doc_id, summary = await document_processor.process_document(
                content=content,
                metadata={"filename": file.filename},
                content_type=content_type
            )
            
            span.set_attribute("processed_doc_id", doc_id)
            
            return {
                "status": "success",
                "document_id": doc_id,
                "summary": summary,
                "trace_id": format(span.get_span_context().trace_id, "032x")
            }
            
    except Exception as e:
        logger.error(f"Error processing uploaded document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_documents(query: str, limit: int = 5, request: Request = None):
    """
    Search for similar documents using a text query.
    """
    try:
        # Extract trace context from request headers
        carrier = dict(request.headers) if request else {}
        ctx = TraceContextTextMapPropagator().extract(carrier)
        
        # Create a new span with the extracted context
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("search_documents", context=ctx, kind=SpanKind.SERVER) as span:
            span.set_attribute("query", query)
            span.set_attribute("limit", limit)
            
            results = await document_processor.search_similar(query, limit)
            
            return {
                "status": "success",
                "results": results,
                "trace_id": format(span.get_span_context().trace_id, "032x")
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