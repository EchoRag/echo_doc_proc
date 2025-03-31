# Document Processing Service

A FastAPI-based service for processing and analyzing documents (PDF, Excel, CSV, Word) with vector embeddings and summarization capabilities.

## Features

- Document processing for multiple file types (PDF, Excel, CSV, Word)
- Text extraction and summarization
- Vector embeddings generation
- Document similarity search
- RabbitMQ integration for asynchronous processing
- Clerk authentication for secure API access
- Vector database storage with pgvector

## Prerequisites

- Python 3.12
- PostgreSQL with pgvector extension
- RabbitMQ
- Clerk account for authentication
- virtualenvwrapper

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd echo_doc_proc
```

2. Set up the virtual environment using virtualenvwrapper:
```bash
# Create and activate virtual environment
mkvirtualenv -p /home/keith/Envs/echoRag/bin/python echo_doc_proc
workon echo_doc_proc

# Install dependencies
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p models
```

4. Set up environment variables in `.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
CLERK_SECRET_KEY=your_clerk_secret_key
CLERK_SERVICE_USER_ID=your_service_user_id
API_URL=your_api_url
```

## Database Setup

1. Create a PostgreSQL database with pgvector extension:
```sql
CREATE DATABASE your_database;
\c your_database
CREATE EXTENSION vector;
```

2. Create the required tables:
```sql
CREATE TABLE documents_proc (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    embedding vector(384)
);
```

## Running the Service

### Development Mode with Debugger

1. Open the project in VS Code
2. Set breakpoints in your code
3. Press F5 or use the Run and Debug menu
4. Select "Python Debugger: FastAPI" configuration

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Process Document
```bash
curl -X POST "http://localhost:8000/api/process-document?document_id=your-document-id"
```

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

### Search Documents
```bash
curl "http://localhost:8000/api/search?query=your-search-query&limit=5"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## Project Structure

```
echo_doc_proc/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── document_processor.py  # Document processing logic
│   └── database.py       # Database operations
├── models/              # Model cache directory
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
└── main.py            # Entry point for running the application
```

## Development

### Debugging

The project is configured with VS Code debugging support. To use the debugger:

1. Ensure you're using the correct Python interpreter:
   - Path: `/home/keith/Envs/echoRag/bin/python`
   - Can be set in VS Code's Python interpreter settings

2. Set breakpoints in your code
3. Press F5 or use the Run and Debug menu
4. Select "Python Debugger: FastAPI" configuration

### Adding Breakpoints

You can add breakpoints in two ways:
1. Click to the left of the line number in VS Code
2. Add the following line in your code:
```python
import pdb; pdb.set_trace()
```

## License

[Your License]