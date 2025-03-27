# Document Processing Service

This service processes documents, creates summaries, and stores vector embeddings for use with LLM applications.

## Features

- Document processing and summarization
- Vector embedding generation using AMD GPU (ROCm)
- Vector storage in ChromaDB
- REST API interface

## Prerequisites

- Python 3.8+
- AMD GPU with ROCm support
- Ollama installed and running locally

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- POST `/api/process-document`: Upload and process a document
  - Accepts multipart/form-data with a file field named "file"
  - Returns document summary and processing status

## GPU Configuration

This application uses ROCm for AMD GPU acceleration. Make sure you have the appropriate ROCm drivers and PyTorch with ROCm support installed. 