import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
from .database import VectorDatabase
import io
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import mimetypes

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, db_connection_string: str):
        # Check if ROCm is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using AMD GPU with ROCm")
        else:
            self.device = torch.device("cpu")
            logger.warning("ROCm not available, using CPU")
        
        # Initialize models
        # Set up model cache directory
        self.cache_dir = "./models/"
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device, cache_folder=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=self.cache_dir)
        # self.summarizer = AutoModel.from_pretrained("facebook/bart-large-cnn", cache_dir=self.cache_dir).to(self.device)
        
        # Initialize database
        self.db = VectorDatabase(db_connection_string)
        
    async def initialize(self):
        """Initialize the document processor and database."""
        await self.db.initialize()
        
    async def process_document(self, content: bytes, metadata: dict = None, document_id: str = None, content_type: str = None) -> Tuple[str, str]:
        """
        Process a document and generate summary and embeddings.
        
        Args:
            content: Raw document content in bytes
            metadata: Optional metadata about the document
            document_id: ID of the document in the documents table
            content_type: MIME type of the document
            
        Returns:
            Tuple containing (document_id, summary)
        """
        try:
            # Extract text based on content type
            text = await self._extract_text(content, content_type)
            
            # Generate summary
            summary = await self.summarize_text(text)
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(text)
            
            # Ensure metadata is a dictionary
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                metadata = {"metadata": str(metadata)}
            
            # Store in database
            doc_id = await self.db.store_embeddings(
                content=text,
                summary=summary,
                embeddings=embeddings,
                metadata=metadata,
                document_id=document_id
            )
            
            return doc_id, summary
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise

    async def _extract_text(self, content: bytes, content_type: str) -> str:
        """
        Extract text from different file types.
        
        Args:
            content: Raw document content in bytes
            content_type: MIME type of the document
            
        Returns:
            Extracted text as string
        """
        if not content_type:
            # Try to detect content type
            content_type = mimetypes.guess_type(content)[0]
            
        if not content_type:
            # Default to UTF-8 text if type cannot be detected
            return content.decode('utf-8')
            
        if content_type == 'application/pdf':
            return await self._extract_pdf_text(content)
        elif content_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            return await self._extract_excel_text(content)
        elif content_type == 'text/csv':
            return await self._extract_csv_text(content)
        elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            return await self._extract_word_text(content)
        else:
            # Default to UTF-8 text for unknown types
            return content.decode('utf-8')

    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF files."""
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    async def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel files."""
        excel_file = io.BytesIO(content)
        df = pd.read_excel(excel_file)
        return df.to_string()

    async def _extract_csv_text(self, content: bytes) -> str:
        """Extract text from CSV files."""
        csv_file = io.BytesIO(content)
        df = pd.read_csv(csv_file)
        return df.to_string()

    async def _extract_word_text(self, content: bytes) -> str:
        """Extract text from Word files."""
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    async def _generate_summary(self, text: str) -> str:
        """
        Generate a summary of the text using BART.
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.summarizer.generate(**inputs, max_length=150, min_length=40)
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    async def _generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the text using SentenceTransformer.
        """
        # Split text into chunks if needed (SentenceTransformer handles this internally)
        embeddings = self.embedding_model.encode(text, convert_to_numpy=True)
        return embeddings.tolist()
        
    async def summarize_text(self, text) -> str:
        summarizer = pipeline("summarization",
                              model="sshleifer/distilbart-cnn-12-6",
                              device=self.device,
                              model_kwargs={"cache_dir": self.cache_dir})
        max_length = 1024
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        summaries = [summarizer(chunk)[0]['summary_text'] for chunk in text_chunks]
        return " ".join(summaries)
        
    async def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query to search for similar documents
            n_results: Number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        query_embedding = await self._generate_embeddings(query)
        return await self.db.search_similar(query_embedding, n_results)
