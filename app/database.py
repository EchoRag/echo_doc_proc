import asyncpg
import logging
from typing import List, Dict, Any
import uuid
import numpy as np
import json

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    async def initialize(self):
        """Initialize database connection and create necessary tables."""
        self.pool = await asyncpg.create_pool(self.connection_string,max_size=5,min_size=2)
        async with self.pool.acquire() as conn:
            print("Initializing database connection and creating necessary tables...")
            # Enable pgvector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create documents table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents_proc (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id),
                    content TEXT,
                    summary TEXT,
                    metadata JSONB
                )
            ''')
            
            # Create chunks table with vector support
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents_proc(id),
                    chunk_text TEXT,
                    chunk_index INTEGER,
                    embedding vector(768),
                    metadata JSONB
                )
            ''')
            
            # Create index for vector similarity search
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
                ON document_chunks 
                USING ivfflat (embedding vector_cosine_ops)
            ''')
    
    async def store_document(
        self,
        content: str,
        summary: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> str:
        """
        Store document content, summary and metadata in PostgreSQL.
        
        Args:
            content: Original document content
            summary: Generated summary
            metadata: Dictionary containing document metadata
            document_id: ID of the document in the documents table
            
        Returns:
            str: Document ID
        """
        try:
            doc_id = str(uuid.uuid4())
            
            # Convert metadata to JSON string if it's not already
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO documents_proc (id, document_id, content, summary, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                ''', doc_id, document_id, content, summary, metadata)
            
            logger.info(f"Stored document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    async def store_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Store document chunks with their embeddings.
        
        Args:
            document_id: ID of the parent document
            chunks: List of dictionaries containing chunk text and embeddings
        """
        try:
            async with self.pool.acquire() as conn:
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    chunk_metadata = json.dumps(chunk.get('metadata', {}))
                    embedding_vector = f"[{','.join(map(str, chunk['embedding']))}]"
                    
                    await conn.execute('''
                        INSERT INTO document_chunks 
                        (id, document_id, chunk_text, chunk_index, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                    ''', chunk_id, document_id, chunk['text'], i, embedding_vector, chunk_metadata)
            
            logger.info(f"Stored {len(chunks)} chunks for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of similar document chunks with their metadata
        """
        try:
            # Convert query embedding to PostgreSQL vector format
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            async with self.pool.acquire() as conn:
                results = await conn.fetch('''
                    SELECT 
                        dc.id as chunk_id,
                        dc.chunk_text,
                        dc.chunk_index,
                        dc.metadata as chunk_metadata,
                        dp.id as document_id,
                        dp.content as full_content,
                        dp.summary,
                        dp.metadata as document_metadata,
                        1 - (dc.embedding <=> $1::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents_proc dp ON dc.document_id = dp.id
                    ORDER BY dc.embedding <=> $1::vector
                    LIMIT $2
                ''', query_vector, n_results)
                
                return [
                    {
                        "chunk_id": str(row['chunk_id']),
                        "chunk_text": row['chunk_text'],
                        "chunk_index": row['chunk_index'],
                        "chunk_metadata": row['chunk_metadata'],
                        "document_id": str(row['document_id']),
                        "full_content": row['full_content'],
                        "summary": row['summary'],
                        "document_metadata": row['document_metadata'],
                        "similarity": float(row['similarity'])
                    }
                    for row in results
                ]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            raise 