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
        self.pool = await asyncpg.create_pool(self.connection_string)
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create documents table with vector support
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents_proc (
                    id UUID PRIMARY KEY,
                    content TEXT,
                    summary TEXT,
                    metadata JSONB,
                    embedding vector(384)
                )
            ''')
            
            # Create index for vector similarity search
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS documents_proc_embedding_idx 
                ON documents_proc 
                USING ivfflat (embedding vector_cosine_ops)
            ''')
    
    async def store_embeddings(
        self,
        content: str,
        summary: str,
        embeddings: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store document content, summary, embeddings and metadata in PostgreSQL.
        
        Args:
            content: Original document content
            summary: Generated summary
            embeddings: List of embedding vectors
            metadata: Dictionary containing document metadata
            
        Returns:
            str: Document ID
        """
        try:
            doc_id = str(uuid.uuid4())
            
            # Convert metadata to JSON string if it's not already
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            # Convert embeddings list to PostgreSQL vector format
            embedding_vector = f"[{','.join(map(str, embeddings))}]"
            
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO documents_proc (id, content, summary, metadata, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5::vector)
                ''', doc_id, content, summary, metadata, embedding_vector)
            
            logger.info(f"Stored document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise
    
    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        try:
            # Convert query embedding to PostgreSQL vector format
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            async with self.pool.acquire() as conn:
                results = await conn.fetch('''
                    SELECT id, content, summary, metadata, 
                           1 - (embedding <=> $1::vector) as similarity
                    FROM documents_proc
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                ''', query_vector, n_results)
                
                return [
                    {
                        "id": str(row['id']),
                        "content": row['content'],
                        "summary": row['summary'],
                        "metadata": row['metadata'],
                        "similarity": float(row['similarity'])
                    }
                    for row in results
                ]
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            raise 