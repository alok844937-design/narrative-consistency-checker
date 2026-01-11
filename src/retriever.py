"""
Retriever module for finding relevant context for claims.
"""
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss

class Retriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def build_index(self, chunks: List[Dict[str, any]]):
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings for chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Query text (claim)
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunk dictionaries with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for idx, (distance, chunk_idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.chunks[chunk_idx].copy()
            chunk['retrieval_score'] = float(1 / (1 + distance))  # Convert distance to similarity
            chunk['rank'] = idx + 1
            results.append(chunk)
        
        return results
    
    def retrieve_for_claims(self, claims: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, any]]]:
        """
        Retrieve relevant context for multiple claims.
        
        Args:
            claims: List of claim texts
            top_k: Number of chunks to retrieve per claim
            
        Returns:
            Dictionary mapping claims to retrieved contexts
        """
        results = {}
        for claim in claims:
            results[claim] = self.retrieve(claim, top_k)
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict[str, any]]]:
        """
        Batch retrieve for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of chunks to retrieve per query
            
        Returns:
            List of retrieval results for each query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode all queries
        query_embeddings = self.model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
        
        # Search for all queries
        distances, indices = self.index.search(query_embeddings.astype('float32'), top_k)
        
        # Prepare results
        all_results = []
        for query_idx in range(len(queries)):
            query_results = []
            for rank, (distance, chunk_idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                chunk = self.chunks[chunk_idx].copy()
                chunk['retrieval_score'] = float(1 / (1 + distance))
                chunk['rank'] = rank + 1
                query_results.append(chunk)
            all_results.append(query_results)
        
        return all_results