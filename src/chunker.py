"""
Text chunking module for splitting documents into manageable pieces.
"""
import re
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str, method: str = 'sentences') -> List[Dict[str, any]]:
        """
        Chunk text using specified method.
        
        Args:
            text: Input text to chunk
            method: Chunking method ('sentences' or 'paragraphs')
            
        Returns:
            List of dictionaries containing chunk info
        """
        if method == 'sentences':
            chunks = self.chunk_by_sentences(text)
        elif method == 'paragraphs':
            chunks = self.chunk_by_paragraphs(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
        
        # Create structured output
        chunked_data = []
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                'chunk_id': idx,
                'text': chunk,
                'length': len(chunk)
            })
        
        return chunked_data