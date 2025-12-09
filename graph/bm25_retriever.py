"""BM25-based retrieval for baseline comparison."""
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Simple BM25-based retriever for baseline.
    Does not use graph structure - just retrieves based on keyword matching.
    """
    
    def __init__(self):
        """Initialize BM25 retriever."""
        pass
    
    def build_index(self, doc_texts: List[str]):
        """
        Build BM25 index from documents.
        
        Args:
            doc_texts: List of document texts
        """
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_texts = doc_texts
    
    def retrieve(self, query: str, top_k: int) -> List[int]:
        """
        Retrieve top-k documents using BM25.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of document indices
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()
