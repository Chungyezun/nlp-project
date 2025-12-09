"""IRCoT-style baseline: BM25 retrieval + LLM reasoning iteration."""
from typing import List, Dict, Any
from .bm25_retriever import BM25Retriever
from .llm_reasoner import LLMReasoner


class IRCoTBaseline:
    """
    IRCoT-style baseline using BM25 retrieval and LLM reasoning.
    
    Pipeline:
    1. Retrieve documents using BM25
    2. Generate reasoning with LLM
    3. Use reasoning as next query
    4. Repeat for max_steps
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        llm_reasoner: LLMReasoner,
        max_steps: int = 3,
        top_k_per_step: int = 3
    ):
        """
        Initialize IRCoT baseline.
        
        Args:
            bm25_retriever: BM25Retriever instance
            llm_reasoner: LLMReasoner instance for reasoning
            max_steps: Maximum number of retrieval-reasoning iterations
            top_k_per_step: Number of documents to retrieve per step
        """
        self.bm25_retriever = bm25_retriever
        self.llm_reasoner = llm_reasoner
        self.max_steps = max_steps
        self.top_k_per_step = top_k_per_step
    
    def retrieve_multi_step(
        self,
        initial_query: str,
        doc_texts: List[str],
        doc_titles: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform multi-step retrieval with BM25 and LLM reasoning.
        
        Args:
            initial_query: Initial question/query
            doc_texts: List of document texts
            doc_titles: Optional list of document titles
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing retrieval results
        """
        if doc_titles is None:
            doc_titles = [f"Doc_{i}" for i in range(len(doc_texts))]
        
        # Build BM25 index
        self.bm25_retriever.build_index(doc_texts)
        
        current_query = initial_query
        all_retrieved_indices = set()
        reasoning_steps = []
        retrieved_per_step = []
        
        for step in range(self.max_steps):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Step {step + 1}/{self.max_steps}")
                print(f"Current Query: {current_query}")
                print(f"{'='*60}")
            
            # 1. Retrieve documents using BM25
            if verbose:
                print(f"Retrieving top-{self.top_k_per_step} documents with BM25...")
            
            retrieved_indices = self.bm25_retriever.retrieve(
                current_query, 
                self.top_k_per_step
            )
            retrieved_per_step.append(retrieved_indices)
            all_retrieved_indices.update(retrieved_indices)
            
            if verbose:
                print(f"Retrieved docs: {retrieved_indices}")
                for idx in retrieved_indices:
                    print(f"  - [{idx}] {doc_titles[idx]}: {doc_texts[idx][:100]}...")
            
            # 2. Generate reasoning with LLM (except for last step)
            if step < self.max_steps - 1:
                retrieved_docs = [doc_texts[i] for i in retrieved_indices]
                
                if verbose:
                    print("\nGenerating reasoning with LLM...")
                
                reasoning = self.llm_reasoner.reason(
                    question=initial_query,
                    retrieved_docs=retrieved_docs
                )
                
                reasoning_steps.append(reasoning)
                
                if verbose:
                    print(f"Reasoning: {reasoning[:200]}...")
                
                # Use reasoning as next query
                current_query = reasoning
            else:
                # Last step, no more reasoning needed
                reasoning_steps.append("Final retrieval completed")
        
        # Compile final results
        final_indices = sorted(list(all_retrieved_indices))
        final_docs = [doc_texts[i] for i in final_indices]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Final Results: Retrieved {len(final_indices)} unique documents")
            print(f"Document indices: {final_indices}")
            print(f"{'='*60}\n")
        
        return {
            'final_docs': final_docs,
            'final_doc_indices': final_indices,
            'reasoning_steps': reasoning_steps,
            'retrieved_per_step': retrieved_per_step,
            'doc_titles': [doc_titles[i] for i in final_indices]
        }
