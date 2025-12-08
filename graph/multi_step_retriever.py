"""Multi-step retrieval pipeline with iterative graph building and LLM reasoning."""
import numpy as np
from typing import List, Tuple, Dict, Any
import networkx as nx
from .graph_builder import GraphBuilder
from .embedder import TextEmbedder
from .llm_reasoner import LLMReasoner


class MultiStepRetriever:
    """
    Multi-hop retrieval system that iteratively:
    1. Builds a graph from documents and current query
    2. Retrieves top documents from graph
    3. Uses LLM to generate next reasoning step/query
    4. Repeats for multiple steps
    """
    
    def __init__(
        self,
        embedder: TextEmbedder,
        graph_builder: GraphBuilder,
        llm_reasoner: LLMReasoner,
        max_steps: int = 3,
        top_k_per_step: int = 3
    ):
        """
        Initialize multi-step retriever.
        
        Args:
            embedder: TextEmbedder instance for encoding
            graph_builder: GraphBuilder instance for graph construction
            llm_reasoner: LLMReasoner instance for generating reasoning
            max_steps: Maximum number of retrieval-reasoning iterations
            top_k_per_step: Number of top documents to retrieve per step
        """
        self.embedder = embedder
        self.graph_builder = graph_builder
        self.llm_reasoner = llm_reasoner
        self.max_steps = max_steps
        self.top_k_per_step = top_k_per_step
    
    def retrieve_from_graph(self, G: nx.Graph, top_k: int) -> List[int]:
        """
        Retrieve top-k nodes from graph based on prize and edge weights.
        
        Strategy:
        1. Start with highest prize node
        2. Expand to connected nodes with high edge scores and prizes
        
        Args:
            G: NetworkX graph with node prizes and edge weights
            top_k: Number of nodes to retrieve
            
        Returns:
            List of selected node indices
        """
        if len(G.nodes) == 0:
            return []
        
        # Get node prizes
        prizes = {node: data.get('prize', 0.0) for node, data in G.nodes(data=True)}
        
        # Start with highest prize node
        selected = []
        remaining = set(G.nodes())
        
        # Select first node with max prize
        if remaining:
            first_node = max(remaining, key=lambda n: prizes[n])
            selected.append(first_node)
            remaining.remove(first_node)
        
        # Iteratively select nodes based on prize and edge scores
        while len(selected) < top_k and remaining:
            best_node = None
            best_score = -float('inf')
            
            for candidate in remaining:
                # Calculate score: prize + connection strength to selected nodes
                node_score = prizes[candidate]
                
                # Add edge contribution (lower edge weight = better, so we use negative)
                edge_contribution = 0.0
                for sel in selected:
                    if G.has_edge(candidate, sel):
                        edge_weight = G[candidate][sel].get('weight', 1.0)
                        # Lower weight is better, so subtract it
                        edge_contribution += (1.0 - edge_weight)
                
                total_score = node_score + 0.5 * edge_contribution
                
                if total_score > best_score:
                    best_score = total_score
                    best_node = candidate
            
            if best_node is not None:
                selected.append(best_node)
                remaining.remove(best_node)
            else:
                break
        
        return selected[:top_k]
    
    def retrieve_multi_step(
        self,
        initial_query: str,
        doc_texts: List[str],
        doc_titles: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform multi-step retrieval with iterative graph building and LLM reasoning.
        
        Args:
            initial_query: Initial question/query
            doc_texts: List of document texts (full documents)
            doc_titles: Optional list of document titles
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
                - 'final_docs': Final set of retrieved documents
                - 'final_doc_indices': Indices of final documents
                - 'reasoning_steps': List of reasoning texts from each step
                - 'retrieved_per_step': List of retrieved doc indices per step
        """
        if doc_titles is None:
            doc_titles = [f"Doc_{i}" for i in range(len(doc_texts))]
        
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
            
            # 1. Encode documents and query
            doc_embeddings = self.embedder.embed(doc_texts)
            query_embedding = self.embedder.embed_single(current_query)
            
            # 2. Build graph
            if verbose:
                print("Building graph...")
            G = self.graph_builder.build_graph_doclevel(
                doc_embeddings,
                query_embedding,
                doc_texts=doc_texts
            )
            
            # 3. Retrieve top-k documents from graph
            if verbose:
                print(f"Retrieving top-{self.top_k_per_step} documents...")
            retrieved_indices = self.retrieve_from_graph(G, self.top_k_per_step)
            retrieved_per_step.append(retrieved_indices)
            all_retrieved_indices.update(retrieved_indices)
            
            if verbose:
                print(f"Retrieved docs: {retrieved_indices}")
                for idx in retrieved_indices:
                    print(f"  - [{idx}] {doc_titles[idx]}: {doc_texts[idx][:100]}...")
            
            # 4. Generate reasoning with LLM (except for last step)
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
                    print(f"Reasoning: {reasoning}")
                
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
