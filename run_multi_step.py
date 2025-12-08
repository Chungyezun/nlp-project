#!/usr/bin/env python3
"""Run multi-step retrieval pipeline on MuSiQue dataset."""
import argparse
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader.musique_loader import MuSiQueLoader
from graph.embedder import TextEmbedder
from graph.graph_builder import GraphBuilder
from graph.llm_reasoner import LLMReasoner
from graph.multi_step_retriever import MultiStepRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Multi-step retrieval pipeline for multi-hop QA"
    )
    parser.add_argument(
        "--input",
        default="./data/raw/musique_ans_v1.0_dev.jsonl",
        help="Path to MuSiQue JSONL file"
    )
    parser.add_argument(
        "--output",
        default="./results/multi_step_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--embed_model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings"
    )
    parser.add_argument(
        "--llm_model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace LLM model for reasoning"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3,
        help="Maximum number of retrieval-reasoning iterations"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of documents to retrieve per step"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of examples to process (for testing)"
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.5,
        help="Weight for document similarity in graph"
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.5,
        help="Weight for union similarity delta in graph"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Scaling factor for delta similarity"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    loader = MuSiQueLoader(args.input)
    
    # Initialize components
    print(f"Initializing embedder: {args.embed_model}...")
    embedder = TextEmbedder(model_name=args.embed_model)
    
    print(f"Initializing LLM: {args.llm_model}...")
    llm_reasoner = LLMReasoner(model_name=args.llm_model)
    
    print("Initializing graph builder...")
    graph_builder = GraphBuilder(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        gamma=args.gamma,
        embedder=embedder,
        union_mode="reencode",
        cost_mode="scaled"
    )
    
    print("Initializing multi-step retriever...")
    retriever = MultiStepRetriever(
        embedder=embedder,
        graph_builder=graph_builder,
        llm_reasoner=llm_reasoner,
        max_steps=args.max_steps,
        top_k_per_step=args.top_k
    )
    
    # Process examples
    results = []
    n_examples = min(len(loader), args.limit)
    
    print(f"\nProcessing {n_examples} examples...")
    print("="*80)
    
    for i in range(n_examples):
        print(f"\n\n{'#'*80}")
        print(f"Example {i+1}/{n_examples}")
        print(f"{'#'*80}")
        
        # Get example data
        question, titles, docs_sentences = loader.get_example(i)
        
        # Convert sentence lists to full document texts
        doc_texts = [" ".join(sents) for sents in docs_sentences]
        
        print(f"\nQuestion: {question}")
        print(f"Number of documents: {len(doc_texts)}")
        
        # Run multi-step retrieval
        try:
            result = retriever.retrieve_multi_step(
                initial_query=question,
                doc_texts=doc_texts,
                doc_titles=titles,
                verbose=True
            )
            
            # Get supporting document indices from original data
            example_data = loader.data[i]
            supporting_indices = [
                para['idx'] for para in example_data.get('paragraphs', [])
                if para.get('is_supporting', False)
            ]
            
            # Calculate recall
            retrieved_set = set(result['final_doc_indices'])
            supporting_set = set(supporting_indices)
            recall = len(retrieved_set & supporting_set) / len(supporting_set) if supporting_set else 0.0
            
            # Store result
            results.append({
                'example_id': i,
                'question': question,
                'answer': example_data.get('answer'),
                'num_documents': len(doc_texts),
                'supporting_doc_indices': supporting_indices,
                'final_doc_indices': result['final_doc_indices'],
                'final_doc_titles': result['doc_titles'],
                'reasoning_steps': result['reasoning_steps'],
                'retrieved_per_step': result['retrieved_per_step'],
                'recall': recall,
                'found_all_supporting': recall == 1.0
            })
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'example_id': i,
                'question': question,
                'error': str(e)
            })
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Completed! Processed {len(results)} examples.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
