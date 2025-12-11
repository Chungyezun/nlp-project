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
from graph.graph_builder_new import GraphBuilder as GraphBuilderNew
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
    parser.add_argument(
        "--graph_builder",
        choices=["default", "new"],
        default="default",
        help="Select graph builder implementation"
    )
    # graph_builder_new 전용 옵션
    parser.add_argument(
        "--para_sim_mode",
        choices=["max", "top-k_avg", "softmax", "threshold", "sym_max"],
        default="max",
        help="Sentence-pair aggregation mode for paragraph similarity (graph_builder_new only)"
    )
    parser.add_argument(
        "--para_topk",
        type=int,
        default=3,
        help="k for top-k_avg paragraph similarity (graph_builder_new only)"
    )
    parser.add_argument(
        "--para_softmax_beta",
        type=float,
        default=10.0,
        help="beta temperature for softmax paragraph similarity (graph_builder_new only)"
    )
    parser.add_argument(
        "--para_threshold",
        type=float,
        default=0.5,
        help="similarity threshold for threshold mode (graph_builder_new only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress during retrieval"
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
    if args.graph_builder == "new":
        graph_builder = GraphBuilderNew(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            gamma=args.gamma,
            embedder=embedder,
            union_mode="reencode",
            cost_mode="scaled",
            para_sim_mode=args.para_sim_mode,
            para_topk=args.para_topk,
            para_softmax_beta=args.para_softmax_beta,
            para_threshold=args.para_threshold
        )
    else:
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
        if args.verbose:
            print(f"\n\n{'#'*80}")
            print(f"Example {i+1}/{n_examples}")
            print(f"{'#'*80}")
        else:
            print(f"Processing example {i+1}/{n_examples}...", end=" ", flush=True)
        
        # Get example data
        question, titles, docs_sentences = loader.get_example(i)
        
        # Convert sentence lists to full document texts
        doc_texts = [" ".join(sents) for sents in docs_sentences]
        
        if args.verbose:
            print(f"\nQuestion: {question}")
            print(f"Number of documents: {len(doc_texts)}")
        
        # Run multi-step retrieval
        try:
            result = retriever.retrieve_multi_step(
                initial_query=question,
                doc_texts=doc_texts,
                doc_titles=titles,
                verbose=args.verbose
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
            
            if not args.verbose:
                print(f"Done (recall: {recall:.2f})")
            
        except Exception as e:
            if args.verbose:
                print(f"Error processing example {i}: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}")
            results.append({
                'example_id': i,
                'question': question,
                'error': str(e)
            })
    
    # Calculate overall metrics
    total_recalls = [r['recall'] for r in results if 'recall' in r]
    avg_recall = sum(total_recalls) / len(total_recalls) if total_recalls else 0.0
    num_perfect = sum(1 for r in results if r.get('found_all_supporting', False))
    
    # Save results with summary
    output_data = {
        'summary': {
            'total_examples': len(results),
            'average_recall': avg_recall,
            'perfect_recall_count': num_perfect,
            'perfect_recall_rate': num_perfect / len(results) if results else 0.0
        },
        'results': results
    }
    
    print(f"\n{'='*80}")
    print(f"Overall Results:")
    print(f"  Total examples: {len(results)}")
    print(f"  Average recall: {avg_recall:.4f}")
    print(f"  Perfect recall (100%): {num_perfect}/{len(results)} ({num_perfect/len(results)*100:.1f}%)")
    print(f"{'='*80}")
    
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
