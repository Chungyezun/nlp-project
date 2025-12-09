#!/usr/bin/env python3
"""Run IRCoT baseline (BM25 + LLM reasoning) on MuSiQue dataset."""
import argparse
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader.musique_loader import MuSiQueLoader
from graph.bm25_retriever import BM25Retriever
from graph.llm_reasoner import LLMReasoner
from graph.ircot_baseline import IRCoTBaseline


def main():
    parser = argparse.ArgumentParser(
        description="IRCoT baseline (BM25 + LLM) for multi-hop QA"
    )
    parser.add_argument(
        "--input",
        default="./data/raw/musique_ans_v1.0_dev.jsonl",
        help="Path to MuSiQue JSONL file"
    )
    parser.add_argument(
        "--output",
        default="./results/ircot_baseline_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--llm_model",
        default="gpt2",
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
    print(f"Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever()
    
    print(f"Initializing LLM: {args.llm_model}...")
    llm_reasoner = LLMReasoner(model_name=args.llm_model)
    
    print("Initializing IRCoT baseline...")
    ircot = IRCoTBaseline(
        bm25_retriever=bm25_retriever,
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
        
        # Run IRCoT baseline
        try:
            result = ircot.retrieve_multi_step(
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
        'method': 'IRCoT Baseline (BM25 + LLM)',
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
