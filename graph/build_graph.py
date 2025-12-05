#!/usr/bin/env python3
"""Simple CLI to build graphs from a MuSiQue corpus file."""
import argparse
import os
import pickle
from pathlib import Path
import networkx as nx
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader.musique_loader import MuSiQueLoader
from embedder import TextEmbedder
from graph_builder import GraphBuilder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="./data/raw/musique_ans_v1.0_dev.jsonl", help="Path to musique jsonl file")
    p.add_argument("--outdir", required=False, default="./graph/graphs", help="Output directory for graph files")
    p.add_argument("--model", required=False, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--limit", type=int, default=10, help="Number of examples to process (for quick tests)")
    args = p.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loader = MuSiQueLoader(str(inpath))
    embedder = TextEmbedder(model_name=args.model)
    gb = GraphBuilder(embedder=embedder)

    all_graphs = []
    n = min(len(loader), args.limit)
    for i in range(n):
        q, titles, docs_sentences = loader.get_example(i)

        # Build sentence-level embeddings for each doc (list of lists)
        doc_sent_embeddings = []
        for sents in docs_sentences:
            if not sents:
                doc_sent_embeddings.append([embedder.embed_single("")])
            else:
                doc_sent_embeddings.append(list(embedder.embed(sents)))

        query_emb = embedder.embed_single(q if q is not None else "")
        G = gb.build_graph(doc_sent_embeddings, query_emb, docs_sentences, doc_titles=titles)
        all_graphs.append(G)

    outfn = outdir / "musique_graphs.pkl"
    with open(outfn, 'wb') as f:
        pickle.dump(all_graphs, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(all_graphs)} graphs to {outfn}")


if __name__ == "__main__":
    main()
