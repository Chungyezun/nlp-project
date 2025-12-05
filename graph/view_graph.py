#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import networkx as nx


def guess_label(G, n):
    d = G.nodes[n]
    for k in ["text", "sentence", "sent", "title", "name", "label", "id"]:
        if k in d and d[k] is not None:
            v = d[k]
            v = str(v)
            return (v[:80] + "…") if len(v) > 80 else v
    return str(n)


def summarize(G: nx.Graph, max_nodes=10, max_edges=10):
    print("=== Graph Summary ===")
    print("Type:", type(G))
    print("Directed:", G.is_directed())
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # Graph-level attrs
    if getattr(G, "graph", None):
        if len(G.graph) > 0:
            print("\nGraph attributes:")
            for k, v in list(G.graph.items())[:20]:
                s = str(v)
                print(f"  - {k}: {(s[:120] + '…') if len(s) > 120 else s}")

    print("\nSample nodes:")
    for i, n in enumerate(list(G.nodes)[:max_nodes]):
        d = G.nodes[n]
        keys = list(d.keys())

        prize = d.get("prize", None)
        sel = d.get("selected_sent_idx", None)

        print(f"  [{i}] node={n}  prize={prize}  selected_sent_idx={sel}  keys={keys[:8]}{'...' if len(keys)>8 else ''}")
        print(f"       label={guess_label(G, n)}")

    print("\nSample edges:")
    for i, (u, v, d) in enumerate(list(G.edges(data=True))[:max_edges]):
        keys = list(d.keys())
        w = d.get("weight", None)
        print(f"  [{i}] {u} -- {v}  weight={w}  keys={keys[:8]}{'...' if len(keys)>8 else ''}")


def plot_graph(G: nx.Graph, out_png=None, max_draw_nodes=200):
    import matplotlib.pyplot as plt

    if G.number_of_nodes() > max_draw_nodes:
        deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
        keep = [n for n, _ in deg[:max_draw_nodes]]
        H = G.subgraph(keep).copy()
        print(f"[plot] Graph is large -> drawing top-{max_draw_nodes} by degree (subgraph has {H.number_of_nodes()} nodes).")
    else:
        H = G

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(H, seed=42, k=None)
    nx.draw_networkx_edges(H, pos, alpha=0.35, width=0.8)
    nx.draw_networkx_nodes(H, pos, node_size=120, alpha=0.9)

    labels = {}
    for n in list(H.nodes)[:30]:
        labels[n] = guess_label(H, n)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[plot] Saved to {out_png}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to .gpickle (pickle) graph file")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib plot")
    p.add_argument("--save_png", default=None, help="Save plot to PNG instead of showing")
    p.add_argument("--index", type=int, default=0, help="Index of the graph to view if file contains a list")
    args = p.parse_args()

    path = Path(args.path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        print(f"Loaded {len(data)} graphs from {path}")
        if args.index < 0 or args.index >= len(data):
            print(f"Error: Index {args.index} out of range (0-{len(data)-1})")
            return
        G = data[args.index]
        print(f"Viewing graph at index {args.index}")
    else:
        G = data
        print(f"Loaded single graph from {path}")

    summarize(G)

    if args.plot or args.save_png:
        plot_graph(G, out_png=args.save_png)


if __name__ == "__main__":
    main()