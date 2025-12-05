import pickle
import networkx as nx
import sys
from pathlib import Path

def verify_graph_nodes(graph_path):
    print(f"Loading graph from {graph_path}")
    try:
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {graph_path}")
        return

    if isinstance(data, list):
        graphs = data
        print(f"Loaded {len(graphs)} graphs.")
    else:
        graphs = [data]
        print("Loaded single graph.")

    total_graphs = len(graphs)
    passed_graphs = 0

    for i, G in enumerate(graphs):
        print(f"\n--- Verifying Graph {i} ---")
        print(f"Number of nodes: {G.number_of_nodes()}")
        
        nodes_with_title = 0
        for n, attrs in G.nodes(data=True):
            if "title" in attrs:
                nodes_with_title += 1
                # print(f"Node {n}: Title='{attrs['title']}', Prize={attrs.get('prize', 'N/A')}")
            else:
                print(f"Node {n}: No title found. Attributes: {attrs}")

        if nodes_with_title == G.number_of_nodes():
            print(f"SUCCESS: All nodes have titles in Graph {i}.")
            passed_graphs += 1
        else:
            print(f"WARNING: Only {nodes_with_title}/{G.number_of_nodes()} nodes have titles in Graph {i}.")

    print(f"\n=== Verification Summary ===")
    print(f"Total Graphs: {total_graphs}")
    print(f"Passed: {passed_graphs}")
    print(f"Failed: {total_graphs - passed_graphs}")

if __name__ == "__main__":
    # Default to the standard output file
    graph_file = Path("./graphs/musique_graphs.pkl")
    verify_graph_nodes(graph_file)
