# NLP Project: Multi-hop QA methodologies by graph construction

This project constructs knowledge graphs from the Musique dataset for NLP tasks. It includes tools for building graphs from JSONL data, embedding text using SentenceTransformers, and visualizing/verifying the generated graphs.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project root.
2.  **Create and activate Conda environment**:
    ```bash
    conda create -n musique python=3.9
    conda activate musique
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The project requires `networkx`, `numpy`, `tqdm`, `sentence-transformers`, and `requests`.*

## Usage

### 1. Build Graphs

To build graphs from a Musique JSONL file, use the `graph/build_graph.py` script.

```bash
python graph/build_graph.py --input ./data/raw/musique_ans_v1.0_dev.jsonl --outdir ./graph/graphs --limit 10
```

**Arguments:**
*   `--input`: Path to the input Musique JSONL file (default: `./data/raw/musique_ans_v1.0_dev.jsonl`).
*   `--outdir`: Output directory for the generated graph pickle files (default: `./graph/graphs`).
*   `--model`: SentenceTransformer model name to use for embeddings (default: `all-MiniLM-L6-v2`).
*   `--limit`: Number of examples to process (useful for testing).

### 2. View Graphs

To inspect the contents of a generated graph pickle file, use `graph/view_graph.py`.

```bash
python graph/view_graph.py ./graphs/musique_graphs.pkl
```

**Arguments:**
*   `path`: Path to the `.pkl` or `.gpickle` file.
*   `--index`: (Optional) Index of the graph to view if the file contains a list (default: 0).
*   `--plot`: (Optional) Display a matplotlib plot of the graph.
*   `--save_png`: (Optional) Save the plot to a PNG file instead of showing it.

### 3. Verify Graph Nodes

To verify that nodes in a graph have the expected attributes (like titles), use `graph/verify_graph_nodes.py`.

```bash
python graph/verify_graph_nodes.py
```
*Note: You may need to adjust the file path inside the script or pass it as an argument if the script is updated to accept one.*

## Data Format

### Input
The input file should be in **Musique JSONL** format. Each line is a JSON object representing a question-answering example.
*   **Required fields**:
    *   `question`: The question string.
    *   `paragraphs`: A list of dictionaries, each containing:
        *   `title`: Title of the document.
        *   `paragraph_text`: The content of the paragraph.

### Output
The output is a **Pickle (`.pkl`)** file containing a **list of `networkx.Graph` objects**.

*   **Nodes**: Represent documents/paragraphs.
    *   ID: Integer (0 to N-1).
    *   Attributes:
        *   `prize`: Relevance score (float).
        *   `selected_sent_idx`: Index of the most relevant sentence in the paragraph (int).
        *   `title`: Title of the document (str).
*   **Edges**: Represent relationships between documents.
    *   Attributes:
        *   `weight`: Connection strength or cost (float).

## Project Structure

*   `graph/`
    *   `build_graph.py`: CLI script to drive the graph building process.
    *   `graph_builder.py`: Core logic for constructing the NetworkX graph from embeddings.
    *   `embedder.py`: Wrapper around `sentence-transformers` for generating text embeddings.
    *   `view_graph.py`: Utility to print graph summaries and plot graphs.
    *   `verify_graph_nodes.py`: Script to check graph node attributes.
    *   `utils_text.py`: Helper functions for text processing.
*   `data/`: Directory for data files (raw data should be placed here).
*   `requirements.txt`: Python dependencies.
