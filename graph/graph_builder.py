import networkx as nx
import numpy as np
from tqdm import tqdm
from .utils_text import merge_two_docs


class GraphBuilder:
    def __init__(self, lambda1=0.5, lambda2=0.5, gamma=1, b=1, embedder=None, union_mode="avg", cost_mode="default", delta_mode="max", use_precomputed=False, precomputed_path=None, precomputed_data=None):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.embedder = embedder
        self.gamma = gamma
        self.b = b
        self.union_mode = union_mode
        self.cost_mode = cost_mode
        self.delta_mode = delta_mode
        self.use_precomputed = use_precomputed
        self.precomputed_path = precomputed_path
        self._precomputed_data = precomputed_data  # dict: idx -> record

    @staticmethod
    def _unit(v, eps=1e-12):
        n = np.linalg.norm(v)
        return v / (n + eps)

    def build_graph(self, doc_sent_embeddings, query_embedding, docs_sentences, doc_titles=None):
        G = nx.Graph()
        n = len(doc_sent_embeddings)
        prizes = []

        for i in range(n):
            sent_embs = doc_sent_embeddings[i]
            sims = [self.embedder.sim(sent_emb, query_embedding) for sent_emb in sent_embs]
            max_idx = int(np.argmax(sims))
            prize = sims[max_idx]
            
            node_attrs = {"prize": prize, "selected_sent_idx": max_idx}
            if doc_titles is not None and i < len(doc_titles):
                node_attrs["title"] = doc_titles[i]
            
            G.add_node(i, **node_attrs)
            prizes.append(prize)

        edge_scores = {}
        for i in range(n):
            for j in range(i+1, n):
                sent_embs_i = doc_sent_embeddings[i]
                sent_embs_j = doc_sent_embeddings[j]

                max_sim_ij = max(
                    self.embedder.sim(e_i, e_j) 
                    for e_i in sent_embs_i 
                    for e_j in sent_embs_j
                )

                sel_i_idx = G.nodes[i]['selected_sent_idx']
                sel_j_idx = G.nodes[j]['selected_sent_idx']

                merged_text = docs_sentences[i][sel_i_idx] + " " + docs_sentences[j][sel_j_idx]
                merged_emb = self.embedder.embed_single(merged_text)
                sim_union_q = self.embedder.sim(merged_emb, query_embedding)

                if self.delta_mode == "max":
                    base_val = max(prizes[i], prizes[j])
                elif self.delta_mode == "sum":
                    base_val = prizes[i] + prizes[j]
                else:
                    raise ValueError(f"Unknown delta_mode={self.delta_mode}")

                delta_sim = sim_union_q - base_val
                delta_sim = self.gamma * delta_sim
                score = self.lambda1 * max_sim_ij + self.lambda2 * delta_sim
                edge_scores[(i, j)] = score

        # Store similarity directly as edge weight
        for (i, j), s in edge_scores.items():
            G.add_edge(i, j, weight=s)
        return G
    
    def build_graph_doclevel(self, doc_embs, query_embedding, doc_texts=None, example_idx=None):
        G = nx.Graph()
        n = len(doc_embs)

        # Try to use precomputed data if available
        if (self.use_precomputed and self._precomputed_data is not None 
            and example_idx is not None and example_idx in self._precomputed_data):
            rec = self._precomputed_data[example_idx]
            rel = np.array(rec["rel"], dtype=float)
            pair_sim = np.array(rec["pair_sim"], dtype=float)
            if self.delta_mode == "sum":
                pair_gain = np.array(rec["pair_gain_sum"], dtype=float)
            else:  # "max"
                pair_gain = np.array(rec["pair_gain_max"], dtype=float)
            
            # Use precomputed values
            prizes = rel.tolist()
            for i in range(n):
                G.add_node(i, prize=prizes[i])
            
            edge_scores = {}
            for i in range(n):
                for j in range(i+1, n):
                    sim_ij = pair_sim[i, j]
                    delta = self.gamma * pair_gain[i, j]
                    score = self.lambda1 * sim_ij + self.lambda2 * delta
                    edge_scores[(i, j)] = score
            
            # Store similarity directly as edge weight
            for (i, j), s in edge_scores.items():
                G.add_edge(i, j, weight=s)
            
            return G

        # Fallback: original computation
        prizes = []
        for i in range(n):
            p = self.embedder.sim(doc_embs[i], query_embedding)
            prizes.append(p)
            G.add_node(i, prize=p)

        edge_scores = {}
        pair_cache = {}

        pairs = []
        merged_texts = []
        
        if self.union_mode == "reencode" and doc_texts is not None:
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((i, j))
                    merged_texts.append(merge_two_docs(doc_texts[i], doc_texts[j]))
            
            if merged_texts:
                merged_embs = self.embedder.embed(merged_texts)
                for idx, (i, j) in enumerate(pairs):
                    pair_cache[(i, j)] = merged_embs[idx]

        for i in range(n):
            for j in range(i+1, n):
                sim_ij = self.embedder.sim(doc_embs[i], doc_embs[j])

                if self.union_mode == "reencode" and doc_texts is not None:
                    merged_emb = pair_cache[(i, j)]
                    sim_union_q = self.embedder.sim(merged_emb, query_embedding)
                else:
                    union_emb = self._unit(doc_embs[i] + doc_embs[j])
                    sim_union_q = self.embedder.sim(union_emb, query_embedding)

                if self.delta_mode == "max":
                    base_val = max(prizes[i], prizes[j])
                elif self.delta_mode == "sum":
                    base_val = prizes[i] + prizes[j]
                else:
                    raise ValueError(f"Unknown delta_mode={self.delta_mode}")
                
                delta = self.gamma * (sim_union_q - base_val)

                score = self.lambda1 * sim_ij + self.lambda2 * delta
                edge_scores[(i, j)] = score

        # Store similarity directly as edge weight
        for (i, j), s in edge_scores.items():
            G.add_edge(i, j, weight=s)

        return G
