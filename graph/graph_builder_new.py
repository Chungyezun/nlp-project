import networkx as nx
import numpy as np
from tqdm import tqdm
from .utils_text import merge_two_docs


class GraphBuilder:
    def __init__(self, lambda1=0.5, lambda2=0.5, gamma=1, b=1, embedder=None, union_mode="avg", cost_mode="default", delta_mode="max", para_sim_mode="max", para_topk=3, para_softmax_beta=10.0, para_threshold=0.5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.embedder = embedder
        self.gamma = gamma
        self.b = b
        self.union_mode = union_mode
        self.cost_mode = cost_mode
        self.delta_mode = delta_mode
        self.para_sim_mode = para_sim_mode
        self.para_topk = para_topk
        self.para_softmax_beta = para_softmax_beta
        self.para_threshold = para_threshold

    @staticmethod
    def _unit(v, eps=1e-12):
        n = np.linalg.norm(v)
        return v / (n + eps)
    
    def _sent_pairwise_sim_matrix(self, sent_embs_i, sent_embs_j):
        """
        sent_embs_i, sent_embs_j: list of np.ndarray (d,) 혹은 (d,) 벡터
        return: S (n1, n2) with sim(e_i, e_j)
        """
        n1 = len(sent_embs_i)
        n2 = len(sent_embs_j)
        S = np.zeros((n1, n2), dtype=np.float32)
        for i, e_i in enumerate(sent_embs_i):
            for j, e_j in enumerate(sent_embs_j):
                S[i, j] = self.embedder.sim(e_i, e_j)
        return S

    def _para_sim_from_sentences(self, sent_embs_i, sent_embs_j):
        """
        문장 임베딩 리스트 두 개를 받아서 para-level similarity 스칼라 반환.
        self.para_sim_mode에 따라 네 가지 집계 방식을 선택.
        """
        if len(sent_embs_i) == 0 or len(sent_embs_j) == 0:
            return 0.0

        S = self._sent_pairwise_sim_matrix(sent_embs_i, sent_embs_j)
        mode = self.para_sim_mode

        # 0) 기존 max 모드 (backward compatibility)
        if mode == "max":
            return float(S.max())

        # 1) top-3 avg (flatten 후 상위 k개 평균)
        if mode == "top-k_avg":
            flat = S.reshape(-1)
            k = min(self.para_topk, flat.size)
            if k <= 0:
                return float(flat.mean())
            # 상위 k개만 뽑기 (np.partition 사용)
            idx = np.argpartition(flat, -k)[-k:]
            topk_vals = flat[idx]
            return float(topk_vals.mean())

        # 2) softmax 가중 평균 (지수 기반)
        if mode == "softmax":
            beta = self.para_softmax_beta
            # overflow 방지용 shifting
            shifted = beta * (S - S.max())
            weights = np.exp(shifted)
            Z = weights.sum()
            if Z <= 1e-12:
                return float(S.mean())
            sim = float((weights * S).sum() / Z)
            return sim

        # 3) threshold 기반 match 수 비율 (threshold=0.5 등)
        if mode == "threshold":
            tau = self.para_threshold
            mask = (S >= tau)
            count = mask.sum()
            # max 가능한 match 수를 min(n1, n2)로 정규화
            denom = min(S.shape[0], S.shape[1])
            if denom == 0:
                return 0.0
            return float(count / denom)

        # 4) symmetrized row/col max-pooling
        if mode == "sym_max":
            row_max_mean = float(S.max(axis=1).mean())
            col_max_mean = float(S.max(axis=0).mean())
            return 0.5 * (row_max_mean + col_max_mean)

        return float(S.max())
    
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

                # 문장×문장 기반 para-level similarity
                sim_ij = self._para_sim_from_sentences(sent_embs_i, sent_embs_j)

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
                score = self.lambda1 * sim_ij + self.lambda2 * delta_sim
                edge_scores[(i, j)] = score

        # Store similarity directly as edge weight
        for (i, j), s in edge_scores.items():
            G.add_edge(i, j, weight=s)
        return G
    
    def build_graph_doclevel(self, doc_embs, query_embedding, doc_texts=None):
        G = nx.Graph()
        n = len(doc_embs)

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
