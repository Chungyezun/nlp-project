from sentence_transformers import SentenceTransformer
import numpy as np


class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text_list):
        return np.array(self.model.encode(text_list, convert_to_numpy=True))
    
    def embed_single(self, text):
        return self.embed([text])[0]
    
    @staticmethod
    def cosine_sim(a, b, eps=1e-12):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    def sim(self, a, b):
        return self.cosine_sim(a, b)
