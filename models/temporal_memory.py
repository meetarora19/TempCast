from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class TemporalMemory:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.events = []

    def build_index(self):
        embeddings = self.encoder.encode(self.events)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        q_emb = self.encoder.encode([query])
        q_emb = np.array(q_emb).astype("float32")

        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + float(dist))
            results.append({
                "event": self.events[idx],
                "similarity": similarity
            })

        return results