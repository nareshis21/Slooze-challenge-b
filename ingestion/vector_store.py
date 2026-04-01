import faiss
import numpy as np
import os
from fastembed import TextEmbedding
from typing import List, Tuple

class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str = ".cache"):
        self.encoder = TextEmbedding(model_name=model_name)
        self.index = None
        self.chunks = []
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def build_index(self, chunks: List[str]):
        """
        Embeds chunks and builds a FAISS index.
        """
        self.chunks = chunks
        embeddings = list(self.encoder.embed(chunks))
        embeddings_np = np.array(embeddings).astype('float32')
        
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)

    def save_index(self, key: str):
        """
        Saves the FAISS index and chunks to the cache.
        """
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.cache_dir, f"{key}.index"))
            np.save(os.path.join(self.cache_dir, f"{key}_chunks.npy"), np.array(self.chunks))

    def load_index(self, key: str) -> bool:
        """
        Loads the FAISS index and chunks from the cache if available.
        """
        index_path = os.path.join(self.cache_dir, f"{key}.index")
        chunks_path = os.path.join(self.cache_dir, f"{key}_chunks.npy")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            self.chunks = np.load(chunks_path, allow_pickle=True).tolist()
            return True
        return False

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Searches for the top-k chunks most relevant to the query.
        """
        if self.index is None:
            return []
            
        query_embedding = list(self.encoder.embed([query]))[0]
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(distances[0][i])))
        return results

if __name__ == "__main__":
    # Test
    vs = VectorStore()
    vs.build_index(["Hello, world!", "The quick brown fox jumps over the lazy dog."])
    results = vs.search("What animal jumps?")
    for res, dist in results:
        print(f"Result: {res} (Distance: {dist})")
