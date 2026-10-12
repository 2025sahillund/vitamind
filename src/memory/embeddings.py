import hashlib
import numpy as np
from typing import List, Union
from src.config import settings

class LocalEmbedder:
    """
    Local zero-cost embedding generator with fallback protection.
    Default model: BAAI/bge-small-en-v1.5 (dimension: 384).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._fastembed_failed = False

    def _get_fastembed_model(self):
        if self._model is None and not self._fastembed_failed:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                # Fallback gracefully if model weights cannot be downloaded offline
                self._fastembed_failed = True
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized 384-dimensional vector for a string."""
        model = self._get_fastembed_model()
        if model is not None:
            try:
                embeddings = list(model.embed([text]))
                if len(embeddings) > 0:
                    return embeddings[0].tolist()
            except Exception:
                pass

        # Deterministic offline fallback (reproducible unit test vector)
        return self._deterministic_hash_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized 384-dimensional vectors for a list of strings."""
        model = self._get_fastembed_model()
        if model is not None:
            try:
                embeddings = list(model.embed(texts))
                return [emb.tolist() for emb in embeddings]
            except Exception:
                pass

        return [self._deterministic_hash_vector(t) for t in texts]

    def _deterministic_hash_vector(self, text: str) -> List[float]:
        """
        Creates a deterministic normalized vector based on SHA-256 hash seeds.
        Guarantees that identical text yields identical vectors offline.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed an array of floats
        seed_ints = [b for b in h]
        vec = []
        for i in range(self.dimension):
            # Deterministic pseudo-random generation based on hash bytes
            idx = i % len(seed_ints)
            val = ((seed_ints[idx] * (i + 1)) % 1000) / 500.0 - 1.0
            vec.append(val)
        
        # Normalize to unit length for cosine similarity
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


embedder = LocalEmbedder(
    model_name=settings.embedding_model_name,
    dimension=settings.embedding_dimension
)
