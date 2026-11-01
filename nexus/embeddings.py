"""Local sentence embeddings. Loading is lazy and offline unless explicitly requested."""
import threading
import numpy as np

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'


class RetrievalUnavailable(RuntimeError):
    pass


class LocalEncoder:
    model_id = MODEL_ID
    revision = MODEL_REVISION
    dimensions = 384
    max_sequence_length = 256

    def __init__(self, download=False):
        self.download = download
        self._model = None
        self._lock = threading.RLock()

    def prepare(self):
        with self._lock:
            if self._model is None:
                try:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    torch.set_num_threads(2)
                    self._model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                        local_files_only=not self.download, trust_remote_code=False,
                        device='cpu', model_kwargs={'use_safetensors':True})
                except Exception as exc:
                    raise RetrievalUnavailable('Local embedding model unavailable. Install requirements-retrieval.txt, then run python -m nexus.retrieval download-model.') from exc
            return self._model

    def count(self, text):
        model = self.prepare()
        return len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def encode(self, texts):
        model = self.prepare()
        if any(self.count(text) > self.max_sequence_length for text in texts):
            raise ValueError('Text exceeds the embedding model token limit; shorten the query or chunk')
        with self._lock:
            return np.asarray(model.encode(texts, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)
