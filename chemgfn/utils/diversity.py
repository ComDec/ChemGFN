from __future__ import annotations

from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class SequenceDiversity:
    """Embedding-based sequence diversity metric.

    Diversity is computed as ``1 - mean_cosine_similarity`` across all unordered
    pairs of sentence embeddings.
    """

    def __init__(
        self,
        method: str | None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self.method = method
        self.model_name = model_name
        if method is None:
            self.model = None
        elif method == "sequence_embedding":
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    @torch.no_grad()
    def __call__(self, sequences: Iterable[str]) -> float | None:
        if self.method is None:
            return None
        seq_list = list(sequences)
        if len(seq_list) <= 1:
            return 0.0

        embeddings = self.model.encode(seq_list, show_progress_bar=False, convert_to_tensor=True)
        sim = cos_sim(embeddings, embeddings)
        indices = torch.triu_indices(len(seq_list), len(seq_list), offset=1)
        return float(1 - sim[indices[0], indices[1]].mean().item())
