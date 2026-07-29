"""Embedding-based diversity metric for generated sentences."""

from __future__ import annotations

from typing import Iterable

import torch

_SENTENCE_TRANSFORMERS_HINT = (
    "The 'sequence_embedding' diversity metric requires sentence-transformers. "
    "Install it with `pip install chemgfn[commongen]` (or `pip install sentence-transformers`), "
    "or set `diversity_metric: null` in the model config to disable the metric."
)


class SequenceDiversity:
    """Sentence-embedding diversity of a set of generated strings.

    Diversity is ``1 - mean cosine similarity`` over all unordered pairs of
    sentence embeddings, so larger values mean a more spread-out sample. Used by
    the CommonGen task, where the reference config sets
    ``diversity_metric: "sequence_embedding"``.
    """

    def __init__(
        self,
        method: str | None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        """Build the metric.

        Args:
            method: ``"sequence_embedding"`` to enable the metric, or ``None`` to
                disable it (``__call__`` then returns ``None``).
            model_name: Sentence-transformers model used to embed the strings.

        Raises:
            ValueError: If ``method`` is neither ``None`` nor ``"sequence_embedding"``.
            ImportError: If ``method`` is enabled but sentence-transformers is missing.
        """
        self.method = method
        self.model_name = model_name
        if method is None:
            self.model = None
        elif method == "sequence_embedding":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(_SENTENCE_TRANSFORMERS_HINT) from exc
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    @torch.no_grad()
    def __call__(self, sequences: Iterable[str]) -> float | None:
        """Return the diversity of ``sequences``, or ``None`` if the metric is disabled.

        Fewer than two sequences yields 0.0, since no pair is available.
        """
        if self.method is None:
            return None
        seq_list = list(sequences)
        if len(seq_list) <= 1:
            return 0.0

        from sentence_transformers.util import cos_sim

        embeddings = self.model.encode(seq_list, show_progress_bar=False, convert_to_tensor=True)
        sim = cos_sim(embeddings, embeddings)
        indices = torch.triu_indices(len(seq_list), len(seq_list), offset=1)
        return float(1 - sim[indices[0], indices[1]].mean().item())
