"""AMP oracle: ProtTrans AlBert features → MLP classifier.

Reproduces the oracle from BioSeq-GFN-AL (Jain et al., 2022).
Architecture: Linear(4096,1024) → Dropout → ReLU → Linear(1024,1024) → Dropout → ReLU → Linear(1024,2)
Score = softmax(logits)[:, 1]  (probability of AMP class)
Weights: D2_target_MLP_best_Layer_1024_AlBert.pt from MJ10/clamp-gen-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AMPMLP(nn.Module):
    """MLP matching the clamp-gen-data oracle architecture.

    Note: layer naming skips fc3 (fc1 → fc2 → fc4) to match saved weights.
    """

    def __init__(
        self,
        num_inputs: int = 4096,
        num_hiddens: int = 1024,
        num_outputs: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc4 = nn.Linear(num_hiddens, num_outputs)  # fc3 skipped (matches saved keys)
        self.act = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor, mc_dropout: bool = False) -> Tensor:
        training_or_mc = self.training or mc_dropout
        x = F.dropout(self.act(self.fc1(x)), p=self.dropout_rate, training=training_or_mc)
        x = F.dropout(self.act(self.fc2(x)), p=self.dropout_rate, training=training_or_mc)
        return self.fc4(x)


class AMPOracle:
    """Full oracle: ProtTrans feature extraction + MLP scoring.

    Parameters
    ----------
    weights_path : str | Path | None
        Path to .pt state dict. If None, MLP uses random weights (for testing).
    prot_albert_name : str
        HuggingFace model name for ProtTrans AlBert.
    device : str
        Torch device.
    mc_samples : int
        Number of MC-dropout forward passes (0 = deterministic).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        prot_albert_name: str = "Rostlab/prot_albert",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mc_samples: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.mc_samples = int(mc_samples)

        # --- MLP ---
        self.mlp = AMPMLP(num_inputs=4096, num_hiddens=1024, num_outputs=2, dropout_rate=0.5)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.mlp.load_state_dict(state)
        self.mlp.to(self.device).eval()

        # --- ProtTrans (lazy-loaded) ---
        self._prot_albert_name = prot_albert_name
        self._tokenizer = None
        self._encoder = None

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from transformers import AlbertModel, AlbertTokenizer

        self._tokenizer = AlbertTokenizer.from_pretrained(
            self._prot_albert_name, do_lower_case=False
        )
        # Build AA -> token_id lookup from vocab (tokens are "▁A", "▁C", etc.)
        vocab = self._tokenizer.get_vocab()
        self._aa_to_id = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            key = f"\u2581{aa}"  # ▁ + AA
            if key in vocab:
                self._aa_to_id[aa] = vocab[key]
        self._cls_id = self._tokenizer.cls_token_id  # [CLS] = 2
        self._sep_id = self._tokenizer.sep_token_id  # [SEP] = 3
        self._pad_id = self._tokenizer.pad_token_id  # [PAD] = 0

        self._encoder = AlbertModel.from_pretrained(self._prot_albert_name)
        self._encoder.to(self.device).eval()

    def _tokenize_sequences(self, sequences: Sequence[str]) -> dict[str, Tensor]:
        """Manually tokenize AA sequences to bypass broken SentencePiece lowercase."""
        max_len = max(len(s) for s in sequences) + 2  # +2 for [CLS] and [SEP]
        input_ids = torch.full((len(sequences), max_len), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids = [self._cls_id]
            for aa in seq:
                ids.append(self._aa_to_id.get(aa, self._tokenizer.unk_token_id))
            ids.append(self._sep_id)
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def _embed(self, sequences: Sequence[str]) -> Tensor:
        """Mean-pooled ProtTrans AlBert embeddings (4096-dim)."""
        self._ensure_encoder()
        enc = self._tokenize_sequences(sequences)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        output = self._encoder(**enc)
        # Mean-pool over AA tokens only (exclude [CLS] at 0 and [SEP] at end)
        # For each sequence of length L: positions 1..L are AA tokens
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # exclude [CLS]
        # exclude [SEP]: find last 1 in each row, set to 0
        for i in range(mask.shape[0]):
            last_one = mask[i].nonzero()[-1].item()
            mask[i, last_one] = 0
        mask = mask.unsqueeze(-1).float()
        emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb

    @torch.no_grad()
    def score_sequences(self, sequences: Sequence[str]) -> Tensor:
        """Score a batch of amino acid sequences. Returns (B,) tensor in [0, 1]."""
        if not sequences:
            return torch.zeros(0, device=self.device)

        embeddings = self._embed(sequences)

        if self.mc_samples > 0:
            logits_sum = torch.zeros(len(sequences), 2, device=self.device)
            for _ in range(self.mc_samples):
                logits_sum += self.mlp(embeddings, mc_dropout=True)
            probs = F.softmax(logits_sum / self.mc_samples, dim=1)
        else:
            logits = self.mlp(embeddings, mc_dropout=False)
            probs = F.softmax(logits, dim=1)

        return probs[:, 1]  # P(AMP)
