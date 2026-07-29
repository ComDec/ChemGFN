"""Shared fixtures for the ChemGFN test suite.

Tests that need a tokenizer or a language model download it from the Hugging Face Hub on first
use. The helpers below turn any failure to obtain one — no network, no cache, or a gated
repository — into a skip, so the suite stays green on a plain CPU machine without credentials.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

LLAMA_MODEL = "meta-llama/Llama-3.2-1B"

_TOKENIZER_CACHE: dict[str, Any] = {}
_MODEL_CACHE: dict[str, Any] = {}


def load_tokenizer_or_skip(name: str):
    """Return the named tokenizer, skipping the calling test if it cannot be obtained."""
    if name in _TOKENIZER_CACHE:
        value = _TOKENIZER_CACHE[name]
        if isinstance(value, str):
            pytest.skip(value)
        return value

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # network error, missing cache, or gated repository
        reason = f"Tokenizer {name!r} unavailable: {type(exc).__name__}"
        _TOKENIZER_CACHE[name] = reason
        pytest.skip(reason)

    _TOKENIZER_CACHE[name] = tokenizer
    return tokenizer


def load_causal_lm_or_skip(name: str):
    """Return the named causal LM in eval mode, skipping the test if it cannot be obtained."""
    if name in _MODEL_CACHE:
        value = _MODEL_CACHE[name]
        if isinstance(value, str):
            pytest.skip(value)
        return value

    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(name)
    except Exception as exc:  # network error, missing cache, or gated repository
        reason = f"Model {name!r} unavailable: {type(exc).__name__}"
        _MODEL_CACHE[name] = reason
        pytest.skip(reason)

    model.eval()
    _MODEL_CACHE[name] = model
    return model


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """GPT-2 tokenizer, used wherever a small public tokenizer suffices."""
    return load_tokenizer_or_skip("gpt2")


@pytest.fixture(scope="session")
def gpt2_model():
    """GPT-2 causal LM, used as a stand-in policy and reference model."""
    return load_causal_lm_or_skip("gpt2")


@pytest.fixture(scope="session")
def llama_tokenizer():
    """Llama-3.2-1B tokenizer, required by the tasks whose vocabularies are tuned to it."""
    return load_tokenizer_or_skip(LLAMA_MODEL)


class ConstantLogitsModel(torch.nn.Module):
    """Deterministic stand-in for a causal LM that emits the same logits at every position."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("token_logits", logits)

    def forward(self, input_ids: torch.Tensor = None, past_key_values=None, **kwargs):
        """Return ``(batch, seq_len, vocab)`` logits broadcast from the configured row."""
        batch, seq_len = input_ids.shape
        logits = self.token_logits.view(1, 1, -1).expand(batch, seq_len, -1).contiguous()
        return type("Output", (), {"logits": logits, "past_key_values": None})()


@pytest.fixture
def constant_logits_model():
    """Factory building a :class:`ConstantLogitsModel` over a vocabulary of the given size."""

    def _build(vocab_size: int, seed: int = 0) -> ConstantLogitsModel:
        generator = torch.Generator().manual_seed(seed)
        return ConstantLogitsModel(torch.randn(vocab_size, generator=generator))

    return _build


@pytest.fixture
def trajectory_batch():
    """Factory building a synthetic ``(log_pf, log_r, log_pterm, tokens)`` batch for a loss.

    The generated part has ``seq_len`` positions and always terminates: the final token is the
    termination token, and ``terminate_at`` optionally moves the first termination earlier.
    """

    def _build(
        batch_size: int = 4,
        seq_len: int = 5,
        prompt_len: int = 2,
        termination_token_id: int = 0,
        terminate_at: int | None = None,
        seed: int = 0,
        requires_grad: bool = False,
    ):
        generator = torch.Generator().manual_seed(seed)
        log_pf = torch.randn(batch_size, seq_len, generator=generator)
        log_r = torch.randn(batch_size, seq_len, generator=generator)
        log_pterm = torch.randn(batch_size, seq_len, generator=generator)

        tokens = torch.randint(
            1, 100, (batch_size, prompt_len + seq_len), generator=generator, dtype=torch.long
        )
        tokens[:, -1] = termination_token_id
        if terminate_at is not None:
            tokens[:, prompt_len + terminate_at :] = termination_token_id

        if requires_grad:
            log_pf.requires_grad_(True)
            log_r.requires_grad_(True)
            log_pterm.requires_grad_(True)

        return log_pf, log_r, log_pterm, tokens, termination_token_id, prompt_len

    return _build
