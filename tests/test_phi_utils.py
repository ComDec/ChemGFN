"""Unit tests for chemgfn/utils/phi_utils.py."""

import torch

from chemgfn.utils.phi_utils import PrefixValueMemory, compute_active_before


def test_compute_active_before_eos_first():
    eos = 2
    gen_tokens = torch.tensor([[eos, 5, 6, eos], [1, 3, 4, 5]])
    active_before = compute_active_before(gen_tokens, eos=eos)

    assert active_before.shape == gen_tokens.shape
    assert active_before[0].tolist() == [True, False, False, False]
    assert active_before[1].tolist() == [True, True, True, True]


def test_prefix_value_memory_query_update_no_leakage():
    mem = PrefixValueMemory(kmax=1, alpha=1.0, gamma=1.0, min_count=1.0)
    sentences = torch.tensor([[3, 4, 5], [3, 4, 5]])
    active_before = torch.ones_like(sentences, dtype=torch.bool)
    y = torch.ones((sentences.shape[0],), dtype=torch.float32)

    pv_before, counts_before = mem.query_pv(sentences, active_before)
    assert torch.allclose(pv_before, torch.full_like(pv_before, 0.5), atol=1e-6)
    assert counts_before.sum().item() == 0.0

    mem.update(sentences, y, active_before)
    pv_after, counts_after = mem.query_pv(sentences, active_before)
    assert (counts_after >= 1.0).all()
    assert (pv_after > 0.5).all()
