"""Tests for the rollout and bookkeeping helpers in ``chemgfn.utils.gfn_utils``."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from chemgfn.utils.gfn_utils import (
    base_to_lora,
    calculate_diversity,
    calculate_diversity_by_length,
    generate_and_return_termination_logprob,
    get_termination_vals,
    lora_to_base,
    prepare_token_mask,
)

EOS = 0
VOCAB_SIZE = 16


def _make_reward_fn(prompt_len: int):
    """Build a stand-in reward returning per-state tensors of the shape the losses expect.

    Mirrors the real rewards: ``(batch, generated + 1)`` entries, one per visited state.
    """

    def reward_fn(state: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        batch_size = state.shape[0]
        num_states = state.shape[1] - prompt_len + 1
        zeros = torch.zeros(batch_size, num_states)
        return {
            "reward": zeros,
            "reward_unpenalized": zeros,
            "log_pf_ref": torch.zeros(batch_size, num_states - 1),
            "log_pterm_ref": zeros,
        }

    return reward_fn


class TestPrepareTokenMask:
    """Vocabulary masks built from a legal-token file."""

    @pytest.fixture
    def vocab_file(self, tmp_path):
        path = tmp_path / "legal_tokens.txt"
        path.write_text("C\nO\nN\n(\n)\n")
        return str(path)

    def test_masks_are_complementary(self, gpt2_tokenizer, vocab_file):
        legal, illegal, legal_ids = prepare_token_mask(gpt2_tokenizer, vocab_file)

        assert legal.shape == (len(gpt2_tokenizer),)
        assert illegal.shape == (len(gpt2_tokenizer),)
        assert torch.equal(illegal, ~legal)
        assert len(legal_ids) == 5

    def test_listed_tokens_are_legal(self, gpt2_tokenizer, vocab_file):
        legal, _, _ = prepare_token_mask(gpt2_tokenizer, vocab_file)

        for token in "CON()":
            token_id = gpt2_tokenizer.encode(token, add_special_tokens=False)[0]
            assert legal[token_id]

    def test_eos_is_always_legal(self, gpt2_tokenizer, vocab_file):
        legal, _, _ = prepare_token_mask(gpt2_tokenizer, vocab_file)
        assert legal[gpt2_tokenizer.eos_token_id]

    def test_bos_is_illegal_when_distinct_from_eos(self, gpt2_tokenizer, vocab_file):
        if gpt2_tokenizer.bos_token_id == gpt2_tokenizer.eos_token_id:
            pytest.skip("GPT-2 reuses the same id for BOS and EOS")
        legal, _, _ = prepare_token_mask(gpt2_tokenizer, vocab_file)
        assert not legal[gpt2_tokenizer.bos_token_id]


class TestCalculateDiversity:
    """Average per-position token entropy."""

    def test_single_sample_has_no_diversity(self):
        assert calculate_diversity(torch.tensor([[1, 2, 3, 4, 5]])) == 0.0

    def test_identical_samples_have_no_diversity(self):
        tokens = torch.tensor([[1, 2, 3]] * 3)
        assert calculate_diversity(tokens) == pytest.approx(0.0, abs=1e-6)

    def test_all_distinct_tokens_reach_maximum_entropy(self):
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert calculate_diversity(tokens) == pytest.approx(torch.log(torch.tensor(3.0)), abs=1e-4)

    def test_partial_overlap_lies_between_the_extremes(self):
        tokens = torch.tensor([[1, 2, 3], [1, 2, 6], [1, 8, 9]])
        assert 0.0 < calculate_diversity(tokens) < float(torch.log(torch.tensor(3.0)))


class TestCalculateDiversityByLength:
    """Diversity reported per pre-EOS length bucket."""

    def test_groups_by_length_before_eos(self):
        tokens = torch.tensor(
            [
                [3, 4, EOS, EOS],
                [3, 5, EOS, EOS],
                [7, 8, 9, EOS],
            ]
        )
        by_length = calculate_diversity_by_length(tokens, eos_id=EOS)

        assert set(by_length) == {2, 3}
        assert by_length[2] > 0.0

    def test_single_member_buckets_have_no_diversity(self):
        tokens = torch.tensor([[3, 4, EOS], [7, 8, EOS]])
        by_length = calculate_diversity_by_length(tokens[:1], eos_id=EOS)
        assert by_length == {2: 0.0}

    def test_sequences_without_eos_use_their_full_length(self):
        tokens = torch.tensor([[3, 4, 5], [6, 7, 8]])
        assert set(calculate_diversity_by_length(tokens, eos_id=EOS)) == {3}


class TestGetTerminationVals:
    """Per-trajectory quantities read off at the terminating step."""

    def test_reads_the_terminating_index(self):
        batch_size, seq_len, prompt_len = 4, 5, 2
        log_r = torch.arange(batch_size * seq_len, dtype=torch.float).view(batch_size, seq_len)
        tokens = torch.full((batch_size, prompt_len + seq_len), 9, dtype=torch.long)
        tokens[:, prompt_len + 3] = EOS

        log_pfs, term_log_r, term_log_r_unpen, gen_len = get_termination_vals(
            tokens,
            torch.zeros(batch_size, seq_len),
            torch.zeros(batch_size, seq_len),
            log_r,
            log_r,
            EOS,
            prompt_len,
        )

        assert torch.equal(gen_len, torch.full((batch_size,), 3))
        assert torch.equal(term_log_r, log_r[:, 3])
        assert torch.equal(term_log_r_unpen, log_r[:, 3])
        assert log_pfs.shape == (batch_size,)

    def test_trajectory_log_probability_is_the_prefix_sum(self):
        prompt_len = 1
        tokens = torch.tensor([[9, 4, 5, EOS, EOS]])
        log_pf = torch.tensor([[-1.0, -2.0, -4.0, -8.0]])
        log_pterm = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])

        log_pfs, _, _, gen_len = get_termination_vals(
            tokens,
            log_pf,
            log_pterm,
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            EOS,
            prompt_len,
        )

        # Terminates after two tokens: log_pf[0] + log_pf[1] + log_pterm[2].
        assert gen_len.item() == 2
        assert log_pfs.item() == pytest.approx(-1.0 - 2.0 - 0.5)

    def test_policy_terms_are_optional(self):
        prompt_len = 2
        tokens = torch.full((3, prompt_len + 4), 9, dtype=torch.long)
        tokens[:, prompt_len + 1] = EOS
        log_r = torch.randn(3, 4)

        log_pfs, term_log_r, _, gen_len = get_termination_vals(
            tokens, None, None, log_r, log_r, EOS, prompt_len
        )

        assert log_pfs is None
        assert torch.equal(term_log_r, log_r[:, 1])
        assert torch.equal(gen_len, torch.ones(3, dtype=gen_len.dtype))


class TestLoRASwitches:
    """Adapter enable/disable helpers."""

    def test_lora_to_base_disables_adapters_and_evaluates(self):
        model = Mock()
        lora_to_base(model)

        model.base_model.disable_adapter_layers.assert_called_once()
        model.eval.assert_called_once()

    def test_base_to_lora_enables_adapters_and_trains(self):
        model = Mock()
        base_to_lora(model)

        model.base_model.enable_adapter_layers.assert_called_once()
        model.train.assert_called_once()


class TestGenerateAndReturnTerminationLogprob:
    """Policy rollout with forced termination and absorbing padding."""

    PROMPT_LEN = 4

    @pytest.fixture
    def encoded_data(self):
        return {"encoded_prompt": torch.randint(1, VOCAB_SIZE, (3, self.PROMPT_LEN))}

    def test_output_shapes(self, constant_logits_model, encoded_data):
        max_len = 5
        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=max_len,
        )

        batch_size, prompt_len = encoded_data["encoded_prompt"].shape
        assert result["state"].shape == (batch_size, prompt_len + max_len + 1)
        assert result["log_pf"].shape == (batch_size, max_len + 1)
        assert result["log_pterm"].shape == (batch_size, max_len + 1)
        assert result["log_r"].shape == (batch_size, max_len + 1)

    def test_every_trajectory_terminates(self, constant_logits_model, encoded_data):
        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=4,
        )

        assert (result["state"][:, -1] == EOS).all()

    def test_termination_is_absorbing(self, constant_logits_model, encoded_data):
        prompt_len = encoded_data["encoded_prompt"].shape[1]
        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=6,
        )

        generated = result["state"][:, prompt_len:]
        after_first_eos = (generated == EOS).cumsum(dim=1) >= 1
        assert (generated[after_first_eos] == EOS).all()

    def test_min_len_suppresses_early_termination(self, constant_logits_model, encoded_data):
        min_len = 3
        prompt_len = encoded_data["encoded_prompt"].shape[1]
        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=6,
            min_len=min_len,
        )

        assert (result["state"][:, prompt_len : prompt_len + min_len] != EOS).all()

    def test_action_seq_is_replayed_verbatim(self, constant_logits_model, encoded_data):
        prompt = encoded_data["encoded_prompt"]
        batch_size, prompt_len = prompt.shape
        stored = torch.randint(1, VOCAB_SIZE, (batch_size, 3))
        action_seq = torch.cat([prompt[:, :-1], stored], dim=1)

        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=5,
            action_seq=action_seq,
        )

        generated = result["state"][:, prompt_len:]
        assert torch.equal(generated[:, : stored.shape[1]], stored)
        # Steps past the stored trajectory fall back to the termination token.
        assert (generated[:, stored.shape[1] :] == EOS).all()

    def test_log_probabilities_are_non_positive_and_finite(
        self, constant_logits_model, encoded_data
    ):
        result = generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data=encoded_data,
            termination_token_id=EOS,
            reward_fn=_make_reward_fn(self.PROMPT_LEN),
            max_len=5,
        )

        for key in ("log_pf", "log_pterm"):
            assert torch.isfinite(result[key]).all()
            assert (result[key] <= 0).all()

    def test_reward_receives_the_state_without_the_forced_column(
        self, constant_logits_model, encoded_data
    ):
        seen = {}

        def recording_reward_fn(state, **kwargs):
            seen["shape"] = state.shape
            seen["scaffold"] = kwargs.get("scaffold")
            return _make_reward_fn(self.PROMPT_LEN)(state, **kwargs)

        max_len = 4
        generate_and_return_termination_logprob(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_data={**encoded_data, "scaffold": "O=C1Nc2cc(*)ccc2N1"},
            termination_token_id=EOS,
            reward_fn=recording_reward_fn,
            max_len=max_len,
        )

        batch_size, prompt_len = encoded_data["encoded_prompt"].shape
        assert seen["shape"] == (batch_size, prompt_len + max_len)
        assert seen["scaffold"] == "O=C1Nc2cc(*)ccc2N1"
