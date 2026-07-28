"""Tests for the reward models in ``chemgfn.models.reward``."""

from __future__ import annotations

import pytest
import torch

from chemgfn.models.reward import (
    AbsorbingPrefixReward,
    PrefixShapedReward,
    compute_active_before,
    score_fast,
)

REWARD_CLASSES = [PrefixShapedReward, AbsorbingPrefixReward]

EOS = 0
VOCAB_SIZE = 12
PROMPT_LEN = 3


class _StubTokenizer:
    """Minimal tokenizer surface used by the reward models."""

    eos_token_id = EOS


class _ConstantScoreValidator:
    """Validator returning a fixed per-state local score."""

    def __init__(self, local_score: torch.Tensor) -> None:
        self.local_score = local_score
        self.calls: list[tuple] = []

    def __call__(self, sentences, tokenizer, scaffold=None, *args, **kwargs):
        self.calls.append((sentences, scaffold))
        batch_size = sentences.shape[0]
        return {
            "local_score": self.local_score,
            "global_score": self.local_score[:, -1],
            "invalid": torch.zeros_like(self.local_score),
            "full_tokens": ["stub"] * batch_size,
        }


@pytest.fixture
def batch():
    """Prompt-plus-generation token ids that terminate two tokens into the generation."""
    tokens = torch.full((2, PROMPT_LEN + 4), 5, dtype=torch.long)
    tokens[:, PROMPT_LEN + 2 :] = EOS
    return tokens


class TestComputeActiveBefore:
    """Mask of states still on-trajectory before each action."""

    def test_positions_after_the_first_eos_are_inactive(self):
        gen_tokens = torch.tensor([[EOS, 5, 6, EOS], [1, 3, 4, 5]])
        active = compute_active_before(gen_tokens, eos=EOS)

        assert active.shape == gen_tokens.shape
        assert active[0].tolist() == [True, False, False, False]
        assert active[1].tolist() == [True, True, True, True]

    def test_the_first_position_is_always_active(self):
        gen_tokens = torch.full((3, 5), EOS)
        assert compute_active_before(gen_tokens, eos=EOS)[:, 0].all()

    def test_single_step_trajectories(self):
        gen_tokens = torch.tensor([[EOS], [7]])
        assert compute_active_before(gen_tokens, eos=EOS).tolist() == [[True], [True]]


class TestScoreFast:
    """Reference prior computed under the frozen model."""

    def test_output_shapes(self, batch, constant_logits_model):
        out = score_fast(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )

        num_generated = batch.shape[1] - PROMPT_LEN
        assert out["ref_log_pf"].shape == (batch.shape[0], num_generated)
        assert out["ref_log_pterm"].shape == (batch.shape[0], num_generated + 1)
        assert out["reward"].shape == (batch.shape[0], num_generated + 1)

    def test_reference_log_probabilities_are_non_positive(self, batch, constant_logits_model):
        out = score_fast(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )

        assert (out["ref_log_pf"] <= 0).all()
        assert (out["ref_log_pterm"] <= 0).all()

    def test_states_past_termination_carry_no_reward(self, batch, constant_logits_model):
        out = score_fast(
            model=constant_logits_model(VOCAB_SIZE),
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )

        # The batch terminates at generated position 2, so states 3 and 4 are past termination.
        assert (out["reward"][:, 3:] == 0.0).all()
        assert (out["reward"][:, :3] != 0.0).all()

    def test_forbidden_tokens_are_penalised(self, batch, constant_logits_model):
        model = constant_logits_model(VOCAB_SIZE)
        invalid = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        invalid[5] = True

        baseline = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )
        masked = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
            invalid_vocab_mask=invalid,
            illegal_vocab_penalty=-50.0,
        )

        emitted_the_masked_token = batch[:, PROMPT_LEN:] == 5
        assert emitted_the_masked_token.any()
        assert (
            masked["ref_log_pf"][emitted_the_masked_token]
            < baseline["ref_log_pf"][emitted_the_masked_token]
        ).all()
        # EOS was left alone, so its share of the distribution can only grow.
        assert (masked["ref_log_pterm"] > baseline["ref_log_pterm"]).all()

    def test_temperature_flattens_the_prior(self, batch, constant_logits_model):
        model = constant_logits_model(VOCAB_SIZE)
        cold = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
            reward_temperature=1.0,
        )
        hot = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
            reward_temperature=5.0,
        )

        assert not torch.allclose(cold["reward"], hot["reward"])


@pytest.mark.parametrize("reward_cls", REWARD_CLASSES)
class TestRewardContract:
    """Behaviour shared by every reward model."""

    def test_reference_only_reward_matches_the_prior(
        self, reward_cls, batch, constant_logits_model
    ):
        model = constant_logits_model(VOCAB_SIZE)
        out = reward_cls(sentence_validator=None).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=model,
            tokenizer=_StubTokenizer(),
        )

        reference = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
            illegal_vocab_penalty=reward_cls(sentence_validator=None).illegal_vocab_penalty,
        )["reward"]

        scale = 0.5
        assert torch.allclose(out["reward"], reference * scale, atol=1e-6)
        assert out["validator_dict"] is None

    def test_output_keys(self, reward_cls, batch, constant_logits_model):
        out = reward_cls(sentence_validator=None).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=constant_logits_model(VOCAB_SIZE),
            tokenizer=_StubTokenizer(),
        )

        for key in ("reward", "reward_unpenalized", "log_pf_ref", "log_pterm_ref"):
            assert key in out

    def test_reward_shape_matches_the_number_of_states(
        self, reward_cls, batch, constant_logits_model
    ):
        num_states = batch.shape[1] - PROMPT_LEN + 1
        validator = _ConstantScoreValidator(torch.zeros(batch.shape[0], num_states))

        out = reward_cls(sentence_validator=validator).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=constant_logits_model(VOCAB_SIZE),
            tokenizer=_StubTokenizer(),
        )

        assert out["reward"].shape == (batch.shape[0], num_states)
        assert torch.isfinite(out["reward"]).all()

    def test_validator_sees_the_generation_and_the_scaffold(
        self, reward_cls, batch, constant_logits_model
    ):
        num_states = batch.shape[1] - PROMPT_LEN + 1
        validator = _ConstantScoreValidator(torch.zeros(batch.shape[0], num_states))

        reward_cls(sentence_validator=validator).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=constant_logits_model(VOCAB_SIZE),
            tokenizer=_StubTokenizer(),
            scaffold="O=C1Nc2cc(*)ccc2N1",
        )

        sentences, scaffold = validator.calls[0]
        assert torch.equal(sentences, batch[:, PROMPT_LEN:])
        assert scaffold == "O=C1Nc2cc(*)ccc2N1"

    def test_scaling_factor_scales_the_task_score(self, reward_cls, batch, constant_logits_model):
        num_states = batch.shape[1] - PROMPT_LEN + 1
        local_score = torch.ones(batch.shape[0], num_states)
        model = constant_logits_model(VOCAB_SIZE)

        def score_with(scaling_factor):
            return reward_cls(
                sentence_validator=_ConstantScoreValidator(local_score.clone())
            ).score(
                input_batch=batch,
                prompt_length=PROMPT_LEN,
                model=model,
                tokenizer=_StubTokenizer(),
                scaling_factor=scaling_factor,
            )[
                "reward"
            ]

        # A unit increase in the scaling factor adds the task score once, at every state the
        # reward attributes a score to.
        delta = score_with(2.0) - score_with(1.0)
        assert delta.max().item() == pytest.approx(1.0)
        assert delta.min().item() >= 0.0
        assert torch.allclose(delta, delta.round())


class TestPrefixShapedReward:
    """Reference prior shaped by the per-prefix task score (Expr24)."""

    def test_unpenalized_reward_is_the_scaled_prior(self, batch, constant_logits_model):
        num_states = batch.shape[1] - PROMPT_LEN + 1
        model = constant_logits_model(VOCAB_SIZE)

        out = PrefixShapedReward(
            sentence_validator=_ConstantScoreValidator(torch.ones(batch.shape[0], num_states))
        ).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=model,
            tokenizer=_StubTokenizer(),
            reference_logits_scale=0.25,
        )

        reference = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )["reward"]

        assert torch.allclose(out["reward_unpenalized"], reference * 0.25, atol=1e-6)

    def test_a_fully_scoring_prefix_cancels_the_terminal_prior(self, batch, constant_logits_model):
        num_states = batch.shape[1] - PROMPT_LEN + 1
        model = constant_logits_model(VOCAB_SIZE)

        out = PrefixShapedReward(
            sentence_validator=_ConstantScoreValidator(torch.ones(batch.shape[0], num_states))
        ).score(
            input_batch=batch,
            prompt_length=PROMPT_LEN,
            model=model,
            tokenizer=_StubTokenizer(),
            scaling_factor=0.0,
            reference_logits_scale=1.0,
        )

        reference = score_fast(
            model=model,
            encoded_input=batch,
            termination_token_id=EOS,
            skip_first=PROMPT_LEN,
        )["reward"]

        expected = reference - reference[:, -1].unsqueeze(-1)
        assert torch.allclose(out["reward"], expected, atol=1e-6)


class TestAbsorbingPrefixReward:
    """Reference prior plus an absorbed suffix target (SMILES, AMP)."""

    def test_terminal_score_is_copied_onto_the_absorbing_states(self, constant_logits_model):
        # Generation terminates after two tokens, so states 3 and 4 are the absorbing region.
        tokens = torch.full((1, PROMPT_LEN + 4), 5, dtype=torch.long)
        tokens[:, PROMPT_LEN + 2 :] = EOS

        local_score = torch.tensor([[0.0, 0.1, 0.7, 0.0, 0.0]])
        model = constant_logits_model(VOCAB_SIZE)

        out = AbsorbingPrefixReward(sentence_validator=_ConstantScoreValidator(local_score)).score(
            input_batch=tokens,
            prompt_length=PROMPT_LEN,
            model=model,
            tokenizer=_StubTokenizer(),
            scaling_factor=1.0,
            reference_logits_scale=0.0,
        )

        # With the reference prior switched off, the reward is the absorbed score alone: state 2
        # terminates and scores 0.7, state 3 absorbs it, and state 4 is never reached.
        assert out["reward"][0].tolist() == pytest.approx([0.0, 0.1, 0.7, 0.7, 0.0])

    def test_a_shorter_local_score_is_padded_at_the_root(self, constant_logits_model):
        tokens = torch.full((2, PROMPT_LEN + 4), 5, dtype=torch.long)
        tokens[:, -1] = EOS

        # One score per generated token rather than per state.
        validator = _ConstantScoreValidator(torch.ones(2, 4))
        out = AbsorbingPrefixReward(sentence_validator=validator).score(
            input_batch=tokens,
            prompt_length=PROMPT_LEN,
            model=constant_logits_model(VOCAB_SIZE),
            tokenizer=_StubTokenizer(),
        )

        assert out["reward"].shape == (2, 5)
