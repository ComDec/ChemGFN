"""Tests for the replay buffers in ``chemgfn.utils.replay_buffer``."""

from __future__ import annotations

import pytest
import torch

from chemgfn.utils.replay_buffer import ReplayBuffer, ReplayBufferSubmodular

BATCH_SIZE = 4
PROMPT_LEN = 3
GEN_LEN = 8


@pytest.fixture
def eos(gpt2_tokenizer):
    return gpt2_tokenizer.eos_token_id


@pytest.fixture
def prompt():
    return torch.arange(1, PROMPT_LEN + 1).view(1, PROMPT_LEN)


def make_batch(eos: int, batch_size: int = BATCH_SIZE, seed: int = 0):
    """Build distinct terminated trajectories plus their log rewards and validator output."""
    generator = torch.Generator().manual_seed(seed)
    sentences = torch.randint(
        100, 1000, (batch_size, GEN_LEN), generator=generator, dtype=torch.long
    )
    sentences[:, -1] = eos
    logrewards = torch.randn(batch_size, GEN_LEN, generator=generator)
    result_dict = {
        "validator_dict": {
            "global_score": torch.ones(batch_size),
            "local_score": torch.ones(batch_size, GEN_LEN),
            "invalid": torch.zeros(batch_size, GEN_LEN),
        }
    }
    return sentences, logrewards, result_dict


def add(buffer, prompt, tokenizer, sentences, logrewards, result_dict) -> None:
    """Call ``add_batch`` with the batch produced by :func:`make_batch`."""
    buffer.add_batch(prompt, sentences, logrewards, tokenizer, result_dict)


class TestReplayBuffer:
    """Reward-prioritised buffer keyed by prompt."""

    @pytest.fixture
    def buffer(self, eos):
        buffer = ReplayBuffer(buffer_size=10, sim_tolerance=0.25)
        buffer.set_termination_token_id(eos)
        return buffer

    def test_starts_empty(self, buffer):
        assert buffer._buffer == {}
        assert buffer.stat() == {}

    def test_adding_requires_a_termination_token(self, eos):
        buffer = ReplayBuffer(buffer_size=10)
        with pytest.raises(ValueError):
            buffer.add({"str_prompt": "p", "str_sentence": "s"})

    def test_add_batch_populates_the_prompt_bucket(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))

        stats = buffer.stat()
        assert stats["prompt_0_total_buffer"] > 0
        assert "prompt_0_avg_logR" in stats

    def test_capacity_is_respected(self, prompt, eos, gpt2_tokenizer):
        buffer = ReplayBuffer(buffer_size=3, sim_tolerance=0.0)
        buffer.set_termination_token_id(eos)
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos, batch_size=16))

        assert buffer.stat()["prompt_0_total_buffer"] <= 3

    def test_duplicates_are_ignored(self, buffer, prompt, eos, gpt2_tokenizer):
        sentences, logrewards, result_dict = make_batch(eos, batch_size=1)
        buffer.add_batch(prompt, sentences.clone(), logrewards, gpt2_tokenizer, result_dict)
        after_first = buffer.stat()["prompt_0_total_buffer"]

        buffer.add_batch(prompt, sentences.clone(), logrewards, gpt2_tokenizer, result_dict)
        assert buffer.stat()["prompt_0_total_buffer"] == after_first

    def test_sample_returns_padded_batches(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))
        sentences, answers = buffer.sample(2, prompt, gpt2_tokenizer)

        assert sentences.shape[0] == 2
        assert answers.shape[0] == 2

    def test_sample_from_an_unknown_prompt(self, buffer, prompt, gpt2_tokenizer):
        assert buffer.sample(2, prompt, gpt2_tokenizer) == (None, None)

    def test_sample_more_than_is_stored(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos, batch_size=2))
        assert buffer.sample(1000, prompt, gpt2_tokenizer) == (None, None)

    def test_reset_clears_everything(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))
        assert buffer._buffer

        buffer.reset()
        assert buffer._buffer == {}

    def test_near_duplicates_are_deduplicated(self, prompt, eos, gpt2_tokenizer):
        # A tolerance of 1.0 makes every completion a near-duplicate of every other. A
        # near-duplicate is rejected when a stored one already scores at least as well, so
        # feeding the batch in decreasing reward order leaves only the first trajectory.
        buffer = ReplayBuffer(buffer_size=10, sim_tolerance=1.0)
        buffer.set_termination_token_id(eos)

        sentences, logrewards, result_dict = make_batch(eos)
        logrewards = torch.zeros(BATCH_SIZE, GEN_LEN)
        logrewards[:, 0] = torch.arange(BATCH_SIZE, 0, -1, dtype=torch.float)
        result_dict["validator_dict"]["global_score"] = torch.zeros(BATCH_SIZE)
        buffer.add_batch(prompt, sentences, logrewards, gpt2_tokenizer, result_dict)

        assert buffer.stat()["prompt_0_total_buffer"] == 1

    def test_validator_approved_near_duplicates_are_force_added(self, prompt, eos, gpt2_tokenizer):
        # Same setup, but the validator accepts every trajectory, which bypasses the rejection.
        buffer = ReplayBuffer(buffer_size=10, sim_tolerance=1.0)
        buffer.set_termination_token_id(eos)

        sentences, logrewards, result_dict = make_batch(eos)
        logrewards = torch.zeros(BATCH_SIZE, GEN_LEN)
        logrewards[:, 0] = torch.arange(BATCH_SIZE, 0, -1, dtype=torch.float)
        buffer.add_batch(prompt, sentences, logrewards, gpt2_tokenizer, result_dict)

        assert buffer.stat()["prompt_0_total_buffer"] == BATCH_SIZE

    def test_prioritized_sampling_stays_within_the_buffer(self, prompt, eos, gpt2_tokenizer):
        buffer = ReplayBuffer(buffer_size=10, prioritized_replay=True)
        buffer.set_termination_token_id(eos)
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))

        sentences, answers = buffer.sample(3, prompt, gpt2_tokenizer)
        assert sentences.shape[0] == 3
        assert answers.shape[0] == 3


class TestReplayBufferSubmodular:
    """Buffer selecting a subset that maximises a submodular objective."""

    @pytest.fixture
    def buffer(self, eos):
        buffer = ReplayBufferSubmodular(buffer_size=5)
        buffer.set_termination_token_id(eos)
        return buffer

    def test_add_batch_requires_a_termination_token(self, prompt, eos, gpt2_tokenizer):
        buffer = ReplayBufferSubmodular(buffer_size=5)
        with pytest.raises(AssertionError):
            add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))

    def test_selection_respects_the_capacity(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos, batch_size=20))
        assert len(buffer.buffer["items"]) <= 5

    def test_sampling_returns_padded_batches(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos, batch_size=20))
        sentences, answers = buffer.sample(2, prompt, gpt2_tokenizer)

        assert sentences.shape[0] == 2
        assert answers.shape[0] == 2

    def test_sampling_an_empty_buffer(self, buffer, prompt, gpt2_tokenizer):
        assert buffer.sample(2, prompt, gpt2_tokenizer) == (None, None)

    def test_per_prompt_buckets_are_separate(self, eos, gpt2_tokenizer):
        buffer = ReplayBufferSubmodular(buffer_size=5, per_prompt=True)
        buffer.set_termination_token_id(eos)

        first = torch.arange(1, PROMPT_LEN + 1).view(1, PROMPT_LEN)
        second = torch.arange(50, 50 + PROMPT_LEN).view(1, PROMPT_LEN)
        add(buffer, first, gpt2_tokenizer, *make_batch(eos, seed=1))
        add(buffer, second, gpt2_tokenizer, *make_batch(eos, seed=2))

        assert len(buffer.buffer) == 2

    def test_reset_clears_everything(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos))
        buffer.reset()
        assert buffer.buffer["items"] == []

    def test_stat_reports_the_selection(self, buffer, prompt, eos, gpt2_tokenizer):
        add(buffer, prompt, gpt2_tokenizer, *make_batch(eos, batch_size=20))
        assert isinstance(buffer.stat(), dict)
