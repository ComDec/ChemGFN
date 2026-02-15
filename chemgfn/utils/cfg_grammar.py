"""
Fork from transformers_cfg repo.
"""

import copy
import math
from typing import Literal, Optional

import torch
from transformers.generation.logits_process import (
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
    LogitsProcessor,
)
from transformers.utils import add_start_docstrings
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.token_grammar_recognizer import AbsTokenRecognizer


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        grammar_constraint: AbsTokenRecognizer,
        valid_token_start_idx: Optional[int] = None,
        execution_mode: Literal["speculation", "full_mask"] = "speculation",
        device: Optional[torch.device] = None,
    ) -> None:
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.valid_token_start_idx = valid_token_start_idx
        self.execution_mode = execution_mode
        self.device = device
        self.shift_vars = 3

    def mask_logits(self, logits: torch.FloatTensor, device: torch.device) -> torch.FloatTensor:
        masked_logits = logits.clone()
        if self.execution_mode == "speculation":
            # try to accept the most likely token
            acceptance = torch.zeros(
                (logits.shape[0], len(self.grammar_constraint.homomorphism)),
                dtype=torch.bool,
                device=device,
            )
            next_tokens = torch.argmax(logits, dim=-1)
            for i, next_token in enumerate(next_tokens.tolist()):
                acceptance[i] = self.grammar_constraint.filter_vocab(
                    self.batch_parsing_states[i], device
                )

                try:
                    is_next_token_accepted = self.grammar_constraint.accept_token_ids(
                        [next_token], self.batch_parsing_states[i]
                    )
                except ValueError:
                    is_next_token_accepted = False
                if is_next_token_accepted:
                    acceptance[i, next_token] = True

                # else:
                #     # resolve each stack to a tensor of True/False for each token
                #     # indicating acceptance
                #     # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
                #     acceptance[i] = self.grammar_constraint.filter_vocab(
                #         self.batch_parsing_states[i], device
                #     )
        else:
            acceptance = self.grammar_constraint.batch_filter_vocab(
                self.batch_parsing_states, device
            )

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = masked_logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        # masked_logits[acceptance] += torch.var(masked_logits) * self.shift_vars
        return masked_logits

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_parsing_state()
                )
                for _ in range(len(input_ids))
            ]

        self.batch_parsing_states = self.grammar_constraint.update_state_with_batch_token_seqs(
            input_ids, self.batch_parsing_states, self.valid_token_start_idx
        )
        masked_scores = self.mask_logits(scores, device)
        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

    def reset(self):
        self.batch_parsing_states = None
        if isinstance(self.grammar_constraint, IncrementalGrammarConstraint):
            self.grammar_constraint.reset()
