# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
from logging import getLogger

from garak.attempt import Conversation, Turn, Message
from garak.generators.base import Generator

logger = getLogger(__name__)


def forward(
    generator, input_ids: torch, attention_mask, batch_size=512
) -> torch.Tensor:
    """Run a forward pass on the specified model

    Args:
        generator (garak.generators.huggingface.Model): Generator with model for forward pass
        input_ids (torch.Tensor): Tokenized input to the model
        attention_mask (torch.Tensor): Attention mask from tokenizer
        batch_size (int): Number of samples to run

    Returns:
        Tensor of logits
    """
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i : i + batch_size]
        else:
            batch_attention_mask = None

        logits.append(
            generator.model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            ).logits
        )

        # Do we need to do garbage collection here?
        gc.collect()

    # Does this do anything since we return from the scope after this call?
    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def check_for_attack_success(gen_str: str, test_prefixes) -> bool:
    """Check if the jailbreak is successful against the specified model

    Args:
        gen_str (str): Generated model output
        test_prefixes (list): List of test prefixes

    Returns:
        Tuple:
            Boolean of whether the model jailbreak was successful
            Generated string
    """
    gen_str = gen_str.lower()
    jailbreak = False
    lowercase_test_prefixes = [word.lower() for word in test_prefixes]
    # This only checks for known rejections -- the model output may still not achieve the goal.
    # TODO: Use a better detector
    if len(gen_str) != 0:
        jailbreak = not any([prefix in gen_str for prefix in lowercase_test_prefixes])
    return jailbreak


class AutoDanPrefixManager:
    def __init__(self, *, generator, instruction, target, adv_string):
        """Prefix manager class for AutoDAN

        Args:
            generator (garak.generators.huggingface.Model): Generator to use
            instruction (str): Instruction to pass to the model
            target (str): Target output string
            adv_string (str): Adversarial (jailbreak) string
        """

        self.generator = generator
        self.tokenizer = generator.tokenizer
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        self.conv = Conversation()

    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string

        self.conv.turns.append(
            Turn(
                role="user",
                content=Message(text=f"{self.adv_string} {self.instruction} "),
            )
        )
        self.conv.turns.append(
            Turn(
                role="assistant",
                content=Message(text=f"{self.target}"),
            )
        )
        prompt = Generator._conversation_to_list(self.conv)

        encoding = self.tokenizer(prompt)

        if (
            self.generator.name == "llama-2"
        ):  # this is likely not the right name and some sort of map may be needed to decide the action taken
            self.conv.turns = []

            last_message = Message(None)
            self.conv.turns.append(Turn(role="user", content=last_message))
            toks = self.tokenizer(Generator._conversation_to_list(self.conv)).input_ids
            self._user_role_slice = slice(None, len(toks))

            last_message.text = f"{self.instruction}"
            toks = self.tokenizer(Generator._conversation_to_list(self.conv)).input_ids
            self._goal_slice = slice(
                self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks))
            )

            separator = " " if self.instruction else ""
            last_message.text = f"{self.adv_string}{separator}{self.instruction}"
            toks = self.tokenizer(Generator._conversation_to_list(self.conv)).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            last_message = Message(None)
            self.conv.turns.append(Turn(role="assistant", content=last_message))
            toks = self.tokenizer(Generator._conversation_to_list(self.conv)).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            last_message.text = f"{self.target}"
            toks = self.tokenizer(Generator._conversation_to_list(self.conv)).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        # This needs improvement
        else:
            python_tokenizer = False or self.generator.name == "oasst_pythia"
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                last_message = Message(None)
                self.conv.turns.append(Turn(role="user", content=last_message))
                toks = self.tokenizer(
                    Generator._conversation_to_list(self.conv)
                ).input_ids
                self._user_role_slice = slice(None, len(toks))

                last_message.text = f"{self.instruction}"
                toks = self.tokenizer(
                    Generator._conversation_to_list(self.conv)
                ).input_ids
                self._goal_slice = slice(
                    self._user_role_slice.stop,
                    max(self._user_role_slice.stop, len(toks) - 1),
                )

                separator = " " if self.instruction else ""
                last_message.text = f"{self.adv_string}{separator}{self.instruction}"
                toks = self.tokenizer(
                    Generator._conversation_to_list(self.conv)
                ).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks) - 1)

                last_message = Message(None)
                self.conv.turns.append(Turn(role="assistant", content=last_message))
                toks = self.tokenizer(
                    Generator._conversation_to_list(self.conv)
                ).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                last_message.text = f"{self.target}"
                toks = self.tokenizer(
                    Generator._conversation_to_list(self.conv)
                ).input_ids
                self._target_slice = slice(
                    self._assistant_role_slice.stop, len(toks) - 1
                )
                self._loss_slice = slice(
                    self._assistant_role_slice.stop - 1, len(toks) - 2
                )
            else:
                self._system_slice = slice(
                    None, encoding.char_to_token(len(self.conv.last_message("system")))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find("user")),
                    encoding.char_to_token(prompt.find("user") + len("user") + 1),
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(
                        prompt.find(self.instruction) + len(self.instruction)
                    ),
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(
                        prompt.find(self.adv_string) + len(self.adv_string)
                    ),
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find("assistant")),
                    encoding.char_to_token(
                        prompt.find("assistant") + len("assistant") + 1
                    ),
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)),
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                    - 1,
                )

        self.conv.turns = []

        return prompt

    def get_input_ids(self, adv_string=None):
        """Get input ids from the tokenizer for a provided string

        Args:
            adv_string (str): String to tokenize

        Returns:
            Torch tensor of input_ids
        """
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[: self._target_slice.stop])

        return input_ids
