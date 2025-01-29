# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


""" Translator that translates a prompt. """


from typing import List
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MarianMTModel, MarianTokenizer

from garak.translators.base import SimpleTranslator
from garak.resources.api.huggingface import HFCompatible


class NullTranslator(SimpleTranslator):

    def _load_translator(self):
        pass

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text

    def translate_prompts(
        self,
        prompts: List[str],
        only_translate_word: bool = False,
        reverse_translate_judge: bool = False,
    ) -> List[str]:
        return prompts


class LocalHFTranslator(SimpleTranslator, HFCompatible):
    """Local translation using Huggingface m2m100 models
    Reference:
      - https://huggingface.co/facebook/m2m100_1.2B
      - https://huggingface.co/facebook/m2m100_418M
      - https://huggingface.co/docs/transformers/model_doc/marian
    """

    DEFAULT_PARAMS = {
        "name": "Helsinki-NLP/opus-mt-{}",  # should this be `model_name` or align with generators?
        "hf_args": {
            "device": "cpu",
        },
    }

    def __init__(self, config_root: dict = {}) -> None:
        self._load_config(config_root=config_root)
        self.device = self._select_hf_device()
        super().__init__(config_root=config_root)

    def _load_translator(self):
        if "m2m100" in self.model_name:
            self.model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        else:
            # if model is not m2m100 expect the model name to be "Helsinki-NLP/opus-mt-{}" where the format string
            # is replace with the language path defined in the configuration as self.language
            model_name = self.model_name.format(self.language)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if "m2m100" in self.model_name:
            self.tokenizer.src_lang = source_lang

            encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)

            translated = self.model.generate(
                **encoded_text,
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
            )

            translated_text = self.tokenizer.batch_decode(
                translated, skip_special_tokens=True
            )[0]

            return translated_text
        else:
            # this assumes MarianMTModel type
            source_text = self.tokenizer.prepare_seq2seq_batch(
                [text], return_tensors="pt"
            ).to(self.device)

            translated = self.model.generate(**source_text)

            translated_text = self.tokenizer.batch_decode(
                translated, skip_special_tokens=True
            )[0]

            return translated_text


DEFAULT_CLASS = "LocalHFTranslator"
