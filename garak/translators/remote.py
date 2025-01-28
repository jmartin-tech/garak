# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


""" Translator that translates a prompt. """


import logging

from garak.translators.base import SimpleTranslator


class RivaTranslator(SimpleTranslator):

    ENV_VAR = "RIVA_API_KEY"
    DEFAULT_PARAMS = {
        "uri": "grpc.nvcf.nvidia.com:443",
        "function_id": "647147c1-9c23-496c-8304-2e29e7574510",
    }

    def _load_translator(self):
        import riva.client

        if self.nmt_client is None:
            # this is not really a nim, this is `riva` consider a rename
            auth = riva.client.Auth(
                None,
                True,
                self.uri,
                [
                    ("function-id", self.function_id),
                    ("authorization", "Bearer " + self.api_key),
                ],
            )
            self.nmt_client = riva.client.NeuralMachineTranslationClient(auth)

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            response = self.nmt_client.translate([text], "", source_lang, target_lang)
            return response.translations[0].text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text


class DeeplTranslator(SimpleTranslator):

    ENV_VAR = "DEEPL_API_KEY"
    DEFAULT_PARAMS = {}

    def _load_translator(self):
        from deepl import Translator

        if self.translator is None:
            self.translator = Translator(self.api_key)

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            return self.translator.translate_text(
                text, source_lang=source_lang, target_lang=target_lang
            ).text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text


DEFAULT_CLASS = "RivaTranslator"
