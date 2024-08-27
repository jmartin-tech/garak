# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVCF LLM interface"""

import json
import logging
import time
from typing import List, Union

import backoff
import requests

from garak import _config
from garak.exception import ModelNameMissingError, BadGeneratorException
from garak.generators.base import Generator


class NvcfChat(Generator):
    """Wrapper for NVIDIA Cloud Functions Chat models via NGC. Expects NVCF_API_KEY environment variable."""

    ENV_VAR = "NVCF_API_KEY"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "temperature": 0.2,
        "top_p": 0.7,
        "fetch_url_format": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/",
        "invoke_url_base": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/",
        "extra_nvcf_logging": False,
        "timeout": 60,
        "version_id": None,  # string
        "stop_on_404": True,
        "extra_params": {  # extra params for the payload, e.g. "n":1 or "model":"google/gemma2b"
            "stream": False
        },
    }

    supports_multiple_generations = False
    generator_family_name = "NVCF"

    def __init__(self, name=None, generations=10, config_root=_config):
        self.name = name
        self._load_config(config_root)
        self.fullname = (
            f"{self.generator_family_name} {self.__class__.__name__} {self.name}"
        )
        self.seed = _config.run.seed

        if self.name is None:
            raise ModelNameMissingError(
                "Please specify a function identifier in model name (-n)"
            )

        self.invoke_url = self.invoke_url_base + self.name

        if self.version_id is not None:
            self.invoke_url += f"/versions/{self.version_id}"

        super().__init__(
            self.name, generations=self.generations, config_root=config_root
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _build_payload(self, prompt) -> dict:

        payload = {
            "messages": [{"content": prompt, "role": "user"}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        for k, v in self.extra_params.items():
            payload[k] = v

        return payload

    def _extract_text_output(self, response) -> str:
        return [c["message"]["content"] for c in response["choices"]]

    @backoff.on_exception(
        backoff.fibo,
        (
            AttributeError,
            TimeoutError,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
        ),
        max_value=70,
    )
    def _call_model(
        self, prompt: str, generations_this_call: int = 1
    ) -> List[Union[str, None]]:

        session = requests.Session()

        payload = self._build_payload(prompt)

        ## NB config indexing scheme to be deprecated
        config_class = f"nvcf.{self.__class__.__name__}"
        if config_class in _config.plugins.generators:
            if "payload" in _config.plugins.generators[config_class]:
                for k, v in _config.plugins.generators[config_class]["payload"].items():
                    payload[k] = v

        if self.seed is not None:
            payload["seed"] = self.seed

        request_time = time.time()
        logging.debug("nvcf : payload %s", repr(payload))
        response = session.post(self.invoke_url, headers=self.headers, json=payload)

        while response.status_code == 202:
            if time.time() > request_time + self.timeout:
                raise TimeoutError("NVCF Request timed out")
            request_id = response.headers.get("NVCF-REQID")
            if request_id is None:
                msg = "Got HTTP 202 but no NVCF-REQID was returned"
                logging.info("nvcf : %s", msg)
                raise AttributeError(msg)
            fetch_url = self.fetch_url_format + request_id
            response = session.get(fetch_url, headers=self.headers)

        if 400 <= response.status_code < 600:
            logging.warning("nvcf : returned error code %s", response.status_code)
            logging.warning("nvcf : returned error body %s", response.content)
            if response.status_code == 400 and prompt == "":
                # error messages for refusing a blank prompt are fragile and include multi-level wrapped JSON, so this catch is a little broad
                return [None]
            if response.status_code == 404 and self.stop_on_404:
                msg = "nvcf : got 404, endpoint unavailable, stopping"
                logging.critical(msg)
                print("\n\n" + msg)
                print("nvcf :", response.content)
                raise BadGeneratorException()
            if response.status_code >= 500:
                if response.status_code == 500 and json.loads(response.content)[
                    "detail"
                ].startswith("Input value error"):
                    logging.warning("nvcf : skipping this prompt")
                    return [None]
                else:
                    response.raise_for_status()
            else:
                logging.warning("nvcf : skipping this prompt")
                return [None]

        else:
            try:
                response_body = response.json()
                return self._extract_text_output(response_body)
            except requests.exceptions.JSONDecodeError as e:
                logging.warn("failed to decode response as json", exc_info=e)
                return [response.text]


class NvcfCompletion(NvcfChat):
    """Wrapper for NVIDIA Cloud Functions Completion models via NGC. Expects NVCF_API_KEY environment variables."""

    def _build_payload(self, prompt) -> dict:

        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        for k, v in self.extra_params.items():
            payload[k] = v

        return payload

    def _extract_text_output(self, response) -> str:
        return [c["text"] for c in response["choices"]]


DEFAULT_CLASS = "NvcfChat"
