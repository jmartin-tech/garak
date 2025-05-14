"""Base Generator

All `garak` generators must inherit from this.
"""

import logging
import re
from typing import List, Union

from colorama import Fore, Style
import tqdm

from garak import _config
from garak.configurable import Configurable
from garak.exception import GarakException
import garak.resources.theme


class Generator(Configurable):
    """Base class for objects that wrap an LLM or other text-to-text service"""

    # avoid class variables for values set per instance
    DEFAULT_PARAMS = {
        "max_tokens": 150,
        "temperature": None,
        "top_k": None,
        "context_len": None,
        "skip_seq_start": None,
        "skip_seq_end": None,
    }

    _run_params = {"deprefix", "seed", "rate_limit"}
    _system_params = {"parallel_requests", "max_workers"}

    active = True
    generator_family_name = None
    parallel_capable = True

    _shm_name = "rate_limit_queue"

    # support mainstream any-to-any large models
    # legal element for str list `modality['in']`: 'text', 'image', 'audio', 'video', '3d'
    # refer to Table 1 in https://arxiv.org/abs/2401.13601
    modality: dict = {"in": {"text"}, "out": {"text"}}

    supports_multiple_generations = (
        False  # can more than one generation be extracted per request?
    )

    def __init__(self, name="", config_root=_config):
        self._load_config(config_root)
        if "description" not in dir(self):
            self.description = self.__doc__.split("\n")[0]
        if name:
            self.name = name
        if "fullname" not in dir(self):
            if self.generator_family_name is not None:
                self.fullname = f"{self.generator_family_name}:{self.name}"
            else:
                self.fullname = self.name
        if not self.generator_family_name:
            self.generator_family_name = "<empty>"

        # setup general rate limiter shared memory
        if self.rate_limit is not None:
            from multiprocessing import shared_memory
            import time
            import numpy as np
            import weakref

            rate_limit_scaler = []
            for _ in range(0, self.rate_limit):
                rate_limit_scaler.append(time.time() - 60)
            rate_limit_values = np.array(rate_limit_scaler)

            shm_ratelimits = shared_memory.SharedMemory(
                create=True,
                size=rate_limit_values.nbytes,
            )
            self._shm_name = shm_ratelimits.name
            self.nd_args = {
                "shape": rate_limit_values.shape,
                "dtype": rate_limit_values.dtype,
            }
            root_ratelimits = np.ndarray(
                **(self.nd_args | {"buffer": shm_ratelimits.buf})
            )
            root_ratelimits[:] = rate_limit_values[:]
            self._finalizer = weakref.finalize(self, self._cleanup_shm, self._shm_name)

        print(
            f"🦜 loading {Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}generator{Style.RESET_ALL}: {self.generator_family_name}: {self.name}"
        )
        logging.info("generator init: %s", self)

    def _cleanup_shm(self, shared_mem_name):
        from multiprocessing import shared_memory

        if hasattr(self, "_shm_rate_limits"):
            self._shm_rate_limits.close()
        else:
            shm = shared_memory.SharedMemory(shared_mem_name)
            shm.close()
            shm.unlink()

    def _call_model(
        self, prompt: str, generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        """Takes a prompt and returns an API output

        _call_api() is fully responsible for the request, and should either
        succeed or raise an exception. The @backoff decorator can be helpful
        here - see garak.generators.openai for an example usage.

        Can return None if no response was elicited"""
        raise NotImplementedError

    def _pre_generate_hook(self):
        # implement rate limiting here per generator
        if self.rate_limit is not None:
            import time
            import random

            if not hasattr(self, "_rate_limits"):
                from multiprocessing import shared_memory
                import numpy as np

                self._shm_rate_limits = shared_memory.SharedMemory(self._shm_name)
                self._rate_limits = np.ndarray(
                    **(self.nd_args | {"buffer": self._shm_rate_limits.buf})
                )

            # count number of requests older that current time, backoff until count is lower than limit
            count = self.rate_limit
            while True:
                cutoff = time.time() - 60
                count = 0
                for limit in self._rate_limits:
                    if limit > cutoff:
                        count += 1
                if count < self.rate_limit:
                    break
                else:
                    time.sleep(random.randint(3, 60))
        pass

    @staticmethod
    def _verify_model_result(result: List[Union[str, None]]):
        assert isinstance(result, list), "_call_model must return a list"
        assert (
            len(result) == 1
        ), f"_call_model must return a list of one item when invoked as _call_model(prompt, 1), got {result}"
        assert (
            isinstance(result[0], str) or result[0] is None
        ), "_call_model's item must be a string or None"

    def clear_history(self):
        pass

    def _post_generate_hook(self, outputs: List[str | None]) -> List[str | None]:
        if self.rate_limit is not None:
            import time

            # find oldest entry and set to current time
            oldest = self._rate_limits.argmin()
            self._rate_limits[oldest] = time.time()
        return outputs

    def _prune_skip_sequences(self, outputs: List[str | None]) -> List[str | None]:
        rx_complete = (
            re.escape(self.skip_seq_start) + ".*?" + re.escape(self.skip_seq_end)
        )
        rx_missing_final = re.escape(self.skip_seq_start) + ".*?$"
        rx_missing_start = ".*?" + re.escape(self.skip_seq_end)

        if self.skip_seq_start == "":
            complete_seqs_removed = [
                (
                    re.sub(rx_missing_start, "", o, flags=re.DOTALL | re.MULTILINE)
                    if o is not None
                    else None
                )
                for o in outputs
            ]
            return complete_seqs_removed

        else:
            complete_seqs_removed = [
                (
                    re.sub(rx_complete, "", o, flags=re.DOTALL | re.MULTILINE)
                    if o is not None
                    else None
                )
                for o in outputs
            ]

            partial_seqs_removed = [
                (
                    re.sub(rx_missing_final, "", o, flags=re.DOTALL | re.MULTILINE)
                    if o is not None
                    else None
                )
                for o in complete_seqs_removed
            ]

            return partial_seqs_removed

    def generate(
        self, prompt: str, generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        """Manages the process of getting generations out from a prompt

        This will involve iterating through prompts, getting the generations
        from the model via a _call_* function, and returning the output

        Avoid overriding this - try to override _call_model or _call_api
        """

        self._pre_generate_hook()

        assert (
            generations_this_call >= 0
        ), f"Unexpected value for generations_per_call: {generations_this_call}"

        if generations_this_call == 0:
            logging.debug("generate() called with generations_this_call = 0")
            return []

        if generations_this_call == 1:
            outputs = self._call_model(prompt, 1)

        elif self.supports_multiple_generations:
            outputs = self._call_model(prompt, generations_this_call)

        else:
            outputs = []

            if (
                hasattr(self, "parallel_requests")
                and self.parallel_requests
                and isinstance(self.parallel_requests, int)
                and self.parallel_requests > 1
            ):
                from multiprocessing import Pool

                multi_generator_bar = tqdm.tqdm(
                    total=generations_this_call,
                    leave=False,
                    colour=f"#{garak.resources.theme.GENERATOR_RGB}",
                )
                multi_generator_bar.set_description(self.fullname[:55])

                pool_size = min(
                    generations_this_call,
                    self.parallel_requests,
                    self.max_workers,
                )

                try:
                    with Pool(pool_size) as pool:
                        for result in pool.imap_unordered(
                            self._call_model, [prompt] * generations_this_call
                        ):
                            self._verify_model_result(result)
                            outputs.append(result[0])
                            multi_generator_bar.update(1)
                except OSError as o:
                    if o.errno == 24:
                        msg = "Parallelisation limit hit. Try reducing parallel_requests or raising limit (e.g. ulimit -n 4096)"
                        logging.critical(msg)
                        raise GarakException(msg) from o
                    else:
                        raise (o)

            else:
                generation_iterator = tqdm.tqdm(
                    list(range(generations_this_call)),
                    leave=False,
                    colour=f"#{garak.resources.theme.GENERATOR_RGB}",
                )
                generation_iterator.set_description(self.fullname[:55])
                for i in generation_iterator:
                    output_one = self._call_model(
                        prompt, 1
                    )  # generate once as `generation_iterator` consumes `generations_this_call`
                    self._verify_model_result(output_one)
                    outputs.append(output_one[0])

        outputs = self._post_generate_hook(outputs)

        if hasattr(self, "skip_seq_start") and hasattr(self, "skip_seq_end"):
            if self.skip_seq_start is not None and self.skip_seq_end is not None:
                outputs = self._prune_skip_sequences(outputs)

        return outputs
