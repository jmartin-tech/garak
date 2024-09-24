# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector only harness

Runs specified detectors on already existing prompt-response pairs from parsing a report.jsonl file.
"""

import logging

from garak import _config
from garak.harnesses import Harness


class DetectorOnly(Harness):
    def run(self, attempts, detector_names, evaluator):
        detectors = []
        for detector in sorted(detector_names):
            d = self._load_detector(detector)
            if d:
                detectors.append(d)

        if len(detectors) == 0:
            msg = "No detectors, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        super().run_detectors(
            detectors, attempts, evaluator
        )  # The probe is None, but hopefully no errors occur with probe.
