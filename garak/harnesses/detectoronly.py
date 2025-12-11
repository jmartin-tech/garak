# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector only harness

Runs specified detectors on already existing prompt-response pairs from parsing a report.jsonl file.
"""

import json
import logging
import os

from garak import _config, _plugins, attempt
from garak.exception import PluginConfigurationError
from garak.harnesses import Harness


class DetectorOnly(Harness):
    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        if not hasattr(self, "report_path") or not os.path.exists(self.report_path):
            raise PluginConfigurationError(
                f"{self.__class__} must be provided a valid report_path."
            )

    def run(self, detector_names, evaluator):
        attempts = []
        with open(self.report_path) as f:
            for line in f:
                entry = json.loads(line)
                match entry["entry_type"]:
                    case "start_run setup":
                        if detector_names == []:
                            # If the user doesn't specify any detectors, repeat the same as the report's
                            logging.info("Using detectors from the report file")
                            if entry["plugins.detector_spec"] == "auto":
                                entry["plugins.probe_spec"]
                                probes, _ = _config.parse_plugin_spec(
                                    entry["plugins.probe_spec"], "probes"
                                )
                                for probe in probes:
                                    primary = _plugins.plugin_info(probe)[
                                        "primary_detector"
                                    ]
                                    detector_names = [primary]
                                    if entry["plugins.extended_detectors"]:
                                        detector_names = list(
                                            set(detector_names)
                                            | set(
                                                _plugins.plugin_info(probe)[
                                                    "extended_detectors"
                                                ]
                                            )
                                        )
                            else:
                                detector_names = entry["plugins.detector_spec"].split(
                                    ","
                                )
                    case "attempt":
                        if entry["status"] == 1:
                            attempts.append(attempt.Attempt.from_dict(entry))

        if detector_names == []:
            raise ValueError(
                "No detectors specified and report file missing setup entry"
            )

        if len(attempts) == 0:
            raise ValueError(f"No attempts found in {self.report_path}")

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

        self.run_detectors(
            detectors, attempts, evaluator
        )  # The probe is None, but hopefully no errors occur with probe.
