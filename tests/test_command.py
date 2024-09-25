# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import pytest
import pathlib
import random

from garak import command
from garak.evaluators import ThresholdEvaluator


@pytest.fixture(autouse=True)
def set_config_env(sample_jsonl, request):
    _config = importlib.import_module("garak._config")
    _config.plugins.harnesses["detectoronly"]["DetectorOnly"] = {
        "report_path": sample_jsonl
    }

    def restore_config_env():
        importlib.reload(_config)

    request.addfinalizer(restore_config_env)


def test_detector_only_run_auto_detectors(monkeypatch):
    import garak.harnesses.detectoronly

    threshold = random.random()

    def mock_method(self, attempts, detectors, evaluator):
        assert attempts is not None
        assert detectors is not None
        assert len(detectors) == 1
        assert evaluator is not None
        assert evaluator.threshold == threshold

    detectors = []
    evaluator = ThresholdEvaluator(threshold)
    monkeypatch.setattr(garak.harnesses.detectoronly.DetectorOnly, "run", mock_method)
    command.detector_only_run(detectors, evaluator)
