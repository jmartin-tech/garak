# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pytest

from garak.detectors.base import Detector
from garak.harnesses.detectoronly import DetectorOnly


@pytest.fixture(autouse=True)
def set_config_env(sample_jsonl, request):
    _config = importlib.import_module("garak._config")
    _config.plugins.harnesses["detectoronly"]["DetectorOnly"] = {
        "report_path": sample_jsonl
    }

    def restore_config_env():
        importlib.reload(_config)

    request.addfinalizer(restore_config_env)


def test_no_detectors():
    attempts = []
    detector_names = []
    evaluator = None
    h = DetectorOnly()
    with pytest.raises(ValueError) as exc_info:
        h.run(attempts, detector_names, evaluator)
    assert "No detectors" in str(exc_info.value)


def test_load_detectors(monkeypatch):
    test_attempts = []
    test_detector_names = ["always.Fail"]
    test_evaluator = None

    def mock_method(self, detectors, attempts, evaluator):
        assert detectors is not None
        assert len(detectors) == len(test_detector_names)
        for detector in detectors:
            assert isinstance(detector, Detector)
        assert attempts == test_attempts
        assert evaluator == test_evaluator

    monkeypatch.setattr(DetectorOnly, "run_detectors", mock_method)
    h = DetectorOnly()
    h.run(test_attempts, test_detector_names, test_evaluator)
