# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pytest
import tempfile

from garak.detectors.base import Detector
from garak.harnesses.detectoronly import DetectorOnly


@pytest.fixture(autouse=True)
def set_config_env(sample_jsonl, request):
    _config = importlib.import_module("garak._config")
    _config.plugins.harnesses["detectoronly"]["DetectorOnly"] = {
        "report_path": sample_jsonl
    }
    temp_report_file = tempfile.NamedTemporaryFile(mode="w+")
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    def restore_config_env():
        importlib.reload(_config)

    request.addfinalizer(restore_config_env)


def test_no_specified_detectors(monkeypatch):
    from garak.evaluators.base import ZeroToleranceEvaluator

    def mock_method(detectors, attempts, evaluator):
        assert detectors is not None
        assert attempts is not None

    detector_names = []
    evaluator = None

    h = DetectorOnly()
    monkeypatch.setattr(h, "run_detectors", mock_method)
    h.run(detector_names, evaluator)


def test_load_detectors(monkeypatch):
    test_detector_names = ["always.Fail"]
    test_evaluator = None

    def mock_method(detectors, attempts, evaluator):
        assert detectors is not None
        assert len(detectors) == len(test_detector_names)
        for detector in detectors:
            assert isinstance(detector, Detector)
        assert attempts is not None
        assert evaluator == test_evaluator

    h = DetectorOnly()
    monkeypatch.setattr(h, "run_detectors", mock_method)
    h.run(test_detector_names, test_evaluator)
