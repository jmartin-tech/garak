# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pathlib
import tempfile
import os

from garak import _config, _plugins


NON_PROMPT_PROBES = ["probes.dan.AutoDAN", "probes.tap.TAP"]
ATKGEN_PROMPT_PROBES = ["probes.atkgen.Tox", "probes.dan.Dan_10_0"]
VISUAL_PROBES = [
    "probes.visual_jailbreak.FigStep",
    "probes.visual_jailbreak.FigStepTiny",
]
PROBES = [
    classname
    for (classname, _) in _plugins.enumerate_plugins("probes")
    if classname not in NON_PROMPT_PROBES
    and classname not in VISUAL_PROBES
    and classname not in ATKGEN_PROMPT_PROBES
]
openai_api_key_missing = not os.getenv("OPENAI_API_KEY")


@pytest.fixture()
def probe_instance(classname):
    _config.run.seed = 42
    local_config_path = str(
        pathlib.Path(__file__).parents[1] / "test_config" / "translation_local_low.yaml"
    )
    if os.path.exists(local_config_path) is False:
        pytest.skip("Local config file does not exist, skipping test.")
    _config.load_config(run_config_filename=local_config_path)
    probe_instance = _plugins.load_plugin(classname, config_root=_config)

    return probe_instance


def make_prompt_list(result):
    prompt_list = []
    for attempt in result:
        for messages in attempt.messages:
            for message in messages:
                prompt_list.append(message["content"])
    return prompt_list


"""
Skip probes.tap.PAIR because it needs openai api key and large gpu resource
"""


@pytest.mark.parametrize("classname", ATKGEN_PROMPT_PROBES)
@pytest.mark.requires_storage(required_space_gb=2, path="/")
def test_atkgen_probe_translation(classname, probe_instance):
    if probe_instance.bcp47 != "en" or classname == "probes.tap.PAIR":
        return

    g = _plugins.load_plugin("generators.test.Repeat", config_root=_config)
    translator_instance = probe_instance.get_translator()
    # the requirement to set a log file here suggests an encapsulation issue
    with tempfile.NamedTemporaryFile(mode="w+") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name
        probe_instance.translator = translator_instance
        result = probe_instance.probe(g)
        prompt_list = make_prompt_list(result)
        assert prompt_list[0] == prompt_list[1]
        assert prompt_list[0] != prompt_list[2]
        assert prompt_list[0] != prompt_list[3]


@pytest.mark.parametrize("classname", PROBES)
@pytest.mark.requires_storage(required_space_gb=2, path="/")
def test_probe_prompt_translation(classname, probe_instance):
    if probe_instance.bcp47 != "en" or classname == "probes.tap.PAIR":
        return

    translator_instance = probe_instance.get_translator()
    if hasattr(probe_instance, "triggers"):
        original_triggers = probe_instance.triggers[:2]
        translate_triggers = translator_instance.translate_triggers(original_triggers)
        assert len(translate_triggers) >= len(original_triggers)

    original_prompts = probe_instance.prompts[:2]
    probe_instance.prompts = original_prompts
    g = _plugins.load_plugin("generators.test.Repeat", config_root=_config)
    with tempfile.NamedTemporaryFile(mode="w+") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name
        probe_instance.generations = 1
        result = probe_instance.probe(g)
        org_message_list = make_prompt_list(result)

        probe_instance.translator = translator_instance
        result = probe_instance.probe(g)
        message_list = make_prompt_list(result)
        assert len(org_message_list) <= len(message_list)
