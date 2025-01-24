# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


""" Translator that translates a prompt. """


import logging
from garak import _config, _plugins

# from garak.exception import PluginConfigurationError
from garak.translators.base import SimpleTranslator
from garak.translators.local import LocalHFTranslator, NullTranslator

# from garak.translators.remote import RivaTranslator, DeeplTranslator


def load_translator(
    translation_service: dict = {}, reverse: bool = False
) -> SimpleTranslator:
    """Load a single translator based on the configuration provided."""
    translator_instance = None
    translator_config = {
        "translators": {translation_service["model_type"]: translation_service}
    }
    logging.debug(
        f"translation_service: {translation_service['language']} reverse: {reverse}"
    )
    source_lang, target_lang = translation_service["language"].split("-")
    if source_lang == target_lang:
        return NullTranslator(translator_config)
    # code org here may need a rethink, this currently relies on the module level `translators` propagating to class types
    # TODO: adjust format to expect `model_config` and build a config_root based on `model_type` then use _plugin_load()
    # match translation_service["model_type"]:
    #     case "local":
    #         translator_instance = LocalHFTranslator(translator_config)
    #     case "riva":
    #         translator_instance = RivaTranslator(translator_config)
    #     case "deepl":
    #         translator_instance = DeeplTranslator(translator_config)
    #     case _:
    #         raise PluginConfigurationError(
    #             f"Unknown translator model_type: {translation_service["model_type"]}"
    #         )
    _plugins.load_plugin(
        path=f"translators.{translation_service["model_type"]}",
        config_root=translator_config,
    )
    return translator_instance


from garak import _config

translators = {}


def load_translators():
    if len(translators) > 0:
        return True

    from garak.exception import GarakException

    for entry in _config.run.translators:
        # example _config.run.language['language']: en-ja classname encoding result in key "en-ja" and expects a "ja-en" to match that is not always present
        translators[entry["language"]] = load_translator(
            # TODO: align class naming for Configurable consistency
            translation_service=entry
        )
    if _config.run.lang_spec != "en" and len(translators) > 1:
        # validate loaded translators have forward and reverse entries
        for translator_key in translators.keys():
            source_lang, target_lang = translator_key.split("-")
            if translators.get(f"{target_lang}-{source_lang}", None) is None:
                break
        return True
    elif len(translators) == 0:
        # continue if no translation support is provided
        return True

    msg = f"To test with target lang_spec: {_config.run.lang_spec} configuration must specify translators for each direction"
    logging.error(msg)
    raise GarakException(msg)


# should this return a set of translators for all requested lang_spec targets?
# in the base case I expect most lang_spec values will target a single language however testing a multi-lingual aware model seems reasonable
# to that end a translator from the source lang of the prompts to each target language seems reasonable however it is unclear where in the process
# the expansion should occur.
# * Should the harness process the probe for each target language in order?
# * Should the probe instantiation just generate prompts in all requested languages and attach the language under test to the prompt values?
# * Should we defer on multi-language runs and initially enforce a single value in `lang_spec` to avoid the need for attempts to know the target language?
def getTranslators(source: str, reverse: bool = False):
    load_translators()
    lang_spec = _config.run.lang_spec if hasattr(_config.run, "lang_spec") else "en"
    target_langs = lang_spec.split(",")
    returned_translators = []
    for dest in target_langs:
        key = f"{source}-{dest}" if reverse else f"{dest}-{source}"
        translator = translators.get(key, None)
        if translator is not None:
            returned_translators.append(translator)
    # return only the first translator for now
    return returned_translators[0] if len(returned_translators) > 0 else None
