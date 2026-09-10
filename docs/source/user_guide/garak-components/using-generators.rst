
🦜 Using generators
===================

Generators are things that generate text, given some input. They are LLMs, they're Python functions, they're HTTP APIs, they're all these things. garak doesn't really care too much - just as long as text goes in and text goes out.

garak wraps a whole bunch of generators, including:


* cohere - models from `Cohere <https://cohere.ai/>`_
* function - call a Python function
* ggml - models than run locally from Gerganov's amazing `ggml <https://ggml.ai/>`_ library
* huggingface - `Hugging Face <https://hf.co/>`_ models, either locally (via pipeline or model) or API
* openai - access to `OpenAI <https://openai.com/>`_\ 's text models
* replicate - run any model on `Replicate <https://replicate.com/>`_

Parameters for garak's generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

huggingface
~~~~~~~~~~~


* ``--model_name huggingface`` (for transformers models to run locally as a pipeline)
* ``--model_type`` - use the model name from Hub. Only generative models will work. If it fails and shouldn't, please open an issue and paste in the command you tried + the exception!


* ``--model_name huggingface.InferenceAPI`` (for API-based model access)
* ``--model_type`` - the model name from Hub, e.g. ``"mosaicml/mpt-7b-instruct"``
* (optional) set the ``HF_INFERENCE_TOKEN`` environment variable to a Hugging Face API token with the "read" role; see `https://huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`_ when logged in

openai
~~~~~~


* ``--model_name openai``
* ``--model_type`` - the OpenAI model you'd like to use. ``text-babbage-001`` is fast and fine for testing; ``gpt-4`` seems weaker to many of the more subtle attacks.
* set the ``OPENAI_API_KEY`` environment variable to your OpenAI API key (e.g. "sk-19763ASDF87q6657"); see `https://platform.openai.com/account/api-keys <https://platform.openai.com/account/api-keys>`_ when logged in

Recognised model types are whitelisted, because the plugin needs to know which sub-API to use. Completion or ChatCompletion models are OK. If you'd like to use a model not supported, you should get an informative error message, and please send a PR / open an issue.

replicate
~~~~~~~~~


* ``--model_name replicate``
* ``--model_type`` - the Replicate model name and hash, e.g. ``"stability-ai/stablelm-tuned-alpha-7b:c49dae36"``
* set the ``REPLICATE_API_TOKEN`` environment variable to your Replicate API token, e.g. "r8-123XXXXXXXXXXXX"; see `https://replicate.com/account/api-tokens <https://replicate.com/account/api-tokens>`_ when logged in

cohere
~~~~~~


* ``--model_name cohere``
* ``--model_type`` (optional, ``command`` by default) - The specific Cohere model you'd like to test
* set the ``COHERE_API_KEY`` environment variable to your Cohere API key, e.g. "aBcDeFgHiJ123456789"; see `https://dashboard.cohere.ai/api-keys <https://dashboard.cohere.ai/api-keys>`_ when logged in

ggml
~~~~


* ``--model_name ggml``
* ``--model_type`` - The path to the ggml model you'd like to load, e.g. ``/home/leon/llama.cpp/models/7B/ggml-model-q4_0.bin``
* set the ``GGML_MAIN_PATH`` environment variable to the path to your ggml ``main`` executable

test
~~~~


* ``--model_name test``
* (alternatively) ``--model_name test.Blank`` For testing. This always generates the empty string, using the ``test.Blank`` generator. Will be marked as failing for any tests that *require* an output, e.g. those that make contentious claims and expect the model to refute them in order to pass.
* ``--model_name test.Repeat`` For testing. This generator repeats back the prompt it received.
