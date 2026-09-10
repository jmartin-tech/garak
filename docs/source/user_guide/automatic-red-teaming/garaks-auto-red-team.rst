
🪖 garak's auto red-team
========================

garak include an auto red-team module, ``art``. Plugins in this have different targets; for example, ``art.Tox`` tries to get a model to produce toxicity.

The probe works by loading a red-teaming model, and conducting a "conversation" between the red-team model and the specified generator. The red-team model is prompted with the generator's output, or with nothing if it's the first conversation. It will try to provoke the generator into a failure mode depending on the plugin. Conversation turns progress a fixed number of times, or until the generator repeats itself, or (optionally) if the generator stops responding.&#x20;

There's a blog post detailing the basic version of garak's ``art`` module here, `https://interhumanagreement.substack.com/p/faketoxicityprompts-automatic-red <https://interhumanagreement.substack.com/p/faketoxicityprompts-automatic-red>`_
