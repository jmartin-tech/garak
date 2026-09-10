
✨ Our Features
===============

Direct focus on LLM security
----------------------------

garak focuses primarily on LLM security. While other tools might look at generic machine learning security, or app security, garak specifically focuses on risks that are inherent in and unique to LLM deployment, such as prompt injection, jailbreaks, guardrail bypass, text replay, and so on.

Automated scanning
------------------

garak has a range of probes but doesn't need supervision - it will run each of these over the model, and manage things like finding appropriate detectors and handling rate limiting itself. You can override many aspects of the config to get a custom scan, but out of the box, it can do a full standard scan and report without intervention.

Connect to many different LLMs
------------------------------

garak supports a ton of LLMs - including OpenAI, Hugging Face, Cohere, Replicate - as well as custom Python integrations. It's a community project, so even more LLM support is always coming.

Structured reporting
--------------------

garak keeps track of everything found, and outputs four kinds of log:


#. Screen output - useful for monitoring scan progress; a precise description of what's happening at any time during the scan, including a list of everything on the schedule
#. Report log - detailing the run down to every single prompt, response, and the evaluation of that response
#. Hit log - describing each time that garak managed to get through and find a vulnerability
#. Debug log - a logfile for troubleshooting and keeping track of garak's operations
