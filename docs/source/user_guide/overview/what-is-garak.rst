
💡 What is garak?
=================

garak finds holes in LLM-based tech
-----------------------------------

We'll see more use of language models in technologies, systems, apps and services. We don't know how to secure language models, yet. garak works with language models and any tech using them to determine where the security holes can be with each solution.

garak identifies how your LLM can fail
--------------------------------------

With dozens of different plugins, a hundred different probes, and tens of thousands of challenging prompts, garak tries hard to explore many different LLM failure modes.

garak reports each failing prompt and response
----------------------------------------------

Once garak finds something, the exact prompt, goal, and response is reported, so you get a full log of everything that's worth checking out and why it might be a problem.

garak adapts itself over time
-----------------------------

The garak community constantly adds new, aggressive probes, and sharpens the probes that are in the toolkit. This means garak's coverage is always increasing, and the same target's scores in a garak run will gradually decrease, because garak is constantly changing.

Each LLM failure found goes into a "hit log". These logs can then be used to train garak's "auto red-team" feature into finding effective exploitation strategies, meaning a more thorough testing and more chance of finding LLM security holes before anyone else does.
