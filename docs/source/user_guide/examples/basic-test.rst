
☑️ Basic test
=============

We can run a super simple self-contained test to check that garak's core code is running OK.

.. code-block::

   $ python -m garak --model_type test.Blank --probes test.Test

This command line will start garak - the bit at the front means "run python and load the module garak" - and specify the model type to be "\ ``test``\ ", an internal testing generator, and then run the probe ``test.Blank``.

.. code-block::

   $ python -m garak --model_type test.Blank --probes test.Test
   garak LLM vulnerability scanner v0.9.0.15.post1 ( https://github.com/leondz/garak ) at 2024-08-14T14:13:23.863402
   📜 logging to .local/share/garak/garak.log
   🦜 loading generator: Test: Blank
   📜 reporting to .local/share/garak/garak_runs/garak.7e777bc4-7ac8-46e4-b071-a00b018389e0.report.jsonl
   🕵️  queue of probes: test.Test
   test.Test                                      always.Pass: PASS  ok on   80/  80         
   📜 report closed :) .local/share/garak/garak_runs/garak.7e777bc4-7ac8-46e4-b071-a00b018389e0.report.jsonl
   📜 report html summary being written to .local/share/garak/garak_runs/garak.7e777bc4-7ac8-46e4-b071-a00b018389e0.report.html
   ✔️  garak run complete in 1.62s

We can see that garak ran OK. It loaded a generator called Blank, which is a test generator that always returns a blank string, "". test.Blank is the automatically-used default generator whenever the test model is used. Then, garak queued up and ran just one probe, test.Blank, which sends blank strings to the generator. So, test.Blank sent empty strings to a generator that always returns empty strings. These outputs were evaluated using the always.Pass detector, which (as you can guess from its name) returned a Pass regardless of the output it was assessing. The final score was 10/10, a pass.

{% hint style="info" %}
The score is 10/10 and not 1/1 because by default, garak collects ten outputs per prompt. Because most LLM systems behave differently each time they're queried, we need to get an idea of model tendencies instead of just individual binary assessments. So, garak has to collect multiple outputs for any prompt; and gets ten by default. You can change this using the ``--generations`` command line parameter, e.g. ``--generations 4``.
{% endhint %}

Looks like we passed the test! That's good.
