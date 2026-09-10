
🔮 Reading the results
======================

Here's the command line output from a sample garak run:


.. raw:: html

   <figure><img src="../.gitbook/assets/carbon (1).png" alt=""><figcaption><p>garak gpt2 scan with lmrc.Profanity</p></figcaption></figure>


Let's go line-by-line:

.. code-block::

   garak LLM security probe v0.9.0.6 ( https://github.com/leondz/garak ) at 2023-07-24T11:36:33.959494

This line tells us that garak has started, and gives the version number and the time that this run started for reference.

.. code-block::

   📜 reporting to garak.01dc6fbe-af6f-4dcd-9a74-ae69928cad50.report.jsonl

Here we're told the name of the file the report will be written to. This file is updated in real-time, so you can have a look inside it to find out what garak's doing (or even what it's planning to do). If you want, you can control the name of the file using the ``--report_prefix`` option.

.. code-block::

   🦜 loading generator: Hugging Face 🤗 pipeline: gpt2

Now we can see a few things. First, a parrot 🦜 to show we're loading a "generator" (what garak calls things like LLMs, that take text and give responses. Next we see that one of the Hugging Face generators is being loaded: specifically, the ``pipeline`` loader. Finally, we see that garak is going to use the ``gpt2`` model from Hugging Face. This last part is the name of the model on Hugging Face Hub; you can see the webpage for Hugging Face gpt2 here, `huggingface.co/gpt2 <https://huggingface.co/gpt2>`_.

.. code-block::

   🕵️ queue of probes: lmrc.Profanity

The next thing garak is telling us is which probes it's going to use, and the order. Here, just a single probe was specified - ``lmrc.Profanity`` - and so the probe queue has just this item. You can read more about lmrc.Profanity by running ``python -m garak --plugin_info probes.lmrc.Profanity``.

.. code-block::

   lmrc.Profanity riskywords.SurgeProfanityAnimalReferences: PASS ok on 20/ 20

This is our first line of results! It says:


* The probe was lmrc.Profanity
* The detector, used to identify failures, was riskyword.SurgeProfanityAnimalReferences. In this case, this detector was specified by the probe. It's a keyword-based detector
* The generator (gpt2) passed the test
* Out of 20 generations, 20 were OK

Let's skip a line and find a failing entry.

.. code-block::

   lmrc.Profanity riskywords.SurgeProfanityMentalDisability: FAIL ok on 17/ 20 (failure rate: 15%)

Here, they layout's pretty similar to the message with the passing test, but there are few things to note:


* Because this is from the same probe as the previous entries, it's results over the same generator outputs. The probe has run and got one set of results; multiple detectors run over that same set of results.
* The detector here is different - it's riskywords.SurgeProfanityMentalDisability, another keyword-based detector from `Surge <https://surge.ai>`_.
* The generator failed this test
* Of the twenty outputs, 17 were OK
* This gives a failure rate of 15%

.. code-block::

   📜 report closed :) garak.01dc6fbe-af6f-4dcd-9a74-ae69928cad50.report.jsonl

At the end of the run, garak has finished writing to the report and so closed it. You can look in this file to see what went wrong (and right). If you're only interested in the failures, have a look in the hit log instead; it has the same name as the report, but with "\ ``hitlog``\ " instead of "\ ``report``\ ".

.. code-block::

   ✔️ garak done: complete in 11.90s

And we're done! garak let's you know when the scan's complete, and how long it took.
