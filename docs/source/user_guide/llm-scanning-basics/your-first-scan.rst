
🚀 Your first scan
==================

{% hint style="danger" %}
Only run garak on systems that you have permission to use! It performs an exhaustive scan with some pretty strong prompts that might be misinterpreted.
{% endhint %}

So you've installed garak - great! Let's run a scan to see what's up.

To keep things simple, let's try a straightforward target that could run on most machines - gpt2. Hugging Face Hub has a free and openly available version of this, so let's use that. And let's try a relatively straightforward probe, one that tries to get rude generations from the model.

First we'd like to see the list of probes available. Open a terminal if you don't have one up already, and run garak by entering the following then pressing enter:

.. code-block::

   python -m garak --list_probes

You should get a list of probes in garak, with symbols at the side of some lines. You can always read more about garak's options by running ``python -m garak --help`` (or sometimes just ``garak --help``\ , depending on your installation).

Back to the list of probes. It might look like this.

.. code-block::

   garak LLM security probe v0.9.0.6 ( https://github.com/leondz/garak ) at 2023-07-24T11:20:16.086762
   probes: art 🌟
   probes: art.Tox
   probes: continuation 🌟
   probes: continuation.ContinueSlursReclaimedSlurs50
   probes: dan 🌟
   probes: dan.Ablation_Dan_11_0
   probes: dan.AntiDAN
   probes: dan.ChatGPT_Developer_Mode_RANTI
   probes: dan.ChatGPT_Developer_Mode_v2
   probes: dan.ChatGPT_Image_Markdown
   probes: dan.DAN_Jailbreak
   probes: dan.DUDE
   probes: dan.Dan_10_0
   probes: dan.Dan_11_0
   probes: dan.Dan_6_0
   probes: dan.Dan_6_2
   probes: dan.Dan_7_0
   probes: dan.Dan_8_0
   probes: dan.Dan_9_0
   probes: dan.STAN
   probes: encoding 🌟
   probes: encoding.InjectAscii85
   probes: encoding.InjectBase16
   probes: encoding.InjectBase2048
   probes: encoding.InjectBase32
   probes: encoding.InjectBase64
   probes: encoding.InjectBraille
   probes: encoding.InjectHex
   probes: encoding.InjectMime 💤
   probes: encoding.InjectMorse
   probes: encoding.InjectQP 💤
   probes: encoding.InjectROT13
   probes: encoding.InjectUU
   probes: glitch 🌟
   probes: glitch.Glitch 💤
   probes: glitch.Glitch100
   probes: goodside 🌟
   probes: goodside.ThreatenJSON
   probes: goodside.WhoIsRiley
   probes: goodside._Davidjl
   probes: knownbadsignatures 🌟
   probes: knownbadsignatures.EICAR
   probes: knownbadsignatures.GTUBE
   probes: knownbadsignatures.GTphish
   probes: leakreplay 🌟
   probes: leakreplay.LiteratureCloze 💤
   probes: leakreplay.LiteratureCloze80
   probes: leakreplay.LiteratureComplete 💤
   probes: leakreplay.LiteratureComplete80
   probes: lmrc 🌟
   probes: lmrc.Anthropomorphisation
   probes: lmrc.Bullying
   probes: lmrc.Deadnaming
   probes: lmrc.Profanity
   probes: lmrc.QuackMedicine
   probes: lmrc.SexualContent
   probes: lmrc.Sexualisation
   probes: lmrc.SlurUsage
   probes: malwaregen 🌟
   probes: malwaregen.Evasion
   probes: malwaregen.Payload
   probes: malwaregen.SubFunctions
   probes: malwaregen.TopLevel
   probes: misleading 🌟
   probes: misleading.FalseAssertion50
   probes: promptinject 🌟
   probes: promptinject.HijackHateHumans 💤
   probes: promptinject.HijackHateHumansMini
   probes: promptinject.HijackKillHumans 💤
   probes: promptinject.HijackKillHumansMini
   probes: promptinject.HijackLongPrompt 💤
   probes: promptinject.HijackLongPromptMini
   probes: realtoxicityprompts 🌟
   probes: realtoxicityprompts.RTPBlank
   probes: realtoxicityprompts.RTPFlirtation
   probes: realtoxicityprompts.RTPIdentity_Attack
   probes: realtoxicityprompts.RTPInsult
   probes: realtoxicityprompts.RTPProfanity
   probes: realtoxicityprompts.RTPSevere_Toxicity
   probes: realtoxicityprompts.RTPSexually_Explicit
   probes: realtoxicityprompts.RTPThreat
   probes: snowball 🌟
   probes: snowball.GraphConnectivity 💤
   probes: snowball.GraphConnectivityMini
   probes: snowball.Primes 💤
   probes: snowball.PrimesMini
   probes: snowball.Senators 💤
   probes: snowball.SenatorsMini
   probes: test 🌟
   probes: test.Blank 💤
   probes: xss 🌟
   probes: xss.MarkdownImageExfil

That's good! The stars 🌟 indicate a whole plugin; if we also garak to run them, it will run all the probes in that category. Except the disabled ones, marked with 💤. We can run disabled probes my naming them directly when we start a garak run, but they won't be selected automatically.

Under ``lmrc`` we can see a probe named "\ ``lmrc.Profanity``\ ". Let's try this one. LMRC stands for "Language Model Risk Cards", and the probes in here come from a framework for `assessing language model deployments <https://arxiv.org/abs/2303.18190>`_. But for now let's just run the profanity probe.

To recap - we'll run a Hugging Face model, called gpt2, and use the lmrc.Profanity probe. So, our command line is:

``python -m garak --model_type huggingface --model_name gpt2 --probes lmrc.Profanity``

Typing that in and pressing enter should start the garak run! It will download gpt2 if you don't have it already - there'll be some progress bars about that - and then it will start the scan.

If all goes well, you should see a progress bar and then a number of lines each saying PASS or FAIL. Congratulations! You're run your first garak scan. The next section covers how to read these results.
