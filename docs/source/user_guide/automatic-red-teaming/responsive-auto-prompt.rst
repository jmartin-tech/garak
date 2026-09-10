
🤼 Responsive auto-prompt
=========================

Red teaming is useful, and we have data about it, and that data is text, why don’t we train an LLM to do it? One option is to use that data to train a red-teaming model on logs of successful red-teaming attempts, so that it tries to respond to a system in a way that's most likely to get a "hit".

To build a simple pipeline.


* Use an off-the-shelf toxicity detector, martin-ha/toxic-comment-model
* Look at an existing red teaming dataset, the red team attempts from Anthropic’s hhrlhf
* Find system dialog turns that were rated toxic, and extract dialog turns in the conversations that led to that toxicity
* Train a 2019 gpt-2 to red-team based on this data

In this data there are conversation sequences of person-system-person-system-… turns. We want to find things that led to the system giving toxic output. We can then look back to see what the person said in order to get that toxic response - that’s the output we’d like the red-team model to produce. But when our auto-red-teamer is generating text, we’d like it to respond to the system; so we need to start with a system output. As a result, our data looks like this:


* System Response (a)
* Human Input (b)
* [Toxic system response]

Where there are number of (ab) pairs followed by a toxic response. When building training data for an auto-red-team model, we don’t include the toxic system response, but we do want our model to generate things that were successful in leading to toxic system responses. Thus a responsive auto-prompt model for red teaming is trained based on system responses (a) as prompts human inputs (b) as responses, including special empty-prompt “opener” pairs, all taken from conversations that resulted in toxicity.
