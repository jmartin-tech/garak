
💯 Scan evaluation
==================

How do we determine what passes? Using a garak evaluator.

Once the harness has had the probes send prompts to the generators and processed the outputs with detectors, the result is a list of scores. Evaluators detemine what scores pass or not. Scores are between 0.0 and 0.5. The default evaluator is ThresholdEvaluator, which considers and score of 0.5 of higher a "hit".
