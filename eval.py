import re
import json
import time


from openai import OpenAI
client = OpenAI()

from setup import Setup


class Eval:
    def __init__(self, model: str, setup: Setup, log_prefix: str = ""):
        self.model = model
        self.setup = setup

        self.current_sample_ix = None

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self._log_fname = f"logs/{log_prefix}_{timestr}.log"
        self._completion_log_fname = f"logs/completion_{log_prefix}_{timestr}.log"

    def run(self, *, sample_size, n_samples):
        self._log({
            "model": self.model,
            "sample_size": sample_size,
            "n_samples": n_samples,
        })

        results = []

        try:
            for sample_ix in range(n_samples):
                self.current_sample_ix = sample_ix

                train_str, test_input, test_label = self.setup.get_sample(sample_size, bool(sample_ix % 2))

                label = self._get_label(train_str, test_input)
                rule = self._get_rule(train_str)
                correct_rule = self._evaluate_rule(rule)

                result ={
                    "sample_ix": sample_ix,
                    "test_input": test_input,
                    "test_label": test_label,
                    "correct_label": label.strip() == str(test_label),
                    "correct_rule": correct_rule,
                    "label": label,
                    "rule": rule,
                    "train_str": train_str,
                }

                self._log(result)
                results.append(result)
        except KeyboardInterrupt:
            pass

        return results

    def _get_label(self, train_str, test_input) -> str:
        task_description = get_label_task_description_template.format(train=train_str, test=test_input)
        messages = [{"role": "system", "content": task_description}]
        return self._get_completion(messages, 0)

    def _get_rule(self, train_str) -> str:
        task_description = get_rule_task_description_template.format(train=train_str)
        messages = [{"role": "system", "content": task_description}]
        return self._get_completion(messages, 0)

    def _evaluate_rule(self, rule: str) -> bool:
        task_description = self._get_evaluate_rule_task_description(rule)
        messages = [{"role": "system", "content": task_description}]
        result = self._get_completion(messages, 0)
        parsed_result = re.sub('[^a-z]', '', result.strip().lower())
        assert parsed_result in ("yes", "no"), f"evaluator returned {result}"
        return parsed_result == "yes"

    def _get_evaluate_rule_task_description(self, rule: str) -> str:
        return evaluate_rule_task_descripion_template.format(func_code=self.setup.func_code, rule=rule)

    def _get_completion(self, messages, temperature):
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        result = completion.choices[0].message.content
        self._log_completion(messages, temperature, result)
        return result

    def _log(self, result) -> None:
        print(result)
        with open(self._log_fname, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _log_completion(self, messages, temperature, completion):
        data = {
            "sample_ix": self.current_sample_ix,
            "temperature": temperature,
            "model": self.model,
            "messages": messages,
            "completion": completion,
        }
        with open(self._completion_log_fname, "a") as f:
            f.write(json.dumps(data) + "\n")


class StartsAEndsBEval(Eval):
    def _get_evaluate_rule_task_description(self, rule: str) -> str:
        return starts_a_ends_b_eval.format(rule=rule)


get_label_task_description_template = """
You are a clever assistant who is good at categorizing things.

You observe the following set of input/label pairs:

{train}

Given another example:

Input: {test}

What is the correct Label?

Respond only with the correct label, say nothing more.
"""


get_rule_task_description_template = """
You are a clever assistant who is good at categorizing things.

You observe the following set of input/label pairs:

{train}

How would you describe the rule governing these input/label pairs, in a single sentence?
"""

evaluate_rule_task_descripion_template = """
Consider the following python function:

```python
{func_code}
```

Is its logic well described by the following rule:

"{rule}"

?

Answer with "Yes" or "No" only, don't say anything more.
"""

starts_a_ends_b_eval = """
Consider the following rule governing sets of input/output pairs:

"{rule}"

Is this exactly equivalent to "Label is True if input starts with 'a' and ends with 'b'"?

Answer with "Yes" or "No" only, don't say anything more.
"""
