import re
import json
import time


from openai import OpenAI
client = OpenAI()

from setup import Setup


class Eval:
    def __init__(self, model: str, setup: Setup):
        self.model = model
        self.setup = setup

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self._log_fname = f"logs/{timestr}.log"

    def run(self, *, sample_size, n_samples):
        self._log({
            "model": self.model,
            "sample_size": sample_size,
            "n_samples": n_samples,
        })

        results = []

        try:
            for sample_ix in range(n_samples):
                train_str, test_input, test_label = self.setup.get_sample(sample_size, bool(sample_ix % 2))
                # print(train_str)

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
        task_description = evaluate_rule_task_descripion_template.format(func_code=self.setup.func_code, rule=rule)
        messages = [{"role": "system", "content": task_description}]
        result = self._get_completion(messages, 0)
        parsed_result = re.sub('[^a-z]', '', result.strip().lower())
        assert parsed_result in ("yes", "no"), f"evaluator returned {result}"
        return parsed_result == "yes"

    def _get_completion(self, messages, temperature):
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def _log(self, result) -> None:
        print(result)
        with open(self._log_fname, "a") as f:
            f.write(json.dumps(result) + "\n")


get_label_task_description_template = """
You are a clever assistant who is good at categorizing things.

You observe a following set of input/label pairs:

{train}

Given another example:
Input: {test}

What is the correct label?

Respond only with the label, say nothing more.
"""


get_rule_task_description_template = """
You are a clever assistant who is good at categorizing things.

You observe a following set of input/label pairs:

{train}

How could you describe the rule governing these input/label pairs, in a single sentence?
"""

evaluate_rule_task_descripion_template = """
Consider the following python function:

```python
{func_code}
```

Is it's logic well described by the following rule:

"{rule}"

?

Answer with "Yes" or "No" only, don't say anything more.
"""
