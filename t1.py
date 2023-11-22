from random import shuffle
import inspect
import string

from openai import OpenAI
client = OpenAI()


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



def calc(x):
    return 1 if "a" in x else 0


func_code = inspect.getsource(calc)

correct_guesses = 0
correct_rule = 0

for i in range(10):

    pairs = []
    for i in range(100):
        letters = list(string.ascii_lowercase[:6])
        shuffle(letters)
        txt = "".join(letters[:3])
        pairs.append((txt, calc(txt)))

    #   TODO: select half 0 and half 1
    shuffle(pairs)
    pairs = pairs[:10]


    train = "\n".join([f"Input: {pair[0]} Label: {pair[1]}" for pair in pairs[:-1]])
    # print(train)
    # 1/0
    test = pairs[-1][0]

    get_label_task_description = get_label_task_description_template.format(train=train, test=test)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": get_label_task_description}],
        temperature=0,
    )

    result = completion.choices[0].message.content

    print(i, int(result) == pairs[-1][1], pairs[-1][0], result)

    if int(result) == pairs[-1][1]:
        correct_guesses += 1

    get_rule_task_description = get_rule_task_description_template.format(train=train)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": get_rule_task_description}],
        temperature=1,
    )

    rule = completion.choices[0].message.content
    print(rule)

    evaluate_rule_task_descripion = evaluate_rule_task_descripion_template.format(func_code=func_code, rule=rule)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": evaluate_rule_task_descripion}],
        temperature=0,
    )
    evaluation = completion.choices[0].message.content
    if evaluation == "Yes":
        correct_rule += 1

    print(evaluation)

    print()

print("GUESSED CORRECLY", correct_guesses)
print("NAMED RULE", correct_rule)
