from itertools import product

from setup import Setup
from eval import Eval


def starts_a(txt) -> bool:
    return txt.startswith("a")

def starts_a_strings():
    no_a = ["".join(x) for x in product('bcdefghij', repeat=5)]
    data = []
    for x in no_a:
        data.extend([x, "a" + x[1:]])
    return data

n_samples = 10
sample_size = 10
setup = Setup(starts_a, starts_a_strings())
eval = Eval("gpt-4", setup, log_prefix=f"simple_function_{sample_size}_{n_samples}")
results = eval.run(sample_size=sample_size, n_samples=n_samples)
print("CORRECT GUESS", sum(x["correct_label"] for x in results))
print("CORRECT RULE", sum(x["correct_rule"] for x in results))
