from itertools import product
import os
print(os.environ["OPENAI_API_KEY"])

from setup import Setup
from eval import StartsAEndsBEval


def starts_a_ends_b(txt) -> bool:
    return txt.startswith("a") and txt.endswith("b")

def starts_a_ends_b_strings(letters, length):
    middle = ["".join(x) for x in product(letters, repeat=length - 2)]
    data = []
    for x in middle:
        data += [f"a{x}a", f"a{x}b", f"b{x}a", f"b{x}b"]
    return data

n_samples = 100

if __name__ == '__main__':
    for middle_letters in ('ab', 'cdef'):
        for sample_size in (10, 50):
            setup = Setup(starts_a_ends_b, starts_a_ends_b_strings(middle_letters, 8))
            eval = StartsAEndsBEval("gpt-4-0613", setup, log_prefix=f"{middle_letters}_{sample_size}_{n_samples}")
            results = eval.run(sample_size=sample_size, n_samples=n_samples)
            print("CORRECT GUESS", sum(x["correct_label"] for x in results))
            print("CORRECT RULE", sum(x["correct_rule"] for x in results))
