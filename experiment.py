from itertools import product
import os
import string
print(os.environ["OPENAI_API_KEY"])

from setup import Setup
from eval import Eval



def txt_has_a(txt: str) -> int:
    return 1 if "a" in txt else 0

def txt_starts_with_a_bool(txt: str) -> int:
    return txt.endswith("a")

def txt_starts_with_a_int(txt: str) -> int:
    return 1 if txt.startswith("a") else 0

def txt_starts_ends(txt: str) -> int:
    return 1 if txt.startswith("a") and txt.endswith("c") else 0

def txt_starts_and_len(txt: str) -> int:
    return 1 if txt.startswith("a") and len(txt) == 5 else 0

def starts_a_ends_b(txt) -> bool:
    return txt.startswith("a") and txt.endswith("b")

def starts_a_ends_b_strings(letters, length):
    middle = ["".join(x) for x in product(letters, repeat=length - 2)]
    data = []
    for x in middle:
        data += [f"a{x}a", f"a{x}b", f"b{x}a", f"b{x}b"]
    return data

setup = Setup(starts_a_ends_b, starts_a_ends_b_strings('cd', 8))

eval = Eval("gpt-4-0613", setup)
results = eval.run(sample_size=10, n_samples=10)
print("CORRECT GUESS", sum(x["correct_label"] for x in results))
print("CORRECT RULE", sum(x["correct_rule"] for x in results))
# results = eval.run(sample_size=50, n_samples=100)
# print("CORRECT GUESS", sum(x["correct_label"] for x in results))
# print("CORRECT RULE", sum(x["correct_rule"] for x in results))



