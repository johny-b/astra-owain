from itertools import combinations, permutations

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

def get_strings():
    result = []
    for selected_letters in list(combinations('abcdef', 6)) + list(combinations('abcdef', 5)):
        result += ["".join(x) for x in permutations(selected_letters)]
    return result

#   TODO: this is probably not balanced (rule "starts with a" might often be correct)
setup = Setup(txt_starts_and_len, get_strings())

import os
print(os.environ["OPENAI_API_KEY"])

eval = Eval("gpt-4", setup)
results = eval.run(sample_size=30, n_samples=10)

print("CORRECT GUESS", sum(x["correct_label"] for x in results))
print("CORRECT RULE", sum(x["correct_rule"] for x in results))
