from setup import Setup
from eval import StartsAEndsBCoTEval
from experiment import starts_a_ends_b, starts_a_ends_b_strings

n_samples = 2
for middle_letters in ('ab',):
    for sample_size in (10,):
        setup = Setup(starts_a_ends_b, starts_a_ends_b_strings(middle_letters, 8))
        eval = StartsAEndsBCoTEval("gpt-4-0613", setup, log_prefix=f"cot_{middle_letters}_{sample_size}_{n_samples}")
        results = eval.run(sample_size=sample_size, n_samples=n_samples)
        print("CORRECT GUESS", sum(x["correct_label"] for x in results))
        print("CORRECT RULE", sum(x["correct_rule"] for x in results))

