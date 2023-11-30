# %%
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.stats import fisher_exact

# %%
def get_data(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            row_data = json.loads(line)
            if "sample_ix" in row_data:
                data.append(row_data)
    return data


# %%
model_name = "gpt-4-0613"
variants = ['ab_10', 'ab_50', 'cdef_10', 'cdef_50']
raw_data = [get_data(f"{model_name}/{variant}.log") for variant in variants]

# %%
#   1. Main plot
def true_ratio(data, key):
    return len([row for row in data if row[key]]) / len(data)

values_A = [true_ratio(data, "correct_label") for data in raw_data]
values_B = [true_ratio(data, "correct_rule") for data in raw_data]

x = np.arange(len(variants))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_A, width, label='Get label')
rects2 = ax.bar(x + width/2, values_B, width, label='Get rule')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Variants')
ax.set_ylabel('Percentage Correct')
ax.set_title(f'Percentage of Correct Responses by Variant and Task - {model_name}')
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

# %%
#   2.  Is the difference between 10 and 50 statistically significant?
sample_size = 100
print("ab",   sm.stats.proportions_ztest([0.68 * sample_size, 0.88 * sample_size], [sample_size, sample_size]))
print("cdef", sm.stats.proportions_ztest([0.87 * sample_size, 0.97 * sample_size], [sample_size, sample_size]))
print("ab - rule",   sm.stats.proportions_ztest([0.16 * sample_size, 0.02 * sample_size], [sample_size, sample_size]))
print("cdef - rule", sm.stats.proportions_ztest([0.16 * sample_size, 0 * sample_size], [sample_size, sample_size]))

# %%
#   3.  Analyze rules
for variant, data in zip(variants, raw_data):
    rules = [str(x["correct_rule"]) + " " + x["rule"] for x in data]
    rules.sort()
    print("VARIANT", variant, len(Counter(rules)))
    print("RULES WITH 'a' or 'b'", len(list(rule for rule in rules if "'a'" in rule or "'b'" in rule)))
    for rule, cnt in Counter(rules).items():
        print(cnt, rule)

# %%
#   4.  Are there easier/harder samples for both experiments?
for variant in ("ab_10", "cdef_10"):
    variant_data = raw_data[variants.index(variant)]

    data = []
    for sample_data in variant_data:
        data.append((sample_data["correct_label"], sample_data["correct_rule"]))

    df = pd.DataFrame(data, columns=['correct_label', 'correct_rule'])

    # Create a frequency table (contingency table)
    frequency_table = pd.crosstab(df['correct_label'], df['correct_rule'])

    print(frequency_table)

    # Assuming frequency_table is a 2x2 DataFrame
    _, p_value = fisher_exact(frequency_table)

    print("P-value:", p_value)

# %%
#   5.  Compare number of different tokens per task
#       (this doesn't make much sense)
import tiktoken
from experiment import starts_a_ends_b_strings

encoder = tiktoken.encoding_for_model(model_name)
for letters in ("ab", "cdef"):
    strings = starts_a_ends_b_strings(letters, 8)
    tokens = [encoder.encode(string) for string in strings]
    print(letters)
    print("AVERAGE TOKENS", sum(len(x) for x in tokens)/len(tokens))
    print("UNIQUE FIRST TOKENS", len(set(x[0] for x in tokens)))
    print("UNIQUE LAST  TOKENS", len(set(x[-1] for x in tokens)))




# %%
