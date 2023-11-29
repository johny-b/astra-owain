# %%
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


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
