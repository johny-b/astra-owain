# %%
import json
import matplotlib.pyplot as plt
import numpy as np


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
variants = ['ab_10', 'ab_50', 'cdef_10', 'cdef_50']
raw_data = [get_data(f"clean_files/{variant}.log") for variant in variants]

# %%
# MAIN PLOT
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
ax.set_title('Percentage of Correct Responses by Variant and Answer Type')
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
