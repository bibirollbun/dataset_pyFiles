# Loading data and packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_path = Path("/kaggle/input").glob("*/train.csv").__next__()
test_path = Path("/kaggle/input").glob("*/test.csv").__next__()
example_submission = Path("/kaggle/input").glob("*/sample_submission.csv").__next__()

print("training path", train_path)
print("test path", test_path)
print("example_submission", example_submission)


# Splitting data in extro and intro
train_data = pd.read_csv(train_path)
train_intro = train_data[train_data.Personality == "Introvert"]
train_extro = train_data[train_data.Personality == "Extrovert"]

print(f"Total data: {len(train_data)} with {len(train_intro)} Intro and {len(train_extro)} Extro and {len(train_data) - len(train_intro) - len(train_extro)} others")


# Plotting Stage fear, Drained after socializing, and groups are okay
fig, axes = plt.subplots(2, 2, figsize=(16,6))

# Stage fear
intro_stage = train_intro.Stage_fear.value_counts(dropna=False)
extro_stage = train_extro.Stage_fear.value_counts(dropna=False)
ax = axes.flatten()
labels_raw = list(extro_stage.keys())
extro_val = []
intro_val = []
labels = []
for l in labels_raw:
    extro_val.append(int(extro_stage[l]))
    intro_val.append(int(intro_stage[l]))
    labels.append(str(l))

width = 0.35       # the width of the bars: can also be len(x) sequence


ax[0].bar(labels, extro_val, width, label='Extroverts')
ax[0].bar(labels, intro_val, width, bottom=extro_val, label='Introverts')

ax[0].set_ylabel('Occurence')
ax[0].set_title('Stage fear')
ax[0].legend()

# Drained_after_socializing
intro_Drained_after_socializing = train_intro.Drained_after_socializing.value_counts(dropna=False)
extro_Drained_after_socializing = train_extro.Drained_after_socializing.value_counts(dropna=False)
ax = axes.flatten()
labels_raw = list(extro_stage.keys())
extro_val = []
intro_val = []
labels = []
for l in labels_raw:
    extro_val.append(int(extro_Drained_after_socializing[l]))
    intro_val.append(int(intro_Drained_after_socializing[l]))
    labels.append(str(l))

width = 0.35       # the width of the bars: can also be len(x) sequence


ax[1].bar(labels, extro_val, width, label='Extroverts')
ax[1].bar(labels, intro_val, width, bottom=extro_val, label='Introverts')

ax[1].set_ylabel('Occurence')
ax[1].set_title('Drained_after_socializing')
ax[1].legend()

# Drained_after_socializing & Stage fear
ax = axes.flatten()
labels_raw = list(extro_stage.keys())
extro_val = []
intro_val = []
labels = ["fear & drained"]
extro_val.append(sum((train_extro.Stage_fear == "Yes") & (train_extro.Drained_after_socializing == "Yes")))
intro_val.append(sum((train_intro.Stage_fear == "Yes") & (train_intro.Drained_after_socializing == "Yes")))
labels.append("No-Yes")
extro_val.append(sum((train_extro.Stage_fear == "No") & (train_extro.Drained_after_socializing == "Yes")))
intro_val.append(sum((train_intro.Stage_fear == "No") & (train_intro.Drained_after_socializing == "Yes")))
labels.append("Yes-No")
extro_val.append(sum((train_extro.Stage_fear == "Yes") & (train_extro.Drained_after_socializing == "No")))
intro_val.append(sum((train_intro.Stage_fear == "Yes") & (train_intro.Drained_after_socializing == "No")))
labels.append("No-No")
extro_val.append(sum((train_extro.Stage_fear == "No") & (train_extro.Drained_after_socializing == "No")))
intro_val.append(sum((train_intro.Stage_fear == "No") & (train_intro.Drained_after_socializing == "No")))
labels.append("1-NaN")
extro_val.append(sum(pd.isna(train_extro.Stage_fear) ^ pd.isna(train_extro.Drained_after_socializing)))
intro_val.append(sum(pd.isna(train_extro.Stage_fear) ^ pd.isna(train_intro.Drained_after_socializing)))
labels.append("NaN-NaN")
extro_val.append(sum(pd.isna(train_extro.Stage_fear) & pd.isna(train_extro.Drained_after_socializing)))
intro_val.append(sum(pd.isna(train_extro.Stage_fear) & pd.isna(train_intro.Drained_after_socializing)))

width = 0.35       # the width of the bars: can also be len(x) sequence


ax[2].bar(labels, extro_val, width, label='Extroverts')
ax[2].bar(labels, intro_val, width, bottom=extro_val, label='Introverts')

ax[2].set_ylabel('Occurence')
ax[2].set_title('Drained_after_socializing')
ax[2].legend()

# Groups are okay
train_intro.loc[:, "Groups_are_okay"] = (train_intro.Stage_fear == "No") | (train_intro.Drained_after_socializing == "No")
train_extro.loc[:, "Groups_are_okay"] = (train_extro.Stage_fear == "No") | (train_extro.Drained_after_socializing == "No")

intro_stage = train_intro.Groups_are_okay.value_counts(dropna=False)
extro_stage = train_extro.Groups_are_okay.value_counts(dropna=False)
ax = axes.flatten()
labels_raw = list(extro_stage.keys())
extro_val = []
intro_val = []
labels = []
for l in labels_raw:
    extro_val.append(int(extro_stage[l]))
    intro_val.append(int(intro_stage[l]))
    labels.append(str(l))

width = 0.35       # the width of the bars: can also be len(x) sequence


ax[3].bar(labels, extro_val, width, label='Extroverts')
ax[3].bar(labels, intro_val, width, bottom=extro_val, label='Introverts')

ax[3].set_ylabel('Occurence')
ax[3].set_title('Groups are okay')
ax[3].legend()

plt.show()


# Make prediction based on "Groups are okay"
predictions = pd.read_csv(example_submission)
test_data = pd.read_csv(test_path)
test_data.loc[:, "Groups_are_okay"] = (test_data.Stage_fear == "No") | (test_data.Drained_after_socializing == "No")

predictions.loc[predictions.id.isin(test_data[test_data.Groups_are_okay].id), "Personality"] = "Extrovert"
predictions.loc[predictions.id.isin(test_data[test_data.Groups_are_okay == False].id), "Personality"] = "Introvert"

predictions.to_csv("submission.csv", index=False)
predictions.head()

