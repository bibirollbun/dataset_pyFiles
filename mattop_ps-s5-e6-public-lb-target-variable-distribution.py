import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

TARGET = "Fertilizer Name"


print(train[TARGET].unique()) # Unique targets in training set


sub.head(3) # What the submission looks like


# Create submission files
for f in train[TARGET].unique()[:-1]:
    sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
    sub[TARGET] = f + " " + "blank" + " " + "blank"
    sub.to_csv(f"{f}_sub.csv", index=False)


lb_n_samples = int(len(test) * 0.2) # Public lb size (20%)
print(lb_n_samples)


public_lb_scores = [0.14728, 0.14766, 0.15616, 0.12598, 0.14716, 0.15210]
lb_counts = []
count = 0

for fert, score in zip(train[TARGET].unique()[:-1], public_lb_scores):
    sub[TARGET] = fert + " " + "blank" + " " + "blank"
    total_samples = int(np.ceil(score * lb_n_samples))
    print(f"{TARGET}: {fert} | Public LB score: {score} | {score} * {lb_n_samples} = {int(score * lb_n_samples)} total samples \n")
    count += total_samples
    lb_counts.append(total_samples)
    print("Submission file:")
    display(sub.head(3))
    print()

urea_n_samples = int(lb_n_samples - count)
print(f"We can conclude from the data above that Urea has {urea_n_samples} total samples.\n")
lb_counts.append(urea_n_samples)
print(f"Total samples: {sum(lb_counts)}")


target_counts = train[TARGET].value_counts(sort=False)

df = pd.DataFrame({TARGET: list(target_counts.index) * 2,
                  "target_count": list(target_counts.values) + lb_counts, # public LB target counts
                  "type": ["train"] * 7 + ["public_LB"] * 7})

train_count = df.iloc[:7]["target_count"]
train_count_norm = train_count / train_count.sum()

public_LB_count = df.iloc[7:]["target_count"]
public_LB_count_norm = public_LB_count / public_LB_count.sum()

normalized_counts = pd.concat([train_count_norm, public_LB_count_norm])
df["normalized_target_count"] = normalized_counts.values


df


plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (12, 5))

sns.barplot(df.query("type == 'public_LB'"), x = "target_count", y = TARGET,
            color = "#0066ff", edgecolor = "#FFFFFF", linewidth = 1, width = 0.6, ax = ax1)

sns.barplot(df.query("type == 'train'"), x = "target_count", y = TARGET,
            color = "#33cc33", edgecolor = "#FFFFFF", linewidth = 1, width = 0.6, ax = ax2)

for container1, container2 in zip(ax1.containers, ax2.containers):
    ax1.bar_label(container1, padding=3, fontsize=10)
    ax2.bar_label(container2, padding=3, fontsize=10)
    
ax1.set_title("public LB")
ax2.set_title("train")
ax1.set_xlim(0, 8_800)
ax2.set_xlim(0, 135_000)
plt.grid(False)
plt.yticks(fontsize = 10)
plt.suptitle("Target value counts in public LB & train set")
plt.show()


plt.figure(figsize = (10, 8))

ax = sns.barplot(df, x = round(df["normalized_target_count"], 3) * 100, y = TARGET, hue = "type",
                 palette = ["#33cc33", "#0066ff"], edgecolor = "#FFFFFF", linewidth = 1, width = 0.6)

for container in ax.containers:
    ax.bar_label(container, padding = 3, fontsize = 8)
    
plt.xlabel("Target Value Percentage")
plt.grid(False)
plt.yticks(fontsize = 10)
plt.title("Target value percentages in public LB & train set")
plt.show()


n_obs = df.target_count[7:].values
n_exp = df.target_count[:7].values
scipy.stats.chisquare(n_obs, n_exp/np.sum(n_exp)*np.sum(n_obs))

