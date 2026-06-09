import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_palette("Set2")
import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
test = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")

print(f"Training Rows: {len(train)}, Columns: {train.shape[1]}")
print(f"Test Rows: {len(test)}, Columns: {test.shape[1]}")

print("\nTrain Columns →", list(train.columns))
print("Test Columns →", list(test.columns))



print("\n===== BASIC LOOK AT TRAIN =====")
display(train.sample(5))

print("\n===== NULL VALUE SUMMARY =====")
missing_info = train.isnull().sum().reset_index()
missing_info.columns = ["column", "missing_count"]
display(missing_info)

print("\n===== DATA TYPES =====")
display(train.dtypes.to_frame("dtype"))



target_summary = {
    "A_wins": train["winner_model_a"].sum(),
    "B_wins": train["winner_model_b"].sum(),
    "Tie": train["winner_tie"].sum()
}

target_df = pd.Series(target_summary)
display(target_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

target_df.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336", "#2196F3"])
axes[0].set_title("Win Counts")

axes[1].pie(target_df.values, labels=target_df.index, autopct="%1.1f%%")
axes[1].set_title("Win Percentages")

plt.show()



train["len_prompt"] = train["prompt"].str.len()
train["len_a"] = train["response_a"].str.len()
train["len_b"] = train["response_b"].str.len()

train["wc_prompt"] = train["prompt"].str.split().str.len()
train["wc_a"] = train["response_a"].str.split().str.len()
train["wc_b"] = train["response_b"].str.split().str.len()

summary_cols = ["len_prompt", "len_a", "len_b", "wc_prompt", "wc_a", "wc_b"]
display(train[summary_cols].describe())



fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.kdeplot(train["len_prompt"], ax=axes[0, 0], fill=True)
axes[0, 0].set_title("Prompt Length KDE")

sns.kdeplot(train["len_a"], ax=axes[0, 1], fill=True)
axes[0, 1].set_title("Response A Length KDE")

sns.kdeplot(train["len_b"], ax=axes[0, 2], fill=True)
axes[0, 2].set_title("Response B Length KDE")

sns.kdeplot(train["wc_prompt"], ax=axes[1, 0], fill=True)
axes[1, 0].set_title("Prompt Word Count KDE")

sns.kdeplot(train["wc_a"], ax=axes[1, 1], fill=True)
axes[1, 1].set_title("Response A Word Count KDE")

sns.kdeplot(train["wc_b"], ax=axes[1, 2], fill=True)
axes[1, 2].set_title("Response B Word Count KDE")

plt.tight_layout()
plt.show()



print("\nUnique Models (A):", train["model_a"].nunique())
print("Unique Models (B):", train["model_b"].nunique())

top_a = train["model_a"].value_counts().head(12)
top_b = train["model_b"].value_counts().head(12)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(x=top_a.values, y=top_a.index, ax=ax[0])
ax[0].set_title("Top 12 Models Appearing as Model A")

sns.barplot(x=top_b.values, y=top_b.index, ax=ax[1])
ax[1].set_title("Top 12 Models Appearing as Model B")

plt.show()



def identify_winner(r):
    if r["winner_model_a"] == 1:
        return "A"
    if r["winner_model_b"] == 1:
        return "B"
    return "Tie"

train["winner"] = train.apply(identify_winner, axis=1)
train["winner"].value_counts()



fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=train, x="winner", y="len_a", ax=axes[0])
axes[0].set_title("Response A Length vs Winner")

sns.boxplot(data=train, x="winner", y="len_b", ax=axes[1])
axes[1].set_title("Response B Length vs Winner")

plt.show()



corr_cols = ["len_prompt", "len_a", "len_b", "wc_prompt", "wc_a", "wc_b"]
corr_matrix = train[corr_cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix of Text Features")
plt.show()



train["diff_len"] = train["len_a"] - train["len_b"]
train["diff_words"] = train["wc_a"] - train["wc_b"]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(data=train, x="diff_len", hue="winner", kde=True, ax=ax[0])
ax[0].set_title("Difference in Length: A - B")

sns.histplot(data=train, x="diff_words", hue="winner", kde=True, ax=ax[1])
ax[1].set_title("Difference in Word Count: A - B")

plt.show()



print("\n===== SAMPLE TRAINING EXAMPLES =====")

for idx in np.random.choice(train.index, 3, replace=False):
    row = train.loc[idx]
    print("\nPrompt:", row["prompt"][:180], "...")
    print("A:", row["response_a"][:180], "...")
    print("B:", row["response_b"][:180], "...")
    print("Winner →", row["winner"])
    print("-" * 70)



print("\n=== EDA COMPLETED SUCCESSFULLY ===")





