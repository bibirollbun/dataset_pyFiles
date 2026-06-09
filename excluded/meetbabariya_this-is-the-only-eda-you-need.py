import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

train_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
sample_submission = pd.read_csv("/kaggle/input/mercor-cheating-detection/sample_submission.csv")
labeled_df = train_df[train_df['is_cheating'].notnull()].copy()

feature_cols = [c for c in train_df.columns if c.startswith("feature_")]

rows = []
for col in feature_cols:
    mask = labeled_df[col].isnull()
    rows.append({
        'feature': col,
        'probability_null_is_cheater': labeled_df.loc[mask, 'is_cheating'].mean(),
        'null_count': mask.sum()
    })

null_cheat_df = pd.DataFrame(rows)
null_cheat_df


labeled_df = labeled_df.fillna(-999)
cat_cols = [c for c in feature_cols if c not in ['feature_010','feature_015','feature_016','feature_017','feature_018']]

def plot_categorical_risk(data, col, target='is_cheating'):
    """Plots the % of Cheaters within each category"""
    # Calculate Risk
    risk = data.groupby(col)[target].mean().reset_index()
    counts = data[col].value_counts().reset_index()
    counts.columns = [col, 'count']
    
    fig, ax1 = plt.subplots(figsize=(14, 5))
    
    # Bar plot for Counts
    sns.barplot(data=counts, x=col, y='count', color='lightgrey', alpha=0.5, ax=ax1)
    ax1.set_ylabel("Sample Count (Grey)")
    
    # Line plot for Risk
    ax2 = ax1.twinx()
    sns.pointplot(data=risk, x=col, y=target, color='red', ax=ax2, order = counts[col].tolist())
    ax2.set_ylabel("Cheater Probability (Red Line)")
    ax2.set_ylim(0, 1.0)
    
    plt.title(f"Cheating Risk per Category: {col}")
    plt.show()

print("\n--- 2. Categorical Risk Analysis ---")
for col in cat_cols:
        plot_categorical_risk(labeled_df, col)


test_df['prediction'] = np.where(
    test_df['feature_007'].isna(),
    1,
    0
)

submission = pd.DataFrame({
    'user_hash': test_df['user_hash'],
    'prediction': np.clip(test_df['prediction'], 0, 1)
})

submission.to_csv("submission.csv", index=False)
print("Submission saved: submission.csv")

