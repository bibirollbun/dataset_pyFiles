ls -l /kaggle/input/27-june-2025-fertilizer 



import pandas as pd
from collections import Counter
from functools import reduce

# Define file paths
model_files = {
    'M1': 'submission__LB__0_38_285.csv',
    'M2': 'submission__LB__0_38_213.csv',
    'M3': 'submission__LB__0_38_192.csv',
    'M4': 'submission__LB__0_38_000.csv',
    'M5': 'submission__LB__0_37_971.csv',
    'G1': 'submission__LB__0_GEN_27.4a.csv',
    'G2': 'submission__LB__0_GEN_27.4b.csv',
    'G3': 'submission__LB__0_GEN_27.4c.csv'
}

path = '/kaggle/input/27-june-2025-fertilizer/'

# Load predictions and rename columns
dfs = []
for name, file in model_files.items():
    df = pd.read_csv(path + file)
    df = df.rename(columns={'Fertilizer Name': name})
    dfs.append(df)

# Merge all predictions on 'id'
df_merged = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)

# Define ensembling strategies
def get_top3(row, method='vote', weights=None):
    preds = [row[col] for col in model_files]
    if method == 'vote':
        return ' '.join([x for x, _ in Counter(preds).most_common(3)])
    elif method == 'weighted':
        score_map = {}
        for name in model_files:
            label = row[name]
            score_map[label] = score_map.get(label, 0) + weights.get(name, 1.0)
        sorted_preds = sorted(score_map.items(), key=lambda x: -x[1])
        return ' '.join([label for label, _ in sorted_preds[:3]])
    elif method == 'consensus_fill':
        counts = Counter(preds)
        common = [x for x, c in counts.items() if c > 1]
        rest = [x for x in preds if x not in common]
        return ' '.join(dict.fromkeys(common + rest))[:3]
    elif method == 'stacked':
        return ' '.join(dict.fromkeys(preds))[:3]
    elif method == 'diverse':
        return ' '.join(dict.fromkeys(preds[::-1]))[:3]
    else:
        raise ValueError("Unknown method")

# Define ensemble weights
weights = {
    'M1': 0.92, 'M2': 0.90, 'M3': 0.89, 'M4': 0.88, 'M5': 0.87,
    'G1': 0.80, 'G2': 0.78, 'G3': 0.76
}

# Create ensemble submissions
methods = ['vote', 'weighted', 'consensus_fill', 'stacked', 'diverse']
for method in methods:
    df_merged['Fertilizer Name'] = df_merged.apply(get_top3, axis=1, method=method, weights=weights)
    df_merged[['id', 'Fertilizer Name']].to_csv(f'ensemble_{method}.csv', index=False)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Load all ensemble outputs
methods = ['vote', 'weighted', 'consensus_fill', 'stacked', 'diverse']
dfs = {m: pd.read_csv(f'ensemble_{m}.csv').rename(columns={'Fertilizer Name': m}) for m in methods}

# Merge into one DataFrame
merged = dfs[methods[0]]
for m in methods[1:]:
    merged = merged.merge(dfs[m], on='id')

# 1. Correlation heatmap (convert predictions to categorical codes)
cat_df = merged.copy()
for m in methods:
    cat_df[m] = cat_df[m].astype('category').cat.codes

plt.figure(figsize=(8, 6))
sns.heatmap(cat_df[methods].corr(), annot=True, cmap='vlag')
plt.title('Correlation Between Ensemble Strategies')
plt.show()

# 2. Distribution of number of unique predictions per row
merged['unique_preds'] = merged[methods].apply(lambda row: len(set(row)), axis=1)

plt.figure(figsize=(8, 4))
sns.histplot(merged['unique_preds'], bins=range(1, 7), discrete=True)
plt.title('Unique Predictions per Row Across Ensembles')
plt.xlabel('Number of Unique Predictions')
plt.ylabel('Count')
plt.show()

# 3. Most common predictions
all_preds = pd.concat([merged[m] for m in methods])
top_preds = all_preds.value_counts().head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_preds.values, y=top_preds.index)
plt.title('Top 20 Most Common Fertilizer Predictions')
plt.xlabel('Count')
plt.ylabel('Fertilizer Name')
plt.show()



from scipy.stats import entropy
from collections import Counter

# Function to compute entropy of model predictions per row
def row_entropy(row):
    counts = Counter(row)
    probs = [v / len(row) for v in counts.values()]
    return entropy(probs)

# Apply to merged ensemble predictions
merged['row_entropy'] = merged[methods].apply(row_entropy, axis=1)

# Plot the distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.histplot(merged['row_entropy'], bins=30)
plt.title('Row-level Prediction Entropy Across Ensembles')
plt.xlabel('Entropy')
plt.ylabel('Number of Rows')
plt.show()




threshold = 0.8  # based on your histogram

# Choose based on entropy
merged['Fertilizer Name'] = merged.apply(
    lambda row: row['weighted'] if row['row_entropy'] < threshold else row['38_285'],
    axis=1
)

# Save to file
merged[['id', 'Fertilizer Name']].to_csv('ensemble_entropy_split.csv', index=False)


