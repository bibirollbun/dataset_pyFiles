import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score


# -----------------------
# Step 1: Load Data
# -----------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



test_id = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


# -----------------------
# Step 2: Encode Categorical Features
# -----------------------
X = train.drop("Fertilizer Name", axis=1)
y = train["Fertilizer Name"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])


X.info()



X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -----------------------
# Step 4: Train RF Classifier
# -----------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)


# -----------------------
# Step 5: Evaluate with MAP@3
# -----------------------
# Predict probabilities for validation set
val_probs = rf_model.predict_proba(X_valid)

# Get true labels and predicted class scores (probabilities)
true_labels = np.zeros_like(val_probs)
for i, label in enumerate(y_valid):
    true_labels[i, label] = 1

# Compute MAP@3
def mapk(true, preds, k=3):
    topk = np.argsort(preds, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for i in range(true.shape[0]):
        actual = np.where(true[i] == 1)[0][0]
        predicted = topk[i]
        if actual in predicted:
            score += 1.0 / (np.where(predicted == actual)[0][0] + 1)
    return score / true.shape[0]

map3_score = mapk(true_labels, val_probs, k=3)
print(f"Validation MAP@3 Score: {map3_score:.5f}")



# -----------------------
# Step 6: Predict on Test Data
# -----------------------
test_probs = rf_model.predict_proba(test)
top3_indices = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top3_labels = label_encoder.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
preds_top3 = [' '.join(row) for row in top3_labels]


submission = pd.DataFrame({
    "id": test_id,
    "Fertilizer Name": preds_top3
})
submission.to_csv("submission.csv", index=False)
submission.head(5)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

top3_labels = label_encoder.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)

# Create DataFrame with each rank
top3_df = pd.DataFrame(top3_labels, columns=['Rank1', 'Rank2', 'Rank3'])

# Melt to long format: Fertilizer + Rank
top3_long = top3_df.melt(var_name='Rank', value_name='Fertilizer Name')

# Count occurrences grouped by Fertilizer and Rank
top3_counts = top3_long.groupby(['Fertilizer Name', 'Rank']).size().reset_index(name='Count')

# Pivot for stacked bar chart
top3_pivot = top3_counts.pivot(index='Fertilizer Name', columns='Rank', values='Count').fillna(0)

# Sort by total count
top3_pivot['Total'] = top3_pivot.sum(axis=1)
top3_pivot = top3_pivot.sort_values('Total', ascending=False).drop(columns='Total')

# Plot stacked bar
top3_pivot.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='viridis')
plt.title("Top-3 Predicted Fertilizer Frequencies by Rank")
plt.ylabel("Count")
plt.xlabel("Fertilizer Name")
plt.xticks(rotation=45)
plt.legend(title="Prediction Rank")
plt.tight_layout()
plt.show()


