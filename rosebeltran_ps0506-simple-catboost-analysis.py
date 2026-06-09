import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


train.head()


train.info()


test.info()


original.info()


# Add 'source' label
#original['Source'] = 'real'
#train['Source'] = 'synth'

# Drop the 'id' column
X = train.drop(columns=['id', 'Fertilizer Name'])

# Extract the target column
y = train['Fertilizer Name']

X.head()


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Output labels are now numbers
y_encoded[:10]


# Just check
X.info()


"""
# Prepare multiple copies of original dataset
orig_copy = original.copy()

# Number of copies
n = 6
for i in range(n):
    original = pd.concat([original, orig_copy], axis=0, ignore_index=True)
    
original.info()
"""


# Correct misspelled words
rename_dict = {
    'Temparature': 'Temperature',
    'Phosphorous': 'Phosphorus'
}

def rename_columns(df, rename_dict):
    return df.rename(columns=rename_dict)

X = rename_columns(X, rename_dict)
original = rename_columns(original, rename_dict)
test = rename_columns(test, rename_dict)


# Create new relative magnitude bins for each numeric column
def quantile_bin_encode(df, cols, q=5, labels=['very low', 'low', 'medium', 'high', 'very high']):
    df_transformed = df.copy()
    
    for col in cols:
        # Step 1: Bin into quantiles
        binned = pd.qcut(df_transformed[col], q=q, labels=labels)
        
        # Step 2: Map labels to ordinal integers
        label_map = {label: idx for idx, label in enumerate(labels)}
        df_transformed[f"{col}_bin"] = binned.map(label_map).astype('int64')
        
    return df_transformed


numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X_binned = quantile_bin_encode(X, numerical_cols)

X_binned.head()


original_binned = quantile_bin_encode(original, numerical_cols)

original_binned.head()


test_binned = quantile_bin_encode(test, numerical_cols)

test_binned.head()


X_binned = pd.concat([X_binned, original_binned], axis=0, ignore_index=True)
X_binned.drop(columns='Fertilizer Name', inplace=True)

X_binned.info()


y_encoded = np.concatenate([y_encoded, le.transform(original_binned['Fertilizer Name'])])

y_encoded.shape


# Store scores
f1_scores = []
map3_scores = []
models = []

# Collect predictions and true labels across all folds
all_y_true = []
all_y_pred = []

# Prepare K-Fold
skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_binned, y_encoded)):
    print(f"\n***** Fold {fold + 1} *****")
    
    # Make full copies to avoid warnings
    X_train = X_binned.iloc[train_idx].copy()
    X_val = X_binned.iloc[val_idx].copy()
    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    # Combine original with train data 
    #X_train = pd.concat([X_train, original_binned], ignore_index=True)
    #y_train = np.concatenate([y_train, le.transform(original_binned['Fertilizer Name'])])

    # Drop target column from training data
    #X_train.drop(columns=['Fertilizer Name'], inplace=True)

    #cols_to_category = list(X_train.select_dtypes(include=['object']).columns) + ['A', 'B', 'C']
    
    # Convert all selected features to categorical 
    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        
    for col in X_val.select_dtypes(include='object').columns:
        X_val[col] = X_val[col].astype('category')
    
    cat_features = X_train.select_dtypes(include='category').columns.tolist()  

    # For debugging purposes
    # print(cat_features)
    # print(X_train.info())
    # print(X_val.info())
    
    model = CatBoostClassifier(
        iterations=20000,
        depth=6,
        learning_rate=0.03,
        early_stopping_rounds=100,
        task_type="GPU",
        loss_function="MultiClass",
        eval_metric="MultiClass",
        l2_leaf_reg=0.15,
        bootstrap_type="Bayesian",
        use_best_model=True,
        bagging_temperature=0.25,
        random_strength=0.5,
        random_state=42,
        border_count=124,
        verbose=1000
    )

    model.fit(X_train, 
              y_train, 
              cat_features=cat_features,
              eval_set=(X_val, y_val))
    
    # Predict class labels and probabilities
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)

    # Store predictions and true labels
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    # F1 Score
    report = classification_report(y_val, y_pred, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    f1_scores.append(f1_macro)
    
    # MAP@3
    top3_preds = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]
    
    def mapk(actual, predicted, k=3):
        def apk(a, p, k):
            if a in p[:k]:
                return 1.0 / (p[:k].index(a) + 1)
            return 0.0
        return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

    map3 = mapk(y_val.tolist(), top3_preds.tolist(), k=3)
    map3_scores.append(map3)
    models.append(model)

    print(f"F1 (macro): {f1_macro:.4f} | MAP@3: {map3:.4f}")

# Final Results
print("\n***** Final CV Results *****")
print(f"Avg F1: {np.mean(f1_scores):.4f}")
print(f"Avg MAP@3: {np.mean(map3_scores):.4f}")



# Step 1: Confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Step 2: Make it pretty
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Greens", 
            xticklabels=le.classes_, yticklabels=le.classes_,
           )

plt.title("Normalized Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred, digits=4))


# Initialize accumulator
importances_total = np.zeros(len(models[0].feature_names_))
feature_names = models[0].feature_names_

# Accumulate importance per model
for model in models:
    importances_total += model.get_feature_importance(type='PredictionValuesChange')

# Average
importances_avg = importances_total / len(models)

# Make a dataframe
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_avg
}).sort_values(by='importance', ascending=False)


# Limit to top N features for readability, default to all columns
top_n = len(X_train.columns)
top_features = importances_df.head(top_n)

# Create a green color palette
green_palette = sns.color_palette("Greens", as_cmap=False, n_colors=len(top_features))

# Plot using barplot
plt.figure(figsize=(8, 5))

sns.barplot(
    data=top_features,
    y='feature',
    x='importance',
    palette=green_palette
)

plt.title("Top Feature Importances (CatBoost)")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()






def top_n_coverage(y_true, y_proba, n=3):
    """Returns the proportion of times the true label is in the top-N predicted labels."""
    top_n_preds = np.argsort(y_proba, axis=1)[:, -n:]  # Get top N indices (classes)
    
    # Check if the true label is in the top N predictions
    hits = [y_true[i] in top_n_preds[i] for i in range(len(y_true))]
    
    return np.mean(hits)

for n in range(1, 8):
    coverage = top_n_coverage(y_val, y_probs, n)
    print(f"Top-{n} Coverage: {coverage:.4f}")



# Setup: Top-N values
top_ns = list(range(1, y_probs.shape[1] + 1))
coverages = [top_n_coverage(y_val, y_probs, n) for n in top_ns]

# Apply Seaborn theme
sns.set(style="whitegrid")

# Create figure
plt.figure(figsize=(8, 5))

# Plot with seaborn line aesthetics
sns.lineplot(x=top_ns, y=coverages, marker='o', color='green', linewidth=2)

# Decorations
plt.title("Top-N Coverage Curve", fontsize=16)
plt.xlabel("N (Top-N Predictions)", fontsize=12)
plt.ylabel("Coverage", fontsize=12)
plt.ylim(0, 1.05)
plt.xticks(top_ns)
plt.yticks([i/10 for i in range(11)])
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Top-3 Threshold')
plt.legend()

plt.tight_layout()
plt.show()



# Convert test data
for col in cat_features:
    test_binned[col] = test_binned[col].astype('category')

# Accumulate prediction probabilities
all_preds = np.zeros((test_binned.shape[0], len(le.classes_)))

X_test = test_binned.drop(columns='id')

for model in models:
    probs = model.predict_proba(X_test)
    all_preds += probs

# Average over folds
avg_preds = all_preds / len(models)

# Get top 3 indices like before
top3_preds = np.argsort(avg_preds, axis=1)[:, -3:][:, ::-1]  # Top 3 class indices, descending order

# Convert class indices back to original label strings
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

submission = pd.DataFrame({
    'id': test['id'],  # Replace with actual ID column name
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission.to_csv('submission.csv', index=False)
print("Done!")


submission.head()

