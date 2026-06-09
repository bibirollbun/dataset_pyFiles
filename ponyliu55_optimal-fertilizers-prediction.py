import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer_Prediction.csv")


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

# Just check
X.info()


# Output labels are now numbers
y_encoded[:10]


# Prepare multiple copies of original dataset
orig_copy = original.copy()

# Number of copies
n = 6
for i in range(n):
    original = pd.concat([original, orig_copy], axis=0, ignore_index=True)
    
original.info()


# Store scores
f1_scores = []
map3_scores = []
models = []

# Collect predictions and true labels across all folds
all_y_true = []
all_y_pred = []

# Prepare K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n***** Fold {fold + 1} *****")

    # Make full copies to avoid warnings
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    # Combine original with train data 
    X_train = pd.concat([X_train, original], ignore_index=True)
    y_train = np.concatenate([y_train, le.transform(original['Fertilizer Name'])])

    # Drop target column from training data
    X_train.drop(columns=['Fertilizer Name'], inplace=True)

    # Convert all features to categorical (except target, which is already separated)
    for col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        
    for col in X_val.columns:
        X_val[col] = X_val[col].astype('category')
    
    cat_features = X_train.columns.tolist()   # capture all input columns

    # For debugging purposes
    # print(cat_features)
    # print(X_train.info())
    # print(X_val.info())

    model = XGBClassifier(
                max_depth=12,
                colsample_bytree=0.467,
                subsample=0.86,
                n_estimators=4000,
                learning_rate=0.03,
                gamma=0.26,
                max_delta_step=4,
                reg_alpha=2.7,
                reg_lambda=1.4,
                objective='multi:softprob',
                random_state=13,
                enable_categorical=True,
                tree_method='hist',     
                device='cuda' 
            )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,
        verbose=200,
    )
    
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
importances_total = np.zeros(len(cat_features))
feature_names = cat_features

# Accumulate importance per model
for model in models:
    importances_total += model.feature_importances_

# Average
importances_avg = importances_total / len(models)

# Make a dataframe
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_avg
}).sort_values(by='importance', ascending=False)


# Limit to top N features for readability
top_n = 8
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

plt.title("Top Feature Importances")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# y_true should be integer labels (encoded)
# y_proba is shape (n_samples, n_classes)
n_classes = y_probs.shape[1]

# Initialize matrix
soft_cm = np.zeros((n_classes, n_classes))

# Fill it
for i in range(len(y_val)):
    true_class = y_val[i]
    soft_cm[true_class] += y_probs[i]  # adds vector of predicted probs

# Normalize each row (optional)
soft_cm_normalized = soft_cm / soft_cm.sum(axis=1, keepdims=True)

# Plot it!
plt.figure(figsize=(8, 6))
sns.heatmap(soft_cm_normalized, annot=True, fmt=".3f", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Soft Confusion Matrix (Probabilities)", fontsize=16)
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
sns.set_palette("Greens")

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
    test[col] = test[col].astype('category')

# Accumulate prediction probabilities
all_preds = np.zeros((test.shape[0], len(le.classes_)))

X_test = test.drop(columns='id')
cat_features = X_test.columns.tolist()   # capture all input columns

for model in models:
    probs = model.predict_proba(X_test)
    all_preds += probs

# Average over folds
avg_preds = all_preds / len(models)

# Get top 3 indices like before
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Top 3 class indices, descending order

# Convert class indices back to original label strings
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

submission = pd.DataFrame({
    'id': test['id'],  # Replace with actual ID column name
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission.to_csv('submission.csv', index=False)
print("Done!")


submission.head()

