import pandas as pd
from datetime import datetime

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# test data
print(test.shape)
test


# train data
print(train.shape)
train


# Separate numerical and categorical columns

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = train.select_dtypes(include=["object"]).columns

print("Numerical columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))



# Distributions of numerical columns

import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train[col].hist(bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



# Detect & Replace Outliers (Z-score method)

import numpy as np
from scipy import stats

# Copy dataset to avoid modifying original
train_clean = train.copy()

# Exclude 'id' column (not a feature)
num_cols = [c for c in num_cols if c != "id"]

# Set Z-score threshold
threshold = 3

for col in num_cols:
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(train_clean[col]))

    # Print how many outliers detected
    print(f"{col}: {np.sum(z_scores > threshold)} outliers detected")

    # Compute boundaries (mean Â± 3*std)
    mean, std = train_clean[col].mean(), train_clean[col].std()
    upper, lower = mean + threshold * std, mean - threshold * std

    # Replace values outside boundaries with caps
    train_clean[col] = np.where(
        train_clean[col] > upper, upper,
        np.where(train_clean[col] < lower, lower, train_clean[col])
    )

print("\nâœ… Outliers replaced with boundary values (capped).")



# Plot Distributions After Outlier Handling
import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train_clean[col].hist(bins=50)
    plt.title(f"Distribution of {col} (after outlier handling)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



train


# Categorical value counts

for col in cat_cols:
    plt.figure(figsize=(6,4))
    train[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()



# category info
for col in cat_cols:
    print(f"\n---- {col} ----")
    print(train[col].value_counts())



# Mixed Binary + One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Columns that are strictly binary â†’ map directly
binary_cols = ["default", "housing", "loan"]

# for col in binary_cols:
#     train[col] = train[col].map({"no": 0, "yes": 1})
for col in binary_cols:
    train[col] = (
        train[col]
        .astype(str)          # convert to string
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # make lowercase
        .map({"no": 0, "yes": 1})
    )


# Columns that need one-hot encoding
onehot_cols = ["job", "marital", "education", "contact", "month", "poutcome"]

# Apply One-Hot
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(train[onehot_cols])

print("One-hot encoded categorical shape:", X_cat.shape)



print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())

print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())
print(train[binary_cols].head(10))



# Numerical + Target
from sklearn.preprocessing import StandardScaler
import numpy as np

# Exclude id + target
num_features = [c for c in num_cols if c not in ["id", "y"]]

# Scale numerical features
scaler = StandardScaler()
X_num = scaler.fit_transform(train[num_features])

# Combine numerical + categorical
X = np.hstack([X_num, X_cat, train[binary_cols].values])

# Target variable
y = train["y"].values

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)



# Final updated dataframe for training


# Get names for one-hot encoded columns
onehot_feature_names = encoder.get_feature_names_out(onehot_cols)

# Combine all features into a DataFrame
train_updated = pd.DataFrame(
    np.hstack([X_num, X_cat, train[binary_cols].values]),
    columns=list(num_features) + list(onehot_feature_names) + binary_cols
)

# Add target column
train_updated["y"] = y

print("Final dataframe shape:", train_updated.shape)
train_updated.head()



# split + class weights

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_val.shape)

# class weights to help with imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
class_weight_dict



import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Convert class_weight dict to sample weights
sample_weights = np.array([class_weight_dict[cls] for cls in y])

# Stratified K-Fold
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage
models = []
oof_preds = np.zeros(len(X))   # out-of-fold predictions
test_probs = np.zeros((test.shape[0], n_splits))

# Prepare test set in the same way
# Binary cols
for col in binary_cols:
    test[col] = (
        test[col]
        .astype(str).str.strip().str.lower()
        .map({"no": 0, "yes": 1})
    )

# One-hot (use trained encoder!)
X_cat_test = encoder.transform(test[onehot_cols])

# Scale numeric (use trained scaler!)
X_num_test = scaler.transform(test[num_features])

# Final test features
X_test_final = np.hstack([X_num_test, X_cat_test, test[binary_cols].values])




# LightGBM CV Loop using optimal params

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nðŸ”¹ Training fold {fold+1}/{n_splits} >>>")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]
    
    w_train, w_val   = sample_weights[train_idx], sample_weights[val_idx]

   
    model = lgb.LGBMClassifier(
        n_estimators=38000, #40000
        class_weight='balanced',   # keep balanced handling
        learning_rate=0.0675, #0.065
        num_leaves=128,
        max_depth=12,
        min_child_samples=12,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=0.3,
        max_bin=5000,
        random_state=2003,
        boosting_type='gbdt',
        metric='auc',
        verbosity=-1
        # device='gpu' # enable GPU
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(300),
            lgb.log_evaluation(500)
        ]
    )
    
    # Store model
    models.append(model)
    
    # OOF predictions
    oof_preds[val_idx] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    
    # Test predictions
    test_probs[:, fold] = model.predict_proba(X_test_final, num_iteration=model.best_iteration_)[:, 1]




# Validation AUC

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nâœ… CV AUC: {oof_auc:.5f}")

# Average test predictions
test_pred_final = test_probs.mean(axis=1)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ROC Curves

plt.figure(figsize=(8,6))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    fpr, tpr, _ = roc_curve(y[val_idx], oof_preds[val_idx])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, label=f"Fold {fold+1} (AUC = {roc_auc:.4f})")

# Mean ROC
fpr, tpr, _ = roc_curve(y, oof_preds)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="black", lw=2, linestyle="--",
         label=f"Overall (AUC = {roc_auc:.4f})")

plt.plot([0,1],[0,1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (LightGBM CV)")
plt.legend(loc="lower right")
plt.show()


# Feature Importance (Gain)

importances = np.zeros(X.shape[1])
for model in models:
    importances += model.booster_.feature_importance(importance_type="gain")

importances /= len(models)

# Build feature names
feature_names = list(num_features) + list(encoder.get_feature_names_out(onehot_cols)) + binary_cols

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10,8))
plt.barh(fi_df["feature"][:20][::-1], fi_df["importance"][:20][::-1])
plt.title("Top 20 Feature Importances (Gain)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# Prediction Probability Distribution
plt.figure(figsize=(8,6))
plt.hist(oof_preds[y==0], bins=50, alpha=0.6, label="Class 0")
plt.hist(oof_preds[y==1], bins=50, alpha=0.6, label="Class 1")
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Submission File
submission = pd.DataFrame({
    "id": test["id"],
    "y": test_pred_final  # probabilities
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")
submission.head()

