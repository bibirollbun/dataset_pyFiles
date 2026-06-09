# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for clean output
import warnings
warnings.filterwarnings("ignore")


# Load CSV files
train = pd.read_csv('/kaggle/input/vector-borne-disease-classification-challenge/train.csv')
test = pd.read_csv('/kaggle/input/vector-borne-disease-classification-challenge/test.csv')


# First few rows of training data
print("Train Set:")
display(train.head())

# First few rows of test data
print("Test Set:")
display(test.head())


# ğŸ§® Shape (Rows Ã— Columns)

# Show the shape of both datasets
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# ğŸ§  Dataset Info

# Get metadata of train set
print("Train info:")
train.info()

# Get metadata of test set
print("\nTest info:")
test.info()


# Unique value count for each column in the train set
print("Unique values per column in train set:")
print(train.nunique())


# ğŸ“Œ Target Column: prognosis
# Value counts for target variable
print("\nDistribution of target (prognosis):")
print(train['prognosis'].value_counts())

# Plot it
plt.figure(figsize=(10, 6))
train['prognosis'].value_counts().plot(kind='barh')
plt.title('Distribution of Prognosis (Target Classes)')
plt.xlabel('Number of Patients')
plt.ylabel('Disease')
plt.show()


# Summary of numerical features
print("\nDescriptive stats:")
display(train.describe())


# Missing values count
print("\nMissing values in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\nMissing values in test:")
print(test.isnull().sum()[test.isnull().sum() > 0])


# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=train, y='prognosis', order=train['prognosis'].value_counts().index)
plt.title('Distribution of Prognosis (Target Classes)')
plt.xlabel('Number of Patients')
plt.ylabel('Disease')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()
train['label'] = le.fit_transform(train['prognosis'])

# Save mapping for decoding predictions later
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Preview
print("Encoded classes:")
print(label_mapping)



# Drop ID and target text columns
X = train.drop(columns=['id', 'prognosis', 'label'])  # Features
y = train['label']                                    # Target

# For test set, drop only 'id'
X_test = test.drop(columns=['id'])



from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.ensemble import RandomForestClassifier

# Initialize model
rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    random_state=42, 
    class_weight='balanced',  # helps with imbalance
    n_jobs=-1
)

# Fit model
rf.fit(X_train, y_train)


# Predict probabilities on validation and test
val_probs = rf.predict_proba(X_val)
test_probs = rf.predict_proba(X_test)

# Convert probability outputs to top 3 class indices
val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
test_top3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Convert each row's top-3 label indices back to label names
val_top3_labels = np.array([[le.inverse_transform([idx])[0] for idx in row] for row in val_top3])
test_top3_labels = np.array([[le.inverse_transform([idx])[0] for idx in row] for row in test_top3])

# Example: first 5 predictions
print("Sample predictions (validation):")
for i in range(5):
    print(val_top3_labels[i])


def mapk(true, pred, k=3):
    score = 0.0
    for t, p in zip(true, pred):
        if t in p:
            score += 1.0 / (p.tolist().index(t) + 1)
    return score / len(true)

# Local MAP@3
map3_score = mapk(y_val.values, val_top3)
print(f"Validation MAP@3 Score: {map3_score:.4f}")


import lightgbm as lgb
from sklearn.metrics import log_loss


# LightGBM expects labels as integers â€” already done
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)


from lightgbm import LGBMClassifier

# Initialize LGBMClassifier
lgb_clf = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=-1,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    class_weight='balanced'  # Helps with imbalance
)

from lightgbm import early_stopping, log_evaluation

# Fit with callbacks instead of early_stopping_rounds directly
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)


# Predict probabilities using the fitted LGBMClassifier
val_probs_lgb = lgb_clf.predict_proba(X_val)
test_probs_lgb = lgb_clf.predict_proba(X_test)

# Get top 3 predicted class indices
val_top3_lgb = np.argsort(val_probs_lgb, axis=1)[:, -3:][:, ::-1]
test_top3_lgb = np.argsort(test_probs_lgb, axis=1)[:, -3:][:, ::-1]

# Convert indices to class labels
val_top3_labels_lgb = np.array([[le.inverse_transform([i])[0] for i in row] for row in val_top3_lgb])
test_top3_labels_lgb = np.array([[le.inverse_transform([i])[0] for i in row] for row in test_top3_lgb])


val_map3_lgb = mapk(y_val.values, val_top3_lgb)
print(f"Validation MAP@3 Score (LightGBM): {val_map3_lgb:.4f}")


import matplotlib.pyplot as plt

# Plot top 20 important features
lgb.plot_importance(lgb_clf, max_num_features=20, importance_type='gain', figsize=(10, 6))
plt.title("Top 20 Important Features (by Gain)")
plt.tight_layout()
plt.show()


# Keep Only Top 40 Most Important Features

# Get feature importances (by gain)
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_clf.feature_importances_
}).sort_values(by='importance', ascending=False)

# Top 40 features
top_features = feature_importances.head(40)['feature'].tolist()

print("Top 40 selected features:")
print(top_features)




# Filter to top 40 features
X_train_top = X_train[top_features]
X_val_top = X_val[top_features]
X_test_top = X_test[top_features]


# Retrain LightGBM on top 40 features
lgb_clf_pruned = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=-1,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    class_weight='balanced'
)

lgb_clf_pruned.fit(
    X_train_top, y_train,
    eval_set=[(X_val_top, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)


# Predict
val_probs_pruned = lgb_clf_pruned.predict_proba(X_val_top)
val_top3_pruned = np.argsort(val_probs_pruned, axis=1)[:, -3:][:, ::-1]

# Evaluate
val_map3_pruned = mapk(y_val.values, val_top3_pruned)
print(f"Validation MAP@3 Score (LGBM + Top 40 Features): {val_map3_pruned:.4f}")


# Rebuild full train/test with top 40 features
X_full = X[top_features]
X_test_final = X_test[top_features]
y_full = y

best_iter = lgb_clf_pruned.best_iteration_

from lightgbm import LGBMClassifier

# Final model trained on all data with top 40 features
final_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    learning_rate=0.05,
    n_estimators=best_iter,
    max_depth=-1,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    class_weight='balanced'
)

# Fit on full data
final_lgb.fit(X_full, y_full)


# Predict class probabilities
test_probs_final = final_lgb.predict_proba(X_test_final)
test_top3_final = np.argsort(test_probs_final, axis=1)[:, -3:][:, ::-1]

# Decode to labels
test_top3_labels_final = np.array([[le.inverse_transform([i])[0] for i in row] for row in test_top3_final])



submission_final = pd.DataFrame({
    'id': test['id'],
    'prognosis': [' '.join(row) for row in test_top3_labels_final]
})

submission_final.to_csv('submission.csv', index=False)
submission_final.head()







