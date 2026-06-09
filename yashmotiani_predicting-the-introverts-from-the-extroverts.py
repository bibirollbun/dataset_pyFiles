print(y_train.value_counts(normalize=True))



# ğŸ“¦ Import libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils import resample

# ğŸ“‚ Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# âš–ï¸� Downsample majority class (Extrovert)
df_majority = df_train[df_train.Personality == 'Extrovert']
df_minority = df_train[df_train.Personality == 'Introvert']
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# ğŸ§¹ Prepare features
X_train = df_balanced.drop(['id', 'Personality'], axis=1).fillna("Missing")
y_train = df_balanced['Personality']
X_test = df_test.drop(['id'], axis=1).fillna("Missing")

# ğŸ§  Feature engineering
for col in X_train.columns:
    if X_train[col].nunique() < 20:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

X_train['yes_count'] = (X_train == 'Yes').sum(axis=1)
X_train['no_count'] = (X_train == 'No').sum(axis=1)
X_train['missing_count'] = (X_train == 'Missing').sum(axis=1)

X_test['yes_count'] = (X_test == 'Yes').sum(axis=1)
X_test['no_count'] = (X_test == 'No').sum(axis=1)
X_test['missing_count'] = (X_test == 'Missing').sum(axis=1)

categorical_features = X_train.columns[X_train.dtypes == 'object'].tolist()

# ğŸ“Š Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# ğŸš€ Train model with class weights (force Introvert learning)
model = CatBoostClassifier(
    iterations=700,
    depth=4,
    learning_rate=0.05,
    cat_features=categorical_features,
    class_weights=[1, 1],
    verbose=100,
    random_seed=42
)
model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

# ğŸ“ˆ Evaluate
y_pred = model.predict(X_val)
print("ğŸ§ª Validation Results")
print("F1 Score:", f1_score(y_val, y_pred, average='weighted'))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

# ğŸ‘�ï¸� Check predicted label distribution
print("Prediction distribution:", pd.Series(y_pred).value_counts())

# ğŸ“� Predict on test set & save
test_preds = model.predict(X_test)
submission['Personality'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission file created successfully!")





