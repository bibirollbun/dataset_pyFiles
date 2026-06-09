import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- Load Data ---
# Assumes data is in the default /kaggle/input/ directory
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError:
    print("Trying local file paths...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')


# --- Feature Engineering ---
def feature_engineer(df):
    """Creates new features to improve model performance."""
    df['Social_Balance'] = df['Social_event_attendance'] / (df['Going_outside'] + 1)
    df['Online_Offline_Ratio'] = df['Post_frequency'] / (df['Social_event_attendance'] + 1)
    df['Alone_x_CircleSize'] = df['Time_spent_Alone'] * df['Friends_circle_size']
    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)


# --- Preprocessing ---
le = LabelEncoder()
train_df['Personality'] = le.fit_transform(train_df['Personality']) # Extrovert: 0, Introvert: 1

y = train_df['Personality']
X = train_df.drop(['id', 'Personality'], axis=1)
X_test = test_df.drop('id', axis=1)

X_test = X_test[X.columns]

categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_features] = imputer_cat.fit_transform(X[categorical_features])
X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])

for col in categorical_features:
    X[col] = X[col].map({'Yes': 1, 'No': 0})
    X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

imputer_num = IterativeImputer(max_iter=10, random_state=42)
X[numerical_features] = imputer_num.fit_transform(X[numerical_features])
X_test[numerical_features] = imputer_num.transform(X_test[numerical_features])


# --- Model Training with Cross-Validation ---
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

# Best performing GradientBoostingClassifier parameters
gb_params = {
    'n_estimators': 400,
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.7,
    'random_state': 42,
    'max_features': 0.8
}

print("Starting model training with best hyperparameters...")
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = GradientBoostingClassifier(**gb_params)
    model.fit(X_train, y_train)

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    sub_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
    print(f"Fold {n_fold+1} completed.")

oof_accuracy = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f"\nFinal Cross-validation Accuracy: {oof_accuracy:.5f}")


# --- Create Submission File ---
test_predictions_encoded = (sub_preds > 0.5).astype(int)
test_predictions = le.inverse_transform(test_predictions_encoded)

submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nFinal submission file 'submission.csv' created successfully!")
print(submission_df.head())

