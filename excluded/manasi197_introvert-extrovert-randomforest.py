!pip install scikit-learn imblearn optuna


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import optuna
import cudf  # For GPU-accelerated data loading
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# --- 2. Data Loading ---
train_df = cudf.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = cudf.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train_df = train_df.to_pandas()
test_df = test_df.to_pandas()


# --- 3. Exploratory Data Analysis ---
print("Class Distribution:")
print(train_df['Personality'].value_counts(normalize=True))

# Check unique values in boolean columns
boolean_cols = ['Stage_fear', 'Drained_after_socializing']
for col in boolean_cols:
    print(f"Unique values in {col} (train):", train_df[col].unique())
    print(f"Unique values in {col} (test):", test_df[col].unique())


# --- 4. Data Preparation ---
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Clip outliers in numerical columns
for col in numerical_cols:
    q1, q3 = train_df[col].quantile([0.01, 0.99])
    train_df[col] = train_df[col].clip(q1, q3)
    test_df[col] = test_df[col].clip(q1, q3)

# Handle missing values for numerical columns
for col in numerical_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)

# Encode boolean columns
for col in boolean_cols:
    train_df[col] = train_df[col].replace({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0})
    test_df[col] = test_df[col].replace({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0})
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# Encode target variable
train_df['Personality'] = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})


# Define features and target
features = numerical_cols + boolean_cols
X = train_df[features]
y = train_df['Personality']
X_test = test_df[features]

# Standardize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])



# Apply Random Oversampling
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print("Class Distribution after Random Oversampling:")
print(pd.Series(y).value_counts(normalize=True))


# --- 5. Hyperparameter Tuning with Optuna ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best Random Forest Parameters:", best_params)


# --- 6. Model Building and Training ---
rf_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)


# K-fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
rf_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train Random Forest
    rf_model.fit(X_train_fold, y_train_fold)
    rf_val_pred = rf_model.predict(X_val_fold)
    print(f"Random Forest Fold {fold+1} Accuracy: {accuracy_score(y_val_fold, rf_val_pred):.4f}")
    
    # Predict on test set
    rf_preds += rf_model.predict_proba(X_test)[:, 1] / n_splits


# Predict with adjusted threshold
threshold = 0.4  # Lower threshold to favor minority class
final_preds = (rf_preds > threshold).astype(int)


# --- 7. Submission Creation ---
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_preds
})
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(submission.head())




