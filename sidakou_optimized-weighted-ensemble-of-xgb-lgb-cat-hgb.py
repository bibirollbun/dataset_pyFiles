# ===============================
# ğŸ“š Library Imports
# ===============================

# --- Basic libraries ---
import os                  # File and directory operations
import numpy as np         # Numerical computations and array operations
import pandas as pd        # DataFrame manipulation

# --- Visualization ---
import seaborn as sns      # Statistical data visualization
import matplotlib.pyplot as plt  # Plotting library

# --- Preprocessing ---
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split ,KFold # Split dataset into train and validation sets

# --- Machine Learning Models ---
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

# --- Evaluation Metrics ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Regression metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)  # Classification metrics

# --- Hyperparameter Optimization ---
import optuna  # Automatic hyperparameter tuning


# --- Kaggle-specific: Display input file paths ---
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  # Print dataset file paths


# ğŸ“‚ Data Loading 
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
predict = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train.head()


# Plots a heatmap showing correlations between numerical features.
corr = train.select_dtypes(['number']).corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)


# Create new features from grade_subgrade and remove unnecessary columns
def create_features(df):
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
    
    return df

train = create_features(train)
predict = create_features(predict)


# One-Hot Encoding
def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=object_cols, drop_first=False)
    return df

train = one_hot_encode(train)
predict = one_hot_encode(predict)

missing_cols = set(train.columns) - set(predict.columns)
for col in missing_cols:
    predict[col] = 0

predict = predict[train.columns]
predict = predict.drop(columns=['loan_paid_back'])


# Convert boolean columns

def bool_to_int(df):
    bool_columns = df.select_dtypes(include='bool').columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    return df

train = bool_to_int(train)
predict = bool_to_int(predict)


# Add target mean encoding and count encoding features efficiently
# This version avoids DataFrame fragmentation by concatenating columns at once

def add_target_count_features(train, predict, target_col, n_splits=10):
    BASE = [c for c in train.columns if c not in [target_col]]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features = pd.DataFrame(index=train.index)
    count_features = pd.DataFrame(index=train.index)
    mean_features_pred = pd.DataFrame(index=predict.index)
    count_features_pred = pd.DataFrame(index=predict.index)

    for col in BASE:
        if train[col].isnull().all():
            continue

        # === Mean Encoding with K-Fold (leakage prevention) ===
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target_col].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features[f'mean_{col}'] = mean_encoded

        # Apply global mean mapping to prediction data
        global_mean = train.groupby(col)[target_col].mean()
        mean_features_pred[f'mean_{col}'] = predict[col].map(global_mean)

        # === Count Encoding ===
        count_map = train[col].value_counts().to_dict()
        count_features[f'count_{col}'] = train[col].map(count_map)
        count_features_pred[f'count_{col}'] = predict[col].map(count_map)

    # === Concatenate all features at once to avoid fragmentation ===
    train = pd.concat([train, mean_features, count_features], axis=1)
    predict = pd.concat([predict, mean_features_pred, count_features_pred], axis=1)

    # Defragment DataFrames for better performance
    train = train.copy()
    predict = predict.copy()

    print(f"{len(mean_features.columns) + len(count_features.columns)} features created!")
    return train, predict


train, predict = add_target_count_features(train, predict, target_col='loan_paid_back')


def split_data(df,test_size=0.2,random_state=42):
    X = df.drop(columns=['id','loan_paid_back'])
    y = df['loan_paid_back']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(train)

predict_X = predict.copy()
predict_X = predict_X.drop(columns=['id'])


# === Function to build an XGBoost model ===
def build_xgboost_model(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=37):

    # Create an XGBoost classifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",  # Binary classification (output = probability)
        n_estimators=n_estimators,    # Number of trees
        max_depth=max_depth,          # Depth of each tree (controls model complexity)
        learning_rate=learning_rate,  # Learning rate (smaller = slower but more stable learning)
        random_state=random_state,    # Set random seed for reproducibility
        eval_metric="auc"             # Evaluation metric = AUC (used during training)
    )
    return model  # Return the constructed model


# === Build and train the model ===
xgb_model = build_xgboost_model()     # Initialize the model with default parameters
xgb_model.fit(X_train, y_train)       # Train the model on training data (features and labels)


# === Function to build a CatBoost model ===
def build_catboost_model(iterations=1000, depth=5, learning_rate=0.1, random_state=37):

    # Create a CatBoost classifier
    model = CatBoostClassifier(
        iterations=iterations,     # Number of boosting rounds
        depth=depth,               # Depth of each decision tree
        learning_rate=learning_rate,  # Learning rate for boosting
        random_seed=random_state,     # Random seed for reproducibility
        eval_metric="AUC",            # Evaluation metric = AUC
        loss_function="Logloss",      # Binary classification loss function
        verbose=False,                 # Suppress training output
        allow_writing_files=False

    )
    return model  # Return the constructed model


# === Build and train the model ===
cat_model = build_catboost_model()   # Initialize the model with default parameters
cat_model.fit(X_train, y_train)      # Train the model on training data (features and labels)


# === Function to build a LightGBM model ===
def build_lightgbm_model(n_estimators=2000, max_depth=-1, learning_rate=0.05, random_state=66):


    # Create a LightGBM classifier
    model = lgb.LGBMClassifier(
        objective="binary",          # Binary classification
        n_estimators=n_estimators,   # Number of boosting rounds
        max_depth=max_depth,         # Maximum depth of each tree
        learning_rate=learning_rate, # Step size for gradient boosting
        random_state=random_state,   # Set random seed
        metric="auc",                 # Evaluation metric = AUC
        verbose = -1
    )
    return model  # Return the constructed model


# === Build and train the model ===
lgb_model = build_lightgbm_model()   # Initialize the model with default parameters
lgb_model.fit(X_train, y_train)      # Train the model on training data (features and labels)


# === Function to build a HistGradientBoosting model === # best 37 200
def build_hgb_model(max_iter=200, max_depth=None, random_state=33):
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,   # Number of boosting iterations
        max_depth=max_depth, # Maximum depth of trees
        random_state=random_state
    )
    return model

# === Build and train the model ===
hgb_model = build_hgb_model()
hgb_model.fit(X_train, y_train) 


def evaluate_metrics(y_true, y_pred_proba):
    results = []

    # Binarize predictions using a 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Function to calculate evaluation metrics
    def calculate_metrics(y_true, y_pred, y_pred_proba):
        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        auc       = roc_auc_score(y_true, y_pred_proba)  

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
        }

    results.append(calculate_metrics(y_true, y_pred, y_pred_proba))
    return pd.DataFrame(results)



xgb_prob = xgb_model.predict_proba(X_test)[:, 1] 
cat_prob = cat_model.predict_proba(X_test)[:, 1]
lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
hgb_prob  = hgb_model.predict_proba(X_test)[:, 1]


xgb_pred = xgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
hgb_pred = hgb_model.predict(X_test)


xgb_results = evaluate_metrics(y_test, xgb_prob).assign(Model="XGBoost")
cat_results = evaluate_metrics(y_test, cat_prob).assign(Model="CatBoost")
lgb_results = evaluate_metrics(y_test, lgb_prob).assign(Model="LightGBM")
hgb_results  = evaluate_metrics(y_test, hgb_prob).assign(Model="HistGradientBoosting")

results = pd.concat([xgb_results, cat_results, lgb_results, hgb_results], ignore_index=True)
results = results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]

display(results)


models = {
    'XGBoost': xgb_pred,
    'CatBoost': cat_pred,
    'LightGBM': lgb_pred,
    'HistGradientBoosting': hgb_pred
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
axes = axes.flatten()

for ax, (name, pred) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f'{name}')

plt.tight_layout()
plt.show()


# --- Optimization Function (Maximize AUC) ---
def optimize_weight(trial):
    # Suggest weights for each model between 0 and 1
    w_xgb = trial.suggest_float('xgb_weight', 0.0, 1.0)
    w_lgb = trial.suggest_float('lgb_weight', 0.0, 1.0)
    w_hgb  = trial.suggest_float('hgb_weight', 0.0, 1.0)
    w_cat = trial.suggest_float('cat_weight', 0.0, 1.0)

    total_weight = w_xgb + w_lgb + w_hgb + w_cat
    if total_weight == 0:
        return 0.5  # Return neutral AUC if all weights are zero

    # Normalize weights so they sum to 1
    w_xgb /= total_weight
    w_lgb /= total_weight
    w_hgb  /= total_weight
    w_cat /= total_weight

    # --- Use probability outputs ---
    final_prob = (
        w_xgb * xgb_prob +
        w_lgb * lgb_prob +
        w_hgb  * hgb_prob +
        w_cat * cat_prob
    )

    # Handle edge case where y_test has only one class
    try:
        auc = roc_auc_score(y_test, final_prob)
    except ValueError:
        auc = 0.5

    return auc  # Optuna will maximize this


# --- Run Optuna Study (maximize AUC) ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optimize_weight, n_trials=1000, show_progress_bar=True)

# --- Retrieve Best Weights ---
best_params = study.best_params
best_w_xgb = best_params['xgb_weight']
best_w_lgb = best_params['lgb_weight']
best_w_hgb  = best_params['hgb_weight']
best_w_cat = best_params['cat_weight']

# Normalize again to ensure sum = 1
total_weight = best_w_xgb + best_w_lgb + best_w_hgb + best_w_cat
best_w_xgb /= total_weight
best_w_lgb /= total_weight
best_w_hgb  /= total_weight
best_w_cat /= total_weight

# --- Final Weighted Probability Prediction ---
final_prob = (
    best_w_xgb * xgb_prob +
    best_w_lgb * lgb_prob +
    best_w_hgb  * hgb_prob +
    best_w_cat * cat_prob
)

# --- Compute Metrics ---
try:
    best_auc = roc_auc_score(y_test, final_prob)
except ValueError:
    best_auc = 0.5

# --- Convert to Binary Predictions (threshold = 0.5) ---
threshold = 0.5
final_pred_binary = (final_prob >= threshold).astype(int)

# --- Display Results ---
print("\n=== Optimized Weights ===")
print(f"XGBoost: {best_w_xgb:.4f}")
print(f"LightGBM: {best_w_lgb:.4f}")
print(f"HistGradientBoosting: {best_w_hgb:.4f}")
print(f"CatBoost: {best_w_cat:.4f}")
print(f"\nFinal AUC: {best_auc:.4f}")

# --- Additional Output ---
print("\n=== Example of Predictions ===")
print("Probabilities (final_prob):", final_prob[:10])
print("Binary (final_pred_binary):", final_pred_binary[:10])



xgb_results = evaluate_metrics(y_test, xgb_prob).assign(Model="XGBoost")
cat_results = evaluate_metrics(y_test, cat_prob).assign(Model="CatBoost")
lgb_results = evaluate_metrics(y_test, lgb_prob).assign(Model="LightGBM")
hgb_results  = evaluate_metrics(y_test, hgb_prob).assign(Model="HistGradientBoosting")
encode_results = evaluate_metrics(y_test, final_prob).assign(Model="Optimized Ensemble")

results = pd.concat([xgb_results, cat_results, lgb_results, hgb_results,encode_results], ignore_index=True)
results = results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]

display(results)


def plot_confusion(y_true, y_pred, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()

encode_results_prob = cat_model.predict_proba(predict_X)[:, 1]
plot_confusion(y_test, final_pred_binary, normalize=True, title='Normalized Confusion Matrix')


xgb_pred_new = xgb_model.predict_proba(predict_X)[:, 1] 
lgb_pred_new = lgb_model.predict_proba(predict_X)[:, 1]
hgb_pred_new  = hgb_model.predict_proba(predict_X)[:, 1]
cat_pred_new = cat_model.predict_proba(predict_X)[:, 1]

final_pred_new = (
    best_w_xgb * xgb_pred_new +
    best_w_lgb * lgb_pred_new +
    best_w_hgb  * hgb_pred_new +
    best_w_cat * cat_pred_new
)
predict_df = pd.DataFrame(final_pred_new, columns=['loan_paid_back_proba'])

submission = pd.concat([predict['id'], predict_df], axis=1)

display(submission.head())
print(submission.isnull().sum())



# --- Save to CSV for Kaggle submission ---
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved as 'submission.csv'")

