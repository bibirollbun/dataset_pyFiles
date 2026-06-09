import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

# --- Use RAPIDS KNN (GPU) if available; otherwise, you can substitute with scikit-learn's version ---
from cuml.neighbors import KNeighborsClassifier  
# For CPU-only you could alternatively use:
# from sklearn.neighbors import KNeighborsClassifier


# ===============================
# 1. Load Data
# ===============================
train = pd.read_csv("/kaggle/input/final-rainfall/final_Rainfall.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# Standardize column names (strip whitespace)
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()
sample_submission.columns = sample_submission.columns.str.strip()

# Save test IDs then drop the id column from test for modeling
test_ids = test["id"]
test.drop(columns=["id"], inplace=True)



# ===============================
# 2. Preprocessing & Feature Engineering
# ===============================
feature_cols = [col for col in train.columns if col != 'rainfall']

imputer = KNNImputer(n_neighbors=5)
train[feature_cols] = imputer.fit_transform(train[feature_cols])
test = pd.DataFrame(imputer.transform(test), columns=test.columns)

# Create interaction features in both datasets.
for df in [train, test]:
    df['cloud_sun'] = df['cloud'] * df['sunshine']
    df['pressure_wind'] = df['pressure'] * df['windspeed']
    df['humidity_pressure'] = df['humidity'] / (df['pressure'] + 1)


# ===============================
# 3. Prepare Data for Stacking Ensemble
# ===============================
# We'll obtain out-of-fold (OOF) predictions from both models using 5-fold CV.
X = train.drop(columns=['rainfall'])
y = train['rainfall']

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)

# Initialize arrays to store OOF predictions for the entire training set
oof_cat = np.zeros(len(train))
oof_knn = np.zeros(len(train))
# Also, prepare to accumulate test predictions from each fold
test_pred_cat = np.zeros(len(test))
test_pred_knn = np.zeros(len(test))


# ===============================
# 4. CatBoost Model: Cross-Validation Predictions
# ===============================
# Define parameter grid and train CatBoost via GridSearchCV on the entire training set
param_grid = {
    'iterations': [500],
    'learning_rate': [0.01],
    'depth': [4, 6],
    'l2_leaf_reg': [1,5]
}

cat_model = CatBoostClassifier(
    verbose=100, random_seed=42, eval_metric='AUC', loss_function='Logloss',
    thread_count=-1
)

grid_search = GridSearchCV(
    estimator=cat_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X, y)
best_cat = grid_search.best_estimator_
print("Best CatBoost Parameters:", grid_search.best_params_)

# Obtain OOF predictions for CatBoost using cross_val_predict:
oof_cat = cross_val_predict(best_cat, X, y, cv=FOLDS, method="predict_proba")[:, 1]
cat_auc_oof = roc_auc_score(y, oof_cat)
print("CatBoost OOF AUC:", cat_auc_oof)

# Train best_cat on full training data, then predict on test
best_cat.fit(X, y)
test_pred_cat = best_cat.predict_proba(test)[:, 1]



# ===============================
# 5. KNN Model: 5-Fold CV with Standardization
# ===============================
KNN_FEATURES = [col for col in train.columns if col != 'rainfall']

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {fold+1}")
    print("#" * 25)
    
    x_train_knn = train.loc[train_idx, KNN_FEATURES].copy()
    x_valid_knn = train.loc[valid_idx, KNN_FEATURES].copy()
    x_test_knn = test[KNN_FEATURES].copy()
    y_train_knn = train.loc[train_idx, "rainfall"].values
    
    # Standardize features using training fold statistics
    for col in KNN_FEATURES:
        m = x_train_knn[col].mean()
        s = x_train_knn[col].std()
        x_train_knn[col] = (x_train_knn[col] - m) / s
        x_valid_knn[col] = (x_valid_knn[col] - m) / s
        x_test_knn[col] = (x_test_knn[col] - m) / s
        x_train_knn[col] = x_train_knn[col].fillna(0)
        x_valid_knn[col] = x_valid_knn[col].fillna(0)
        x_test_knn[col] = x_test_knn[col].fillna(0)
    
    # Convert DataFrames to NumPy arrays for RAPIDS KNN
    X_train_knn_arr = x_train_knn.values
    X_valid_knn_arr = x_valid_knn.values
    X_test_knn_arr = x_test_knn.values
    
    # Train RAPIDS KNN with n_neighbors=101 and Manhattan distance
    knn_model = KNeighborsClassifier(n_neighbors=101, metric="manhattan")
    knn_model.fit(X_train_knn_arr, y_train_knn)
    
    # Out-of-fold predictions for current fold
    oof_knn[valid_idx] = knn_model.predict_proba(X_valid_knn_arr)[:, 1]
    # Accumulate test predictions
    test_pred_knn += knn_model.predict_proba(X_test_knn_arr)[:, 1]

# Average test predictions over folds
test_pred_knn /= FOLDS
knn_auc = roc_auc_score(y, oof_knn)
print("KNN OOF AUC:", knn_auc)


# ===============================
# 6. Stacking Ensemble using Logistic Regression
# ===============================
# Combine the OOF predictions from both models as features
X_stack = np.column_stack([oof_cat, oof_knn])
stack_model = LogisticRegression(random_state=42)
stack_model.fit(X_stack, y)

# Evaluate stacking on training data (via OOF predictions)
stack_oof = stack_model.predict_proba(X_stack)[:, 1]
stack_auc = roc_auc_score(y, stack_oof)
print("Stacking OOF AUC:", stack_auc)

# For the test set, combine predictions from both models
X_stack_test = np.column_stack([test_pred_cat, test_pred_knn])
ensemble_pred = stack_model.predict_proba(X_stack_test)[:, 1]


# # ===============================
# # 7. Optional Condition-Based Adjustment
# # ===============================
# adjustment_factor = 1.007
# condition1 = (test['cloud'] > 73.5) & (test['sunshine'] < 0.5) & \
#              (test['pressure'] <= 1020.35) & (test['windspeed'] > 20.35)
# ensemble_pred[condition1] = np.clip(ensemble_pred[condition1] * adjustment_factor, 0, 1)



# ===============================
# 8. Save Submission File
# ===============================
submission = pd.DataFrame({'id': test_ids, 'rainfall': ensemble_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")

