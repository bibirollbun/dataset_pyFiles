import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
# Suppress warnings for clean output
warnings.filterwarnings('ignore')



# 1. Load the Data
# ----------------------------
train_file = '/kaggle/input/thapar-kaggle-hack-v02/train.csv'
test_file = '/kaggle/input/thapar-kaggle-hack-v02/test.csv'
id_column = 'id'
target_column = 'target'

df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


df.head(5)


test_df.head(5)


# Separate features and target from training data
X = df.drop(columns=[target_column, id_column])
y = df[target_column]


# 2. Preprocessing
# ----------------------------
# Fill missing values with column means
X.fillna(X.mean(), inplace=True)

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# 3. Hyperparameter Tuning with RandomizedSearchCV
# ----------------------------
# Define parameter grid for CatBoost
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [500, 750, 1000],
    'l2_leaf_reg': [1, 3, 5, 7]
}


# Initialize CatBoostClassifier
cat_model = cb.CatBoostClassifier(loss_function='MultiClass', verbose=0, random_state=42)

# Use StratifiedKFold to preserve class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=cv,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

# Fit on the entire training set
random_search.fit(X_scaled, y)
print("Best parameters found:", random_search.best_params_)


# Build a new CatBoostClassifier with the best parameters
best_params = random_search.best_params_
cat_model_best = cb.CatBoostClassifier(loss_function='MultiClass',
                                         **best_params,
                                         verbose=0,
                                         random_state=42)



# 4. Model Evaluation using Stratified K-Fold CV with Early Stopping
# ----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, valid_index in kf.split(X_scaled, y):
    X_train_fold, X_valid_fold = X_scaled.iloc[train_index], X_scaled.iloc[valid_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
    
    # Train with early stopping; note that eval_set is used for early stopping
    cat_model_best.fit(X_train_fold, y_train_fold,
                       early_stopping_rounds=50,
                       eval_set=(X_valid_fold, y_valid_fold))
    
    y_pred_fold = cat_model_best.predict(X_valid_fold)
    fold_acc = accuracy_score(y_valid_fold, y_pred_fold)
    accuracies.append(fold_acc)

print("CatBoost Mean Accuracy after tuning:", np.mean(accuracies))



# ----------------------------
# 5. Train Final Model on Full Training Data
# ----------------------------
cat_model_best.fit(X_scaled, y, early_stopping_rounds=50, verbose=100)

# ----------------------------
# 6. Prepare Test Data and Make Predictions
# ----------------------------
# Save the test IDs
test_ids = test_df[id_column]

# Drop the id column and fill missing values
test_features = test_df.drop(columns=[id_column])
test_features.fillna(test_features.mean(), inplace=True)

# Apply the same scaling as the training data
test_features_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

# Make predictions on test data
test_pred = cat_model_best.predict(test_features_scaled).ravel()

# ----------------------------
# 7. Create and Save Submission File
# ----------------------------
submission = pd.DataFrame({id_column: test_ids, target_column: test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")

