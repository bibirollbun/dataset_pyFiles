import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold 
from catboost import CatBoostClassifier, Pool
import optuna


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')  # Update with your actual file path
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')    # Update with your actual file path


# Display information about the DataFrames
print("\nTrain Data Describe: ")
train_data.describe().T


# Display information about the DataFrames
print("\nTest Data Describe:")
test_data.describe().T


print(train_data)
display(train_data.head())
print(test_data)
display(test_data.head())


# Select features and target variable
X = train_data.drop(['id','rainfall'], axis=1)
y = train_data['rainfall']
X_test = test_data.drop(['id'], axis=1)


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)



def objective(trial):
    # Define the hyperparameter search space
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50),
    }
    
    # Add conditional parameters - ONLY add bagging_temperature if bootstrap_type is Bayesian
    if param['bootstrap_type'] == 'Bayesian':
        param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    
    # Use same CV split strategy as your original code
    FOLDS = 13
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    # Store out-of-fold predictions
    oof_preds = np.zeros(len(y))
    
    # Cross-validation
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize model with trial parameters
        model = CatBoostClassifier(
            **param,
            random_state=42,
            verbose=0
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        # Predict on validation set
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Calculate AUC score
    auc_score = roc_auc_score(y, oof_preds)
    
    return auc_score

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

# Get best parameters
best_params = study.best_params
print('Best parameters:', best_params)
print('Best AUC score:', study.best_value)

# Train the final model with the best parameters
best_model = CatBoostClassifier(
    **best_params,
    random_state=42,
    verbose=0
)


# Plot ROC curves
plt.figure(figsize=(8, 6))
for best_model, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{best_model} (AUC = {auc_score:.5f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# Find the best model overall
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f"Best Model Overall: {best_model_name} with AUC = {auc_scores[best_model_name]:.5f}")


# Select the best model based on AUC
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
    
# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': train_data.drop(['rainfall', 'id'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)


# List of top N features to try 
top_feature_counts = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18]

# Variables to track the best AUC and corresponding top features
best_auc_top = 0
best_top_n = 0
best_oof_preds_top = None

# Loop over different top feature counts
for top_n in top_feature_counts:
    # Select the top N features
    top_features = feature_df.head(top_n)['Feature']
    
    # Prepare the data with the selected top N features
    X_top = X[:, [train_data.drop(['rainfall', 'id'], axis=1).columns.get_loc(col) for col in top_features]]
    X_test_top = X_test[:, [train_data.drop(['rainfall', 'id'], axis=1).columns.get_loc(col) for col in top_features]]

    # Retrain the best model using the top N features
    best_model.fit(X_top, y)

    # Make predictions and calculate AUC for the top N features
    oof_preds_top = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X_top, y):
        X_train, X_val = X_top[train_idx], X_top[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        oof_preds_top[val_idx] = best_model.predict_proba(X_val)[:, 1]

    # Calculate and print AUC score for top N features model
    auc_score_top = roc_auc_score(y, oof_preds_top)
    print(f"AUC for top {top_n} features model: {auc_score_top:.4f}")
    
    # Track the best AUC and corresponding features
    if auc_score_top > best_auc_top:
        best_auc_top = auc_score_top
        best_top_n = top_n
        best_oof_preds_top = oof_preds_top

# Now plot the feature importance for the set with the highest AUC
best_features = feature_df.head(best_top_n)


# Predictions for the test set with the top N features
test_preds = best_model.predict_proba(X_test_top)[:, 1]

# Submission
submission = pd.DataFrame({'id': test_data['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")


submission

