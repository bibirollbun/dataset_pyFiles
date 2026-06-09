!pip -q install autogluon


import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Quick data examination
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Target distribution: {train_df['rainfall'].value_counts(normalize=True)}")


# Check for missing values
missing_values = train_df.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values detected:")
    print(missing_values[missing_values > 0])
else:
    print("No missing values in the training data.")


# Define AutoGluon training parameters for fast execution
time_limit = 600  # 10 minutes (600 seconds)
ag_predictor_params = {
    'eval_metric': 'roc_auc',  # Competition metric
    'problem_type': 'binary',
    'label': 'rainfall',
}


# Initialize and train the AutoGluon predictor with fast settings
print("\nTraining AutoGluon model with time limit of 10 minutes...")
predictor = TabularPredictor(**ag_predictor_params).fit(
    train_data=train_df,
    time_limit=time_limit,
    presets='best_quality',  # Use medium_quality instead of best_quality for speed
    num_bag_folds=5,  # Reduce number of bagging folds
    num_bag_sets=1,   # Only use one bagging set
    num_stack_levels=1,  # Limit stacking to 1 level
    hyperparameters={
        'GBM': {'num_boost_round': 100},  # Limit boosting rounds
        'CAT': {'iterations': 100},       # Limit CatBoost iterations
        'RF': {'n_estimators': 100},      # Limit number of trees
        'XT': {'n_estimators': 100},      # Limit number of trees
        'XGB': {'n_estimators': 100}      # Limit number of trees
    },
    excluded_model_types=['NN', 'KNN'],  # Exclude slower models
    verbosity=2
)


# Show the best models
print("\nModel leaderboard:")
leaderboard = predictor.leaderboard(silent=False)


# Make predictions on test data
print("\nMaking predictions on test data...")
test_predictions = predictor.predict_proba(test_df)
test_predictions_rainfall = test_predictions[1]  # Probability of rainfall (class 1)


# Create submission file
submission = sample_submission.copy()
submission['rainfall'] = test_predictions_rainfall
submission.to_csv('autogluon_fast_submission.csv', index=False)
print("\nSubmission file created: autogluon_fast_submission.csv")


# Quick validation check with holdout data (optional, if time permits)
from sklearn.model_selection import train_test_split

# Split off a small validation set
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
val_predictions = predictor.predict_proba(val_data)
val_score = roc_auc_score(val_data['rainfall'], val_predictions[1])
print(f"\nValidation ROC AUC: {val_score:.4f}")

print("\nProcess completed successfully!")

