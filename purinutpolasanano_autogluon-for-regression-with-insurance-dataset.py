pip -q install autogluon


# Import necessary libraries
import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv', index_col=[0])
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv', index_col=[0])


# Identify target column
target_column = (set(train.columns) - set(test.columns)).pop()
print(f"Target column: {target_column}")


# Quick feature engineering
print("Performing quick feature engineering...")
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target_column]


# Create a few high-value interaction features (limited to save time)
if len(numeric_cols) >= 2:
    # Sort features by correlation with target to prioritize important ones
    corr_with_target = train[numeric_cols].corrwith(train[target_column]).abs().sort_values(ascending=False)
    top_features = corr_with_target.index[:5].tolist()  # Take only top 5 features
    
    for i, col1 in enumerate(top_features):
        for col2 in top_features[i+1:]:
            train[f'{col1}_mult_{col2}'] = train[col1] * train[col2]
            test[f'{col1}_mult_{col2}'] = test[col1] * test[col2]


# Initialize and train predictor with time-optimized settings
print("Training AutoGluon model (10 minute time limit)...")
predictor = TabularPredictor(
    label=target_column,
    path='autogluon_output',
    problem_type='regression',
    eval_metric='root_mean_squared_error'
)


# Configure training to fit within 10 minutes
predictor.fit(
    train_data=train,
    time_limit=600,  # 10 minutes total
    presets='medium_quality_faster_train',  # Faster training preset
    hyperparameters={
        'GBM': [  # LightGBM - fast and effective
            {'num_boost_round': 100, 'num_leaves': 31}
        ],
        'XGB': [  # XGBoost with limited iterations
            {'n_estimators': 100, 'max_depth': 6}
        ],
        'RF': [  # Random Forest with fewer trees
            {'n_estimators': 100}
        ]
    },
    # Disable slower models to save time
    excluded_model_types=['NN_TORCH', 'CAT', 'KNN'],
    num_bag_folds=3,  # Still use some bagging for robustness
    num_bag_sets=1,
    num_stack_levels=0,  # Disable stacking to save time
    verbosity=2
)



# Quick model evaluation
print("\nModel performance:")
leaderboard = predictor.leaderboard(silent=False)


# Generate and save feature importance if available
try:
    importance = predictor.feature_importance(train)
    print("\nTop 10 important features:")
    print(importance.head(10))
except:
    print("Feature importance calculation unavailable")

# Make predictions on test data
print("\nGenerating predictions...")
test_pred = predictor.predict(test)


# Save predictions for submission
submission = pd.DataFrame({
    'Premium Amount': test_pred
})
submission.index = test.index
submission.to_csv('submission.csv')

print(f"Predictions saved to submission.csv")
print(f"Prediction summary: min={test_pred.min():.4f}, max={test_pred.max():.4f}, mean={test_pred.mean():.4f}")

