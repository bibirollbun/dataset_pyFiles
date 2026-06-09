import sys
print(sys.version)


# Completely remove existing packages
!pip uninstall -y scikit-learn autogluon.tabular autogluon.core autogluon.features autogluon.common autogluon mxnet torch torchvision

# Install specific versions known to work together
!pip install --no-cache-dir scikit-learn==1.2.2
!pip install --no-cache-dir mxnet==1.9.1 torch==1.13.1 torchvision==0.14.1
!pip install --no-cache-dir autogluon==0.7.0


# --------------------------------------------------------
# Import Libraries
# --------------------------------------------------------
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from autogluon.tabular import TabularPredictor
import joblib

# --------------------------------------------------------
# Options
# --------------------------------------------------------

TIME_LIMIT_FOLD = 3600 * 0.25
TIME_LIMIT = 3600 * 5

# --------------------------------------------------------
# Load Data
# --------------------------------------------------------
train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

# --------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Drop rows with missing target
train_data = train_data.dropna(subset=['num_sold'])
print("Train shape after dropping missing target:", train_data.shape)

# Create date-based features
for df in [train_data, test_data]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek

# --------------------------------------------------------
# Sort Training Data by Date
# --------------------------------------------------------
train_data_sorted = train_data.sort_values(by='date').reset_index(drop=True)

# --------------------------------------------------------
# Split Features & Target
# --------------------------------------------------------
# Define feature columns (excluding 'id', 'date', 'num_sold')
feature_cols = [col for col in train_data_sorted.columns if col not in ['id', 'date', 'num_sold']]
X_sorted = train_data_sorted[feature_cols]
y_sorted = train_data_sorted['num_sold']

# Prepare Test Features
X_test = test_data.drop(columns=['id', 'date'])

# --------------------------------------------------------
# Generate Timestamp
# --------------------------------------------------------
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------------------------------------
# Time-based Cross Validation with OOF Predictions
# --------------------------------------------------------
# Initialize OOF predictions array
oof_predictions = np.zeros(len(train_data_sorted))
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_index, valid_index) in enumerate(tscv.split(X_sorted), 1):
    # Split data
    X_train_cv, X_valid_cv = X_sorted.iloc[train_index], X_sorted.iloc[valid_index]
    y_train_cv, y_valid_cv = y_sorted.iloc[train_index], y_sorted.iloc[valid_index]
    
    # Combine X and y for AutoGluon
    train_cv = X_train_cv.copy()
    train_cv['num_sold'] = y_train_cv
    valid_cv = X_valid_cv.copy()
    valid_cv['num_sold'] = y_valid_cv
    
    # Initialize AutoGluon Predictor
    predictor = TabularPredictor(label='num_sold', problem_type='regression').fit(
        train_data=train_cv,
        presets='best_quality',  # You can choose 'medium_quality' or other presets
        verbosity=0,
        time_limit=TIME_LIMIT_FOLD
    )
    
    # Predict on validation set
    preds = predictor.predict(valid_cv)
    
    # Store OOF predictions
    oof_predictions[valid_index] = preds
    
    # Compute MAPE
    mape = mean_absolute_percentage_error(y_valid_cv, preds)
    scores.append(mape)
    print(f"Fold {fold} MAPE: {mape:.2%}")

print("TimeSeriesSplit MAPE Scores:", scores)
print("Average MAPE:", np.mean(scores))

# --------------------------------------------------------
# Save OOF Predictions
# --------------------------------------------------------
oof_df = pd.DataFrame({
    'id': train_data_sorted['id'],
    'oof_num_sold': oof_predictions
})

oof_filename = f"oof_predictions_m05_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# --------------------------------------------------------
# Train on Full Dataset & Predict on Test
# --------------------------------------------------------
# Combine X and y for full training
full_train = X_sorted.copy()
full_train['num_sold'] = y_sorted

# Initialize and train the predictor on the full dataset
final_predictor = TabularPredictor(label='num_sold', problem_type='regression').fit(
    train_data=full_train,
    presets='best_quality',
    verbosity=0,
    time_limit=TIME_LIMIT
)

# Predict on test data
test_preds = final_predictor.predict(X_test)

# --------------------------------------------------------
# Save Trained Model
# --------------------------------------------------------
model_filename = f"model_05_{timestamp_str}.pkl"
final_predictor.save(model_filename)
print(f"Trained model saved as {model_filename}")

# --------------------------------------------------------
# Submission
# --------------------------------------------------------
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_preds
})

submission_filename = f"sub_m05_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")

