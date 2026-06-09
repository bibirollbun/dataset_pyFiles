pip install autogluon


# Import necessary modules
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split


# Load and prepare the data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


# Feature engineering function
def add_features(df):
    # Create interaction features
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
    
    # Convert binary features to numeric
    binary_map = {'Yes': 1, 'No': 0}
    df['Laptop Compartment'] = df['Laptop Compartment'].map(binary_map)
    df['Waterproof'] = df['Waterproof'].map(binary_map)
    
    return df


# Apply feature engineering to all datasets
train = add_features(train)
train_extra = add_features(train_extra)
test = add_features(test)


# Combine train and train extra data sets
df = pd.concat([train, train_extra], axis=0, ignore_index=True)


# Define target column
target = 'Price'


# Split data into train and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)


# Initialize AutoGluon predictor with time constraints
predictor = TabularPredictor(
    label=target,
    problem_type='regression',
    eval_metric='root_mean_squared_error',
    path='ag_models_backpack'
).fit(
    train_data=train_data,
    tuning_data=val_data,
    # Use medium_quality preset instead of best_quality for faster training
    presets='medium_quality',
    # Set a strict 10-minute time limit (600 seconds)
    time_limit=600,
    # Skip hyperparameter tuning to save time
    hyperparameters='default',
    # Limit model types to faster ones
    excluded_model_types=['KNN', 'NN_TORCH', 'FASTAI'],
    verbosity=2
)


# Evaluate on validation data
performance = predictor.evaluate(val_data)
print("Validation performance:", performance)


# Generate predictions on test data
test_pred = predictor.predict(test)


# Create submission file
submission = pd.DataFrame({'id': test.index, 'Price': test_pred})
submission.set_index('id', inplace=True)
submission.to_csv('submission.csv')
print("Submission file created")

