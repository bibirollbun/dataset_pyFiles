import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import os


train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')


# Extract target from ID by removing the last segment after final underscore
train_labels['target'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
# Merge train_sequences with train_labels using left join on target_id and target
train = train_sequences.merge(train_labels, how='left', left_on='target_id', right_on='target')
# Preprocess data for modeling
# Convert temporal cutoff to numerical feature
train['temporal_cutoff'] = pd.to_datetime(train['temporal_cutoff']).astype('int64') // 10**9
test_sequences['temporal_cutoff'] = pd.to_datetime(test_sequences['temporal_cutoff']).astype('int64') // 10**9




submission['target'] = submission['ID'].str.rsplit('_', n=1).str[0]


test = test_sequences.merge(submission, how='left', left_on='target_id', right_on='target')
test['temporal_cutoff'] = pd.to_datetime(test['temporal_cutoff']).astype('int64') // 10**9



# Create sequence length feature
train['seq_length'] = train['sequence'].str.len()
test['seq_length'] = test['sequence'].str.len()


categorical_columns = ['ID', 'target_id', 'description','all_sequences','sequence','target','resname']


# Label encode categorical columns
from sklearn.preprocessing import LabelEncoder

# Create dictionary to store label encoders
label_encoders = {}

for col in categorical_columns:
    # Handle potential NaN values before encoding
    train[col] = train[col].astype(str).fillna('missing')
    
    # Create and fit label encoder
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le

for col in categorical_columns:
    # Handle potential NaN values before encoding
    test[col] = test[col].astype(str).fillna('missing')
    
    # Create and fit label encoder
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le

# Optional: Add embedding dimensions for high-cardinality features
# (This would require neural network approach and more complex handling)



# Features and targets for training
features = ['temporal_cutoff', 'resname', 'resid', 'seq_length', 'target_id','ID','description','all_sequences']
coord_targets = ['x_1','y_1','z_1']


models = {}
for coord in coord_targets:
    print(f'Training model for {coord}...')
    
    # Set up parameter grid for tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [500, 1000, 1500],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 2, 3]
    }
    
    # Base model for grid search
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        early_stopping_rounds=10,
        eval_metric='rmse'
    )
    
    # Handle missing values
    mean_val = train[coord].mean()
    train[coord].fillna(mean_val, inplace=True)
    valid_idx = train[coord].notna()
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(
        train[valid_idx][features], 
        train[valid_idx][coord],
        eval_set=[(train[valid_idx][features], train[valid_idx][coord])],
        verbose=0
    )
    
    # Store best model
    best_model = grid_search.best_estimator_
    print(f'Best parameters for {coord}: {grid_search.best_params_}')
    models[coord] = best_model


# models = {}
# for coord in coord_targets:
#     print(f'Training model for {coord}...')
#     model = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         n_estimators=1000,
#         max_depth=7,
#         learning_rate=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8
#     )
    
#     # Filter out rows with missing coordinates
#     mean_val = train[coord].mean()
#     train[coord].fillna(mean_val, inplace=True)
#     valid_idx = train[coord].notna()  # Now just indicates all rows are valid
#     model.fit(train[valid_idx][features], train[valid_idx][coord])
#     models[coord] = model


test = test[['ID','resname','resid','sequence','target_id','target','description','all_sequences','temporal_cutoff','seq_length']]


# Predict coordinates for test set
test_predictions = test.copy()
for coord in coord_targets:
    # Use trained model to make predictions
    test_predictions[coord] = models[coord].predict(test[features])
    
# Display sample predictions
print("Test predictions sample:")
test_predictions[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']].head()



# Create x_2-x_5, y_2-y_5, z_2-z_5 columns with same values as x_1/y_1/z_1
for i in range(2, 6):
    test_predictions[f'x_{i}'] = test_predictions['x_1']
    test_predictions[f'y_{i}'] = test_predictions['y_1']
    test_predictions[f'z_{i}'] = test_predictions['z_1']



submission[['x_1','y_1','z_1','x_2','y_2','z_2','x_3','y_3','z_3','x_4','y_4','z_4','x_5','y_5','z_5']] = test_predictions[['x_1','y_1','z_1','x_2','y_2','z_2','x_3','y_3','z_3','x_4','y_4','z_4','x_5','y_5','z_5']] 



submission.to_csv('submission.csv', index=False)




