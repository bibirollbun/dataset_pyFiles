import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb



import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv('../input/rajan1/train.csv')
test_df = pd.read_csv('../input/rajan1/test.csv')

# Print dataset information
print("Train DataFrame Shape:", train_df.shape)
print("Test DataFrame Shape:", test_df.shape)

# Print column information
print("\nTrain Columns:")
print(train_df.columns.tolist())
print("\nTest Columns:")
print(test_df.columns.tolist())

# Check for any preprocessing issues
print("\nMissing Values in Train:")
print(train_df.isnull().sum())
print("\nMissing Values in Test:")
print(test_df.isnull().sum())


def preprocessing(train_df, test_df):
    # Identify columns present in both datasets
    common_columns = list(set(train_df.columns) & set(test_df.columns))
    common_columns = [col for col in common_columns if col not in ['ID', 'efs', 'efs_time']]
    
    # One-hot encode categorical columns
    categorical_columns = train_df[common_columns].select_dtypes(include=['object']).columns
    
    # Combine train and test for consistent encoding
    combined_df = pd.concat([train_df[common_columns], test_df[common_columns]], axis=0)
    combined_encoded = pd.get_dummies(combined_df, columns=categorical_columns)
    
    # Split back into train and test
    train_encoded = combined_encoded.iloc[:len(train_df)]
    test_encoded = combined_encoded.iloc[len(train_df):]
    
    # Prepare train data
    X = train_encoded
    T = train_df['efs_time']
    E = train_df['efs']
    
    # Prepare test data
    X_test = test_encoded
    
    return X, T, E, X_test, test_df['ID']


def advanced_cv_prediction(X, T, E, X_test, n_splits=5):
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Prediction storage
    test_predictions = np.zeros(len(X_test))
    
    # Advanced LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'dart',
        'num_leaves': 127,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 10,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbosity': -1
    }
    
    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, E), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        T_train, T_val = T.iloc[train_idx], T.iloc[val_idx]
        E_train, E_val = E.iloc[train_idx], E.iloc[val_idx]
        
        train_data = lgb.Dataset(
            X_train, 
            label=-T_train, 
            weight=E_train
        )
        
        val_data = lgb.Dataset(
            X_val, 
            label=-T_val, 
            weight=E_val
        )
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
        )
        
        # Predict test data
        test_predictions += model.predict(X_test_scaled) / n_splits
    
    return test_predictions



# Main execution
train_df = pd.read_csv('../input/rajan1/train.csv')
test_df = pd.read_csv('../input/rajan1/test.csv')

# Preprocessing
X, T, E, X_test, test_ids = preprocessing(train_df, test_df)



# Prediction
predictions = advanced_cv_prediction(X, T, E, X_test)



# Create submission with normalization
submission = pd.DataFrame({
    'ID': test_ids,
    'prediction': (predictions - predictions.min()) / (predictions.max() - predictions.min())
})

# Save submission
submission.to_csv('submission.csv', index=False)



# Print diagnostics
print("Submission Preview:")
print(submission.head())
print("\nPrediction Statistics:")
print(submission['prediction'].describe())





