import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings

warnings.filterwarnings('ignore') # To not display warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_list=[
    pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv'),
    pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
]
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


dataset = pd.concat(dataset_list, ignore_index=True, axis=0)
del dataset_list


dataset.shape


dataset.head(10)


df_test.head()


dataset.drop(columns=['id'],inplace=True)
df_test.drop(columns=['id'],inplace=True)


dataset.info()


df_test.info()


for column in dataset.columns:
    plt.figure()
    dataset[column].hist(bins=30)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


dataset.isnull().sum()


df_test.isnull().sum()


import xgboost as xgb
features=dataset.select_dtypes(include=['object']).columns


# One-Hot Encoder for categorical data (except the one being imputed)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


for col in features:
    print(f"Imputing missing values for: {col}")

    # Identify missing values
    missing_mask = dataset[col].isna()

    if missing_mask.sum() > 0:  # If missing values exist
        # Split data into train (no missing) and test (missing)
        train_data = dataset[~missing_mask]
        test_data = dataset[missing_mask]

        # One-hot encode all other categorical features (excluding 'col' being imputed)
        encoded_cols = [f for f in features if f != col]

        # Fit-transform on train & transform on test
        X_train_encoded = encoder.fit_transform(train_data[encoded_cols])
        X_test_encoded = encoder.transform(test_data[encoded_cols])

        # Convert NumPy array to DataFrame and add column names
        encoded_feature_names = encoder.get_feature_names_out(encoded_cols)
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names)

        # Reset index to avoid mismatch issues
        X_train_encoded_df.reset_index(drop=True, inplace=True)
        X_test_encoded_df.reset_index(drop=True, inplace=True)

        # Concatenate one-hot encoded features with numeric column 'Price'
        X_train_final = pd.concat([X_train_encoded_df, train_data[['Price']].reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([X_test_encoded_df, test_data[['Price']].reset_index(drop=True)], axis=1)

        # Convert categorical labels to numeric using factorize
        labels, uniques = pd.factorize(train_data[col])
        
        # Convert data to XGBoost DMatrix format (GPU-accelerated)
        dtrain = xgb.DMatrix(X_train_final, label=labels)
        dtest = xgb.DMatrix(X_test_final)

        # Train an XGBoost classifier using GPU
        model = xgb.train(
            params={
                'objective': 'multi:softmax',  # Multi-class classification
                'num_class': len(uniques),       # Number of categories (unique values)
                'tree_method': 'gpu_hist',       # Use GPU
                'predictor': 'gpu_predictor'
            },
            dtrain=dtrain,
            num_boost_round=100
        )

        # Predict missing values (numeric codes)
        predicted_values = model.predict(dtest)

        # Optionally convert numeric codes back to original labels
        predicted_labels = uniques[predicted_values.astype(int)]

        # Replace missing values
        dataset.loc[missing_mask, col] = predicted_labels


# 1. Identify the missing values for the numeric feature
missing_mask = dataset['Weight Capacity (kg)'].isna()

# Proceed only if there are missing values
if missing_mask.sum() > 0:
    # 2. Split the data:
    #    - Training data: rows where 'RealFeature' is not missing.
    #    - Data to impute: rows where 'RealFeature' is missing.
    train_data = dataset[~missing_mask].copy()
    test_data = dataset[missing_mask].copy()

    # 3. Define predictor columns.
    #    Typically, you want to use all available features except the one you are imputing
    #    and possibly also excluding the target to avoid data leakage.
    predictor_cols = [col for col in dataset.columns if col not in ['Weight Capacity (kg)', 'Price']]
    
    # 4. Prepare the predictors.
    #    If your predictors include categorical variables (which are already imputed),
    #    it is often a good idea to convert them into a numeric representation.
    #    One common approach is to use one-hot encoding.
    X_train = pd.get_dummies(train_data[predictor_cols])
    X_test  = pd.get_dummies(test_data[predictor_cols])
    
    # Ensure that both training and test predictors have the same columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # 5. Create the DMatrix objects for XGBoost.
    #    Here we use 'reg:squarederror' because our task is regression (predicting a real value).
    dtrain = xgb.DMatrix(X_train, label=train_data['Weight Capacity (kg)'])
    dtest  = xgb.DMatrix(X_test)
    
    # 6. Set up XGBoost parameters.
    #    Note: With XGBoost 2.0.0 or later, use tree_method='hist' and specify 'device': 'cuda' for GPU.
    params = {
        'objective': 'reg:squarederror',  # Regression objective
        'tree_method': 'hist',             # Use histogram-based algorithm
        'device': 'cuda'                   # Use GPU acceleration (if available)
    }
    
    # 7. Train the model.
    num_boost_round = 100  # Adjust the number of boosting rounds as needed
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
    
    # 8. Predict the missing values.
    predicted_values = model.predict(dtest)
    
    # 9. Replace the missing values in the dataset.
    dataset.loc[missing_mask, 'Weight Capacity (kg)'] = predicted_values
    
    
    print("Imputation of 'Weight Capacity (kg)' completed successfully.")
else:
    print("No missing values detected in 'Weight Capacity (kg)'.")


categorical_cols = ['Brand', 'Material', 'Size','Waterproof', 'Laptop Compartment', 'Style', 'Color']
for col in categorical_cols:
    df_test[col].fillna(df_test[col].mode()[0], inplace=True)



df_test['Weight Capacity (kg)'].fillna(df_test['Weight Capacity (kg)'].mean(), inplace=True)



dataset.isnull().sum()


df_test.isnull().sum()


features


dataset['Laptop Compartment'] = dataset['Laptop Compartment'].replace({'No': 0, 'Yes': 1}).astype(int)
df_test['Laptop Compartment'] = df_test['Laptop Compartment'].replace({'No': 0, 'Yes': 1}).astype(int)
dataset['Waterproof'] = dataset['Waterproof'].replace({'No': 0, 'Yes': 1}).astype(int)
df_test['Waterproof'] = df_test['Waterproof'].replace({'No': 0, 'Yes': 1}).astype(int)


abc=['Brand', 'Material', 'Size', 'Style', 'Color']
for i in abc:
    dataset = pd.get_dummies(dataset, columns=[i], dtype=int)
    df_test = pd.get_dummies(df_test, columns=[i], dtype=int)


dataset.info()


df_test.info()


Y=dataset['Price']


dataset.drop(columns=['Price'],inplace=True)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#------------------------------------------------------------------------------
X = dataset.copy()  
y = Y.copy()        
n_test = len(df_test)

# Number of CV folds
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#------------------------------------------------------------------------------
# Parameters for the models
#------------------------------------------------------------------------------

# LightGBM Parameters for GPU
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'device_type': 'gpu',
    'learning_rate': 0.001,
    'n_estimators': 10000,
    'early_stopping_rounds': 50,
    'random_state': 42
}



# XGBoost Parameters for GPU 
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.001,
    'n_estimators': 10000,
    'random_state': 42,
    'tree_method': 'gpu_hist'  
}





# For the validation ensemble, we will accumulate predictions for each CV fold.
ensemble_val_predictions = np.zeros(len(y))

#------------------------------------------------------------------------------
# Cross-validation loop
#------------------------------------------------------------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold + 1} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # -------------------------------
    # LightGBM Model Training
    # -------------------------------
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_val_pred = lgb_model.predict(X_val)
    fold_lgb_rmse = rmse(y_val, lgb_val_pred)
    print(f"LightGBM Fold {fold + 1} RMSE: {fold_lgb_rmse:.4f}")

    
    # -------------------------------
    # XGBoost Model Training
    # -------------------------------
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=50, 
                  verbose=100)
    xgb_val_pred = xgb_model.predict(X_val)
    fold_xgb_rmse = rmse(y_val, xgb_val_pred)
    print(f"XGBoost Fold {fold + 1} RMSE: {fold_xgb_rmse:.4f}")

    
    # -------------------------------
    # Ensemble: Average predictions from all models
    # -------------------------------
    fold_ensemble_pred = (lgb_val_pred +  xgb_val_pred ) / 2.0
    ensemble_val_predictions[val_idx] = fold_ensemble_pred

#------------------------------------------------------------------------------
# Compute final RMSE on the full training set for each individual model (using the last fold's model as an example)
#------------------------------------------------------------------------------
# Note: These "final" RMSE values are computed on the entire training set using the model
# trained on the last fold. For a better estimate, you might retrain on all data.
final_lgb_rmse = rmse(y, lgb_model.predict(X))

final_xgb_rmse = rmse(y, xgb_model.predict(X))


print("\n===== Final Model RMSE (using the last fold's model) =====")
print(f"LightGBM Final RMSE: {final_lgb_rmse:.4f}")

print(f"XGBoost Final RMSE:   {final_xgb_rmse:.4f}")


#------------------------------------------------------------------------------
# Evaluate Ensemble on the Validation Set (CV predictions)
#------------------------------------------------------------------------------
ensemble_val_rmse = rmse(y, ensemble_val_predictions)
print(f"\nEnsemble CV RMSE: {ensemble_val_rmse:.4f}")




# PREDICT ON df_test USING ALL MODELS (Single Full-Model Predictions)
# =============================================================================
# Make predictions for the test set using each trained model
lgb_test_pred = lgb_model.predict(df_test)

xgb_test_pred = xgb_model.predict(df_test)

# Ensemble prediction: simple averaging of  models' predictions
ensemble_test_pred = (
    lgb_test_pred +
    xgb_test_pred  ) / 2.0

# ensemble_test_pred now holds your final ensemble predictions for the test set.



sub=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


sub['Price']= lgb_test_pred



sub.to_csv('stacking_test_pred.csv',index=False)





