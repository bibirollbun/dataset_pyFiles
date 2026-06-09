#!pip install xgbtune


# Import necessary modules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, validation_curve, cross_val_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import itertools

# Uncomment for hyperparameter tuning
# from xgbtune import tune_xgb_model 

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

# Combine train and train extra data sets into one
df = pd.concat([train, train_extra], axis=0, ignore_index=True)

# Rows and Cols
display(df.shape)
display(test.shape)


# Descriptive stats
df.describe()

# Columns and 
df.info()
test.info()

# Data types
display(df.dtypes)

#Summarize
df.head()

# Check for null/missing values
display(df.dtypes, df.isnull().sum())


# Plot price distributions
plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=50, kde=True, color="green")
plt.title("Price Distribution")
plt.show()

# Compare distributions between train and test
sns.kdeplot(df['Weight Capacity (kg)'], label="Train", shade=True)
sns.kdeplot(test['Weight Capacity (kg)'], label="Test", shade=True)


# Drop the target Price column
X = df.drop(columns=['Price'])
y = df['Price']
display(X)


# Feature engineering function to determine better feature relationships/correlations.
def feature_engineering_xgb(X_train, y_train, test, cat_features, top_corr_features=5, max_new_features=10, k_best=10):
    """
    Enhances dataset by:
    1. Handling missing values.
    2. Encoding categorical features using CatBoostEncoder.
    3. Identifying top correlated numerical features with Price (train only).
    4. Generating limited interaction features (train & test).
    5. Removing duplicate features.
    6. Selecting the best features using SelectKBest.

    Parameters:
    - X_train: Training Feature DataFrame
    - y_train: Training target variable (Price)
    - test: Test Feature DataFrame (no y (Price) available)
    - cat_features: List of categorical feature names
    - top_corr_features: Number of top correlated numerical features to use in interactions
    - max_new_features: Maximum number of new features to generate
    - k_best: Number of top features to select with SelectKBest

    Returns:
    - Transformed `X_train` and `test` DataFrames with engineered features.
    - List of selected features from training data.
    """
   
    # Copy data to prevent modification
    X_train = X_train.copy()
    test = test.copy()

    print("\nğŸ› ï¸� Handling Missing Values...")

    # Show NaN counts before handling
    print("\nğŸ”� NaN Counts Before Handling:")
    print("Training Data:\n", X_train.isna().sum().sort_values(ascending=False).head(10))
    print("Test Data:\n", test.isna().sum().sort_values(ascending=False).head(10))

    # Fill categorical NaN with "Unknown" (Train & Test)
    X_train[cat_features] = X_train[cat_features].fillna("Unknown")
    test[cat_features] = test[cat_features].fillna("Unknown")
    
    # Log Transformation (Train & Test)
    X_train['Weight Capacity (kg)'] = X_train['Weight Capacity (kg)'].fillna(X_train['Weight Capacity (kg)'].mean())
    X_train['Compartments'] = X_train['Compartments'].fillna(X_train['Compartments'].mean())
    
    test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())
    test['Compartments'] = test['Compartments'].fillna(test['Compartments'].mean())

    # Show NaN counts after handling
    print("\nâœ… NaN Counts After Handling:")
    print("Training Data:\n", X_train.isna().sum().sort_values(ascending=False).head(10))
    print("Test Data:\n", test.isna().sum().sort_values(ascending=False).head(10))
    print("âœ… Filled missing values in categorical and numerical features.")

    # Step 2: Encode Categorical Features Using CatBoostEncoder
    print("\nğŸ”¢ Applying CatBoost Encoding to Categorical Features...")
    encoder = CatBoostEncoder(cols=cat_features, random_state=42)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    test_encoded = encoder.transform(test)

    print(f"âœ… CatBoost Encoded {len(cat_features)} categorical features.")

    # Step 3: Compute Correlations (Train Only)
    corr_matrix = X_train_encoded.corrwith(y_train).abs().sort_values(ascending=False)
    print("\nğŸ”� Top 10 Features Most Correlated with Price:\n", corr_matrix.head(10))

    # Select top correlated numerical features for interactions (Train Only)
    top_features = corr_matrix.head(top_corr_features).index.tolist()
    numeric_cols = X_train_encoded[top_features].select_dtypes(include=[np.number]).columns.tolist()

   # Step 4: Generate LIMITED Interaction Features (Train & Test)
    print("\nâš™ï¸� Generating Limited Interaction Features...")
    new_features = []
    interaction_count = 0  # Track number of generated features

    for col1, col2 in itertools.combinations(numeric_cols, 2):
        if interaction_count >= max_new_features:
            break  # Stop generating new features if max_new_features is reached

        X_train_encoded[f"{col1}_x_{col2}"] = X_train_encoded[col1] * X_train_encoded[col2]
        X_train_encoded[f"{col1}_div_{col2}"] = X_train_encoded[col1] / (X_train_encoded[col2] + 1e-6)  # Avoid division by zero
        test_encoded[f"{col1}_x_{col2}"] = test_encoded[col1] * test_encoded[col2]
        test_encoded[f"{col1}_div_{col2}"] = test_encoded[col1] / (test_encoded[col2] + 1e-6)

        new_features.extend([f"{col1}_x_{col2}", f"{col1}_div_{col2}"])
        interaction_count += 2

    print(f"âœ… Created {interaction_count} new interaction features.")

    # Step 5: Remove Duplicate Features (Train & Test)
    print("\nğŸ—‘ï¸� Removing Duplicate Features...")
    X_train_encoded = X_train_encoded.loc[:, ~X_train_encoded.columns.duplicated()]
    test_encoded = test_encoded.loc[:, ~test_encoded.columns.duplicated()]
    print(f"âœ… Removed duplicate features. Remaining Train Features: {X_train_encoded.shape[1]}, Test Features: {test_encoded.shape[1]}")

    # Step 6: Feature Selection Using SelectKBest
    print("\nğŸ”� Selecting Best Features Using SelectKBest...")
    selector = SelectKBest(score_func=f_regression, k=k_best)
    X_train_selected = selector.fit_transform(X_train_encoded, y_train)
    test_selected = selector.transform(test_encoded)
    selected_features = X_train_encoded.columns[selector.get_support()].tolist()
    print(f"\nğŸ�¯ Selected Top {len(selected_features)} Features Based on SelectKBest:")
    print(selected_features, "\n\n")

    # Convert to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_encoded.index)
    test_selected = pd.DataFrame(test_selected, columns=selected_features, index=test_encoded.index)

    return X_train_selected, test_selected, selected_features


# Function to find best hyperparameters to tune the XGBRegressor.
def randomized_grid_search(X_train, y_train):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    params = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.25, 0.1],
        'max_depth': [1, 3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.5, 0.8, 0.9],
        'reg_lambda': [0, 1, 5, 10],
        'min_split_loss': [0, 0.1, 0.5, 1, 3, 5, 10],
        'reg_alpha': [0, 0.1, 1]
    }
    
    grid_search = RandomizedSearchCV(xgb, params, cv=5, random_state=42,
                                     scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    return grid_search.best_params_, grid_search.best_score_

# Do a random grid search to understand the impact of the hyperparameters.
# Commented out since this takes LOTS of time. 
# randomized_grid_search(X_train, y_train)


# Explore xgbtune to determine the best hyperparameters
# import xgboost as xgb
# best_params = {'eval_metric': 'rmse', 'tree_method': 'hist'}

# params, round_count = tune_xgb_model(best_params, X_train_transformed, y)

# new_Train = xgb.DMatrix(X_train_transformed, label=y)
# xgb_model = xgb.train(params, new_Train, num_boost_round=round_count)

# Final model prediction
# final_Test = xgb.DMatrix(X_test_transformed)
# final_test_pred = xgb_model.predict(final_Test)
# display(pd.DataFrame({'id': test.index, 'Price': final_test_pred}))

# RMSE:38.3 {'eval_metric': 'rmse', 'tree_method': 'hist', 'max_depth': 8, 'min_child_weight': 2, 'gamma': 0.0, 'subsample': 0.95, 'colsample_bytree': 1.0, 'alpha': 0.1, 'lambda': 1, 'seed': 0}
# RMSE: 38.4 {'eval_metric': 'rmse', 'tree_method': 'hist', 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 0.95, 'colsample_bytree': 1.0, 'alpha': 1, 'lambda': 1, 'seed': 0}
# RMSE: 38.8 {'eval_metric': 'rmse', 'tree_method': 'hist', 'max_depth': 8, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 1.0, 'colsample_bytree': 1.0, 'alpha': 0, 'lambda': 1, 'seed': 0}


cat_features = ['Size', 'Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
X_train_transformed, X_test_transformed, selected_features = feature_engineering_xgb(X, y, test, cat_features)

X_train, X_test, y_train, y_test = train_test_split(X_train_transformed, y, test_size=0.2, random_state=42, shuffle=True)

best_params = {'eval_metric': 'rmsle', 'tree_method': 'hist', 'max_depth': 8, 'min_child_weight': 2,  
               'subsample': 0.8, 'colsample_bytree': 1.0, 'alpha': 0, 'lambda': 1, 'seed': 0, 'random_state': 42}

# Run XGB given the tuned parameters
xgb_boost = XGBRegressor(early_stopping_rounds=100, **best_params)

xgb_boost.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
print(f"Best XGBoost iteration using RMSE: {xgb_boost.best_iteration} \n")

y_pred = xgb_boost.predict(X_test)

# Compare MAE & RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"ğŸ“Š Mean Absolute Error (MAE): {mae:.4f}")
print(f"ğŸ“Š Root Mean Squared Error (RMSE): {rmse:.4f}")


# Try CatBoost for comparison
#cbr = CatBoostRegressor(iterations=500,
#                        depth=5,
#                        learning_rate=0.3,
#                        loss_function='RMSE',
#                        random_state=42)

#cbr.fit(X_train_encoded, y_train, eval_set=[(X_test_encoded, y_test)], verbose=200)
#y_pred = cbr.predict(X_test_encoded)
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print(f"CatBoost Root Mean Squared Error (RMSE): {rmse}")


# Handle the prediction and submission file generation
test_pred = xgb_boost.predict(X_test_transformed)

submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission = pd.DataFrame({'id': submission.id, 'Price': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)


# Remove old file(s)
# import os
# os.remove('/kaggle/working/submission.csv')

