import numpy as np # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from itertools import combinations
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


file_path = '../input/playground-series-s5e9/train.csv'
test_path = '../input/playground-series-s5e9/test.csv'

data = pd.read_csv(file_path) 
test_data = pd.read_csv(test_path) 
sample_submission = pd.read_csv('../input/playground-series-s5e9/sample_submission.csv')

# Create target object and call it y
y = data.BeatsPerMinute
# Create X
original_features = [
    'RhythmScore',
    'AudioLoudness',
    'VocalContent', 
    'AcousticQuality',
    'InstrumentalScore',
    'LivePerformanceLikelihood', 
    'MoodScore', 
    'TrackDurationMs', 
    'Energy'
]

def preprocessing(df):
    """
    Engineers new features by creating log, product, and division combinations
    of the original numerical columns. This version is optimized to avoid
    PerformanceWarning by creating new columns in a dictionary and then
    concatenating them all at once.
    """
    # Use .copy() to prevent SettingWithCopyWarning
    df_processed = df.copy()
    
    # We will only create combinations from the original features
    numerical_cols = original_features
    
    # Dictionary to hold all new columns before adding them to the DataFrame
    new_cols_dict = {}

    combination_orders = [1, 2, 3, 4]
    for order in combination_orders:
        if order == 1:
            # Create log-transformed features for single columns
            for col in numerical_cols:
                new_cols_dict[f"{col}_log"] = np.log1p(df_processed[col])
        else:
            # Create product and division features for combinations of columns
            for cols_tuple in combinations(numerical_cols, order):
                # --- Product Features ---
                product_val = 1
                for col in cols_tuple:
                    product_val *= df_processed[col]
                
                product_feature_name = f"{'_m_'.join(cols_tuple)}"
                new_cols_dict[product_feature_name] = np.log1p(product_val)

                # --- Division Features ---
                # Example: for (A, B, C), this creates A / (B*C)
                if order >= 2:
                    numerator_col = cols_tuple[0]
                    denominator_product = 1
                    denominator_feature_name_parts = []
                    
                    for col_idx in range(1, order):
                        denominator_product *= df_processed[cols_tuple[col_idx]]
                        denominator_feature_name_parts.append(cols_tuple[col_idx])
                    
                    denominator = denominator_product + 1e-6
                    
                    division_feature_name = f"{numerator_col}_d_{'_d_'.join(denominator_feature_name_parts)}"
                    new_cols_dict[division_feature_name] = np.log1p(df_processed[numerator_col] / denominator)

    # Convert the dictionary of new features to a DataFrame
    new_features_df = pd.DataFrame(new_cols_dict)
    
    # Concatenate the new features with the original DataFrame in one go
    df_final = pd.concat([df_processed, new_features_df], axis=1)

    return df_final

# --- 4. Apply Preprocessing and Create Final Datasets ---
print("Starting feature engineering...")
processed_data = preprocessing(data)
processed_test_data = preprocessing(test_data)
print("Feature engineering complete.")

# Define the final list of features from the processed training data,
# excluding the target variable and any ID columns.
non_feature_cols = ['id', 'BeatsPerMinute']
final_features = [col for col in processed_data.columns if col not in non_feature_cols]

# Create X (features for training) and test_X (features for testing)
# Ensure both dataframes use the same set of columns
X = processed_data[final_features].copy()
test_X = processed_test_data[final_features].copy()

print(f"\nOriginal number of features: {len(original_features)}")
print(f"Number of features after engineering: {len(final_features)}")

# --- 5. Display Head of the Processed Data ---
print("\nFirst 5 rows of the data with new features:")
X.head()


# plt.subplots(figsize=(15, 3 * math.ceil(len(original_features))))
# for i, col in enumerate(X.columns):
#     plt.subplot(math.ceil(len(original_features)), 3, i + 1)
#     x = X.sample(1000)
#     sb.scatterplot(x=col, y=y, data=x)
# plt.tight_layout()
# plt.show()


best_params = {
    'learning_rate': 0.025,
    'max_depth': 5,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.05, random_state=1)

# Define the XGBoost model
dtrain = xgb.DMatrix(train_X, label=train_y, enable_categorical=True)
dval = xgb.DMatrix(val_X, label=val_y, enable_categorical=True)

evals_result = {}
xgb_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=1500,
    evals=[(dtrain, "train"), (dval, "valid")], 
    early_stopping_rounds=100,
    evals_result=evals_result,
    verbose_eval=10
)


val_predictions = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration))
rmsd = np.sqrt(mean_squared_error(val_y, val_predictions))
print(f"\nValidation RMSD for XGBoost Model: {rmsd:,.4f}")


dtest = xgb.DMatrix(test_X, enable_categorical=True) 
preds = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration))

# Use the original test_data DataFrame to get the full list of test IDs (750,000 rows)
all_test_preds_df = pd.DataFrame({
    "id": test_data['id'],
    "Listening_Time_minutes": preds 
})

# Merge with sample_submission to get only the required 250,000 IDs
submission_df = sample_submission[['id']].merge(all_test_preds_df, on='id', how='left')

# Save the submission file
submission_df.to_csv("submission_ensemble.csv", index=False)

