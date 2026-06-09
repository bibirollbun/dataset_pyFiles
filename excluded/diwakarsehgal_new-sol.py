import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression



# Load data
data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")  # Replace with actual file path

# Process Date Column
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.weekday
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)


data.drop(columns=['date'], inplace=True)  # Drop original date column


# Encode Categorical Features
for col in ['country', 'store', 'product']:
    data[col] = data[col].astype(str)
    data[col + '_freq'] = data[col].map(data[col].value_counts() / len(data))  # Frequency Encoding
    data[col] = LabelEncoder().fit_transform(data[col])  # Label Encoding

# Split into training and missing data
train_data = data[data['num_sold'].notna()]
missing_data = data[data['num_sold'].isna()].drop(columns=['num_sold'])


# Train-Test Split
X = train_data.drop(columns=['num_sold'])
y = train_data['num_sold']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train LightGBM Model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_estimators': 500,
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)], categorical_feature=['country', 'store', 'product'])


# Predict missing num_sold values
missing_data['num_sold'] = model.predict(missing_data)
data.loc[data['num_sold'].isna(), 'num_sold'] = missing_data['num_sold']

# Train Final Model on Full Data
X_final = data.drop(columns=['num_sold'])
y_final = data['num_sold']
# Define model parameters
lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9}
xgb_params = {'objective': 'reg:squarederror', 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100}

# Initialize models
lgb_model = lgb.LGBMRegressor(**lgb_params)
xgb_model = xgb.XGBRegressor(**xgb_params)

# Train models
lgb_model.fit(X_final, y_final, categorical_feature=['country', 'store', 'product'])
xgb_model.fit(X_final, y_final)

# Make predictions using the base models on the validation set
lgb_valid_preds = lgb_model.predict(X_final)
xgb_valid_preds = xgb_model.predict(X_final)

# Stack the predictions from the base models (stacked as features for meta-model)
stacked_preds = np.column_stack((lgb_valid_preds, xgb_valid_preds))

# Train the meta-model (Linear Regression in this case) on the stacked predictions
meta_model = LinearRegression()
meta_model.fit(stacked_preds, y_final)


# Predict on Test Data
test_data = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")  # Replace with actual test file
test_data['date'] = pd.to_datetime(test_data['date'])
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['day_of_week'] = test_data['date'].dt.weekday
test_data['is_weekend'] = (test_data['day_of_week'] >= 5).astype(int)
test_data.drop(columns=['date'], inplace=True)

for col in ['country', 'store', 'product']:
    test_data[col] = test_data[col].astype(str)
    test_data[col + '_freq'] = test_data[col].map(data[col].value_counts() / len(data))
    test_data[col] = LabelEncoder().fit_transform(test_data[col])

# Make predictions using each model
lgb_preds = lgb_model.predict(test_data)
xgb_preds = xgb_model.predict(test_data)

# Stack test predictions
stacked_test_preds = np.column_stack((lgb_preds, xgb_preds))

# Meta-model makes the final prediction
final_preds = meta_model.predict(stacked_test_preds)


# Save Submission
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


print(submission.head())




