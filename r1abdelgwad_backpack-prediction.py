import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.preprocessing import TargetEncoder
from xgboost import XGBRegressor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv") 
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train = pd.concat([train_data,train_data_ex], axis=0, ignore_index=True)


train.head()


test_data.head()


train.shape


train.isnull().sum()


test_data.isnull().sum()


train_data.duplicated().sum()


test_data.duplicated().sum()


train_data["Material"].value_counts()


plt.figure(figsize=(6,6))
sns.heatmap(train_data.isnull(),cmap="viridis",cbar = False , yticklabels = False )
plt.show()


num_cols = test_data.select_dtypes(include=['number']).columns

imputation_value = train[num_cols].median()

train[num_cols] = train[num_cols].fillna(imputation_value)
test_data[num_cols] = test_data[num_cols].fillna(imputation_value)


# Impute Missing Values in Object Columns with 'None'

obj_cols = train.select_dtypes(include=['object']).columns

train[obj_cols] = train[obj_cols].fillna('None')
test_data[obj_cols] = test_data[obj_cols].fillna('None')


train.isnull().sum()


test_data.isnull().sum()


TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test_data.columns.tolist()

for col in features:
    TE.fit(train[col], train['Price'])
    train[col] = TE.transform(train[col])
    test_data[col] = TE.transform(test_data[col])


display( test_data.dtypes)


#  train dataset
X_train = train.drop(columns=["Price"])  # Features
y_train = train["Price"]  # Target variable

#  test dataset 
X_test = test_data


X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",  # Suitable for regression tasks
    n_estimators=500,  # Number of trees
    learning_rate=0.05,  # Step size shrinkage
    max_depth=6,  # Maximum depth of a tree
    subsample=0.8,  # Fraction of samples used per tree
    colsample_bytree=0.8,  # Fraction of features used per tree
    random_state=42
)

# Train the model
xgb_model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)

# Predictions
y_pred_val = xgb_model.predict(X_val)
y_pred_test = xgb_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_val, y_pred_val)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred_val)

print(f"Validation MAE: {mae:.4f}")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R² Score: {r2:.4f}")

# Save predictions
test_data["Predicted_Price"] = y_pred_test
test_data.to_csv("test_predictions.csv", index=False)


# Adjusting the ID to start from 300000
submission1 = pd.DataFrame({'id': test_data.index + 300000, 'Price': y_pred_test})

# Save to CSV
submission1.to_csv('submission1.csv', index=False)

# Display the first few rows
display(submission1.head())




