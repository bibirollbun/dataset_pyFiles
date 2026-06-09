import pandas as pd

train_df = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test_df = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')



X = train_df.drop(columns=["target"])
y = train_df["target"]
test_features = test_df.drop(columns=["id"])



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_val)
linear_r2 = r2_score(y_val, linear_preds)
print("Linear Regression R^2:", linear_r2)



from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state=42, n_estimators=900, max_depth=10)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_r2 = r2_score(y_val, xgb_preds)
print("XGBoost Regression R^2:", xgb_r2)


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)
params = {
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'device': 'cpu',  
    'verbose': -1
}
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=2000,

)

# Predict and evaluate
preds = model.predict(X_val)
r2 = r2_score(y_val, preds)
print("LightGBM R^2 (Accuracy):", r2)


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the data
train = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')

# Explore the data
print(train.head())
print(train.info())

# Features and target variable
X = train.drop(columns=['target'])  # Drop the target column
y = train['target']                # Target variable

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using R² score
r2 = r2_score(y_val, y_pred)
print(f"R² Score: {r2}")





test_preds = model.predict(test_features)



submission = pd.DataFrame({
    "id": test_df["id"],
    "target": test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")





