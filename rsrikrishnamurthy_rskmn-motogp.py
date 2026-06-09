import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



train = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
sample_submission = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv")

print(train.head())
print(train.info())
print(train.columns)



print(sample_submission.columns)


y = train['Lap_Time_Seconds']


# Drop unnecessary columns
drop_cols = ['Lap_Time_Seconds', 'Unique ID', 'Rider_ID', 'rider_name', 'team_name', 'bike_name', 'shortname', 'circuit_name']
X = train.drop(columns=drop_cols, errors='ignore')
X_test = test.drop(columns=drop_cols, errors='ignore')

# Combine train+test for consistent encoding
X_all = pd.concat([X, X_test], axis=0)

# One-hot encode categoricals
X_all = pd.get_dummies(X_all)

# Split back
X = X_all.iloc[:len(train), :]
X_test = X_all.iloc[len(train):, :]


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM dataset format
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# LightGBM parameters (fast + good accuracy)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_jobs': -1
}

# Train the model (10x faster than RF)
model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=100)

# Predict
test_preds = model.predict(X_test)

# Submission
submission = pd.DataFrame({
    "Unique ID": test["Unique ID"],
    "Lap_Time_Seconds": test_preds
})
submission.to_csv("/kaggle/working/solution.csv", index=False)
print("✅ Submission saved as solution.csv")



print(submission.isnull().sum())



print(len(submission), len(test))  # Both must be equal



submission = submission.sort_values("Unique ID").reset_index(drop=True)



# Predict on test set
test_preds = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "Unique ID": test["Unique ID"],
    "Lap_Time_Seconds": test_preds
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission saved as solution.csv")



from sklearn.preprocessing import LabelEncoder

X = train.drop(columns=['Lap_Time_Seconds', 'Unique ID'])
y = train['Lap_Time_Seconds']

# Label encode all object (string) columns
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
from sklearn.metrics import mean_squared_error
import numpy as np
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)





