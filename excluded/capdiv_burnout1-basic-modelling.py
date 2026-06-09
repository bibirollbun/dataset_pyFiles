import pandas as pd
import numpy as np


train_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


print(train_df.shape,test_df.shape,val_df.shape)


train_df.head()


train_df.nunique()


import pandas as pd

# Assuming train_df and val_df are already defined
train_df = pd.concat([train_df, val_df], ignore_index=True)



print(train_df.shape,test_df.shape,val_df.shape)


train_df.nunique()


train_df.dtypes


train_df.isna().sum()


train_df=train_df.fillna(0)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import time

# Target column
target_col = 'Lap_Time_Seconds'

# Drop rows with missing target
train_df = train_df.dropna(subset=[target_col])

# Split features and target
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features (CatBoost handles them natively)
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Optional: Fill missing values
X_train = X_train.fillna(-999)
X_val = X_val.fillna(-999)

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

# Record training start time
start_time = time.time()

# Train model
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Record training end time
end_time = time.time()

# Predict and evaluate RMSE
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
training_time = end_time - start_time

# Print results
print(f"\nValidation RMSE: {rmse:.4f}")
print(f"Training Time: {training_time:.2f} seconds")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import time

# Target column
target_col = 'Lap_Time_Seconds'

# Drop rows with missing target
train_df = train_df.dropna(subset=[target_col])

# Treat ID-like numeric columns as categorical by converting them to string
id_like_cats = ['Rider_ID', 'Rider', 'Team', 'Bike']
for col in id_like_cats:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

# Split features and target
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features for CatBoost
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Fill missing values (CatBoost can handle missing values too, but filling ensures no surprises)
X_train = X_train.fillna(-999)
X_val = X_val.fillna(-999)

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

# Record training start time
start_time = time.time()

# Train model
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Record training end time
end_time = time.time()

# Predict and evaluate RMSE
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
training_time = end_time - start_time

# Print results
print(f"\nValidation RMSE: {rmse:.4f}")
print(f"Training Time: {training_time:.2f} seconds")






