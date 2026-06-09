import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df["Sex"] = train_df["Sex"].map({"male":0, "female":1})
test_df["Sex"] = test_df["Sex"].map({"male":0, "female":1})


from itertools import combinations
from tqdm import tqdm

encoded_columns = []
encode_columns = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
pair_size = [2,3]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = "_".join(cols)

        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        train_df[new_col_name] = train_df[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')

        encoded_columns.append(new_col_name)


train_df.head()


X = train_df.drop(["id", "Calories"], axis=1)
y = train_df["Calories"]
X_test = test_df.drop(["id"], axis = 1)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

catboost_model = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)

# Train the model
catboost_model.fit(X_train, y_train)

# Make predictions on the validation set
catboost_pred_val = catboost_model.predict(X_val)

# Evaluate using RMSLE
catboost_rmsle = np.sqrt(mean_squared_log_error(y_val, catboost_pred_val))
print(f"CatBoost Validation RMSLE: {catboost_rmsle:.4f}")


catboost_pred_test = catboost_model.predict(X_test)
submission = pd.DataFrame({"id": test_df["id"], "Calories": catboost_pred_test})
submission.to_csv("submission.csv", index=False)

