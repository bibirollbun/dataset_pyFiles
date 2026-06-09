import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


print(train.head())
print(train.describe())
print(train.info())


# Drop ID column
#test_ids = test["id"]
#train.drop("id", axis=1, inplace=True)
#test.drop("id", axis=1, inplace=True)

# Target and Features
y = train["Listening_Time_minutes"]
X = train.drop("Listening_Time_minutes", axis=1)

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include="number").columns.tolist()

# Categorical preprocessing: most frequent imputation + label encoding
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])

# Impute missing values in numeric columns
num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
test[numerical_cols] = num_imputer.transform(test[numerical_cols])

# Scale numeric features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("Validation MSE:", mse)


model.fit(X, y)
test_preds = model.predict(test)

submission["listening_time"] = test_preds
submission.to_csv("submission.csv", index=False)

print("Submission saved as submission.csv")


