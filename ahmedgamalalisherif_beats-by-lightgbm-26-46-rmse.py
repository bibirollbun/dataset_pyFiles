import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


# Show the first 5 column 
train_df.head(5)


len(train_df) , len(test_df)


# Show the information of train/test data 
train_df.info()


test_df.info()


# Check for missing values 
train_df.isnull().sum()


test_df.isnull().sum()


train_df.describe()


# Drop id and target column
df_features  = train_df.drop(columns=["id", "BeatsPerMinute"])


# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Distribution of Features
df_features.hist(figsize=(12,10), bins=20, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.show()


# Imports 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor


# Define Features and Target
X = train_df.drop(columns=["id", "BeatsPerMinute"])  
y = train_df["BeatsPerMinute"]

X_test = test_df.drop(columns=["id"])   # drop id only


# Train/Validation Split 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Preprocessing 
numeric_features = X.columns  
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)


from sklearn.metrics import mean_squared_error

lgbm_params = {
    "n_estimators": 1000,       
    "learning_rate": 0.05,      
    "num_leaves": 31,            
    "max_depth": -1,            
    "subsample": 0.8,           
    "colsample_bytree": 0.8,    
    "reg_alpha": 0.1,           
    "reg_lambda": 0.1,          
    "random_state": 42,
    "n_jobs": -1
}

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(**lgbm_params))
])

#  Train on Train-Split 
model.fit(X_train, y_train)

# Validate on Validation-Split 
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Retrain on Full Training Data 
model.fit(X, y)

# Predict on Test Set 
preds = model.predict(X_test)


# --- Submission ---
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Fill predictions directly into the target column
submission.loc[:, "BeatsPerMinute"] = model.predict(X_test)

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("ðŸŽµ Submission file saved as submission.csv")

