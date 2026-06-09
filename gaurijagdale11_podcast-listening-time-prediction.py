# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_data.head(10)


train_data.info()


train_data.describe()


train_data.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train_data["Listening_Time_minutes"], bins=50, kde=True)
plt.title("Distribution of Listening Time")
plt.show()



num_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads'
]

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=col, y="Listening_Time_minutes", data=train_data)
    plt.title(f"{col} vs Listening Time")
    plt.show()


corr = train_data.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


train_data.drop(columns=["id", "Podcast_Name", "Episode_Title", "Publication_Time"], inplace=True)


# Fill numeric columns with median
numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_features:
    train_data[col] = train_data[col].fillna(train_data[col].median())


# Fill categorical columns with mode
categorical_features = train_data.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])


train_data.isnull().sum()


# ðŸ”„ Encode Categorical Features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le


train_data.head()


# ðŸ§¾ Separate features and target
X = train_data.drop("Listening_Time_minutes", axis=1)
y = train_data["Listening_Time_minutes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("âœ… Preprocessing complete")
print("Train shape:", X_train.shape)
print("test shape:", X_test.shape)


import xgboost as xgb
from sklearn.metrics import mean_squared_error


 #Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=200,       # number of trees
    learning_rate=0.05,     # step size shrinkage
    max_depth=6,            # depth of each tree
    subsample=0.8,          # % of rows used per tree
    colsample_bytree=0.8,   # % of features used per tree
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)



y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", round(rmse, 4))


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
# --- Step 1: Define Hyperparameters for Grid Search ---
params = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8],  # Optional: can add more
    'colsample_bytree': [0.8],  # Optional
    'random_state': [42]
}

# --- Step 2: Setup Grid Search with XGBRegressor ---
model = xgb.XGBRegressor()
grid = GridSearchCV(estimator=model,
                    param_grid=params,
                    scoring='neg_root_mean_squared_error',
                    cv=3,
                    verbose=1,
                    n_jobs=-1)

# --- Step 3: Fit the Grid Search ---
grid.fit(X_train, y_train)

print("âœ… Best Parameters:", grid.best_params_)
print("ðŸ“‰ Best RMSE (CV):", -grid.best_score_)

# --- Step 4: Evaluate on Validation Set ---
best_model = grid.best_estimator_
test_preds = best_model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
print("ðŸ“Š Validation RMSE with best model:", test_rmse)


 #Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=300,       # number of trees
    learning_rate=0.1,     # step size shrinkage
    max_depth=8,            # depth of each tree
    subsample=0.8,          # % of rows used per tree
    colsample_bytree=0.8,   # % of features used per tree
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", round(rmse, 4))


#Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=565,       # number of trees
    learning_rate=0.04222221,     # step size shrinkage
    max_depth=14,            # depth of each tree
    subsample=0.8,          # % of rows used per tree
    colsample_bytree=0.8,   # % of features used per tree
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", round(rmse, 4))


import lightgbm as lgb
# LightGBM Dataset format (optional but better performance)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'verbose': -1,
    'random_state': 42
}


# Train the model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],  # Validation set is used for early stopping
    #early_stopping_rounds=50,  # Stop after 50 rounds without improvement
    #verbose_eval=100  # Print progress every 100 rounds
)



# Predict on test set
test_preds = model.predict(X_test, num_iteration=model.best_iteration)
test_rmse = mean_squared_error(y_test, test_preds, squared=False)
print("âœ… LightGBM Test RMSE:", test_rmse)




