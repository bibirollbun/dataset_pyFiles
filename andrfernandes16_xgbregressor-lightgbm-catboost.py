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


SEED = 42

resting_heart_rate = 70
normal_human_body_temp = 37


# Load
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Encode the gender feature from categorical to numerical
train_df_encoded = pd.get_dummies(train_df, columns=['Sex'], drop_first=True, dtype = int)
test_df_encoded = pd.get_dummies(test_df, columns=['Sex'], drop_first=True, dtype = int)


# Lets add feature: BMI - Body Mass Index
train_df_encoded["BMI"] = train_df_encoded["Weight"] / ((train_df_encoded["Height"] / 100) ** 2)
test_df_encoded["BMI"] = test_df_encoded["Weight"] / ((test_df_encoded["Height"] / 100) ** 2)

# Lets add feature: CL - Cardio Load / Intensity
train_df_encoded["CL"] = (train_df_encoded["Heart_Rate"] - resting_heart_rate) * train_df_encoded["Duration"]
test_df_encoded["CL"] = (test_df_encoded["Heart_Rate"] - resting_heart_rate) * test_df_encoded["Duration"]

# Lets add feature: BTC - Body Temp Change from Normal
train_df_encoded["BTC"] = train_df_encoded["Body_Temp"] - normal_human_body_temp
test_df_encoded["BTC"] = test_df_encoded["Body_Temp"] - normal_human_body_temp

# Lets add feature: WHR - Weight-to-Height Ratio
train_df_encoded["WHR"] = train_df_encoded["Weight"] / train_df_encoded["Height"]
test_df_encoded["WHR"] = test_df_encoded["Weight"] / test_df_encoded["Height"]

train_df_encoded.head()


import seaborn as sns
import matplotlib.pyplot as plt


sns.histplot(data = train_df_encoded, x = "Age", kde = True)
plt.title("Age Distribution")


sns.histplot(data = train_df_encoded, x = "Sex_male", kde = False)
plt.title("Sex Distribution ")


sns.histplot(data = train_df_encoded, x = "BMI", kde = True)
plt.title("BMI Distribution")


sns.histplot(data = train_df_encoded, x = "CL", kde = True)
plt.title("Calority Load Distribution")


sns.histplot(data = train_df_encoded, x = "BTC", kde = True)
plt.title("Body Temp Change from Normal")


sns.histplot(data = train_df_encoded, x = "WHR", kde = True)
plt.title("Weight-to-Height Ratio")


# Lets get corralation matrix
corr_matrix = train_df_encoded.corr()
# Lets get correaltion of features in realtion to caloires
corr_matrix_RELATION_calories = train_df_encoded.corr()["Calories"].drop("Calories")

# Show correlation of all features with the target label (Anomal)
print(corr_matrix["Calories"].sort_values(ascending=False))


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot bar chart
corr_matrix_RELATION_calories.sort_values().plot(kind='barh', color='teal', ax=axes[0])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("Correlation with Calories")
axes[0].set_xlabel("Correlation Coefficient")
axes[0].set_ylabel("Features")

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title("Correlation Matrix")

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

# Data preparation
X_all_features = train_df_encoded.drop(columns=["id", "Calories"])
y_all_features = train_df_encoded["Calories"]

X_test_all_features = test_df_encoded.drop(columns=["id"])

# Data splitting
X_train, X_val, y_train, y_val = train_test_split(X_all_features, y_all_features, random_state=SEED,test_size=0.10, shuffle=True)


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import GridSearchCV


# Custom scorer for RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # avoid log of negative
    y_true = np.maximum(y_true, 0)  # extra safety
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Turn into a scorer for GridSearchCV (negated so lower is better)
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Parameters to tune
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}

lgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [20, 31, 50],
}

cat_param_grid = {
    'iterations': [50, 100],
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
}


# Define model
xgb_model_all_features = XGBRegressor(objective='reg:squarederror', random_state=SEED) # The "objective" is needed for regression task

# Define the GridSearch to perfome hyperparmenter tunning
xgb_grid_all_features = GridSearchCV(
    estimator=xgb_model_all_features,
    param_grid=xgb_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

xgb_grid_all_features.fit(X_train, y_train)


lgb_model_all_features = LGBMRegressor(random_state=SEED)

lgb_grid_all_features = GridSearchCV(
    estimator=lgb_model_all_features,
    param_grid=lgb_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

lgb_grid_all_features.fit(X_train, y_train)


cat_model_all_features = CatBoostRegressor(random_seed=SEED, verbose=0)

cat_grid_all_features = GridSearchCV(
    estimator=cat_model_all_features,
    param_grid=cat_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

cat_grid_all_features.fit(X_train, y_train)


# Data preparation
X_few_features = train_df_encoded.drop(columns=["id", "Calories", "Age", "BMI", "WHR", "Weight", "Sex_male", "Height"])
y_few_features = train_df_encoded["Calories"]

X_test_few_features = test_df_encoded.drop(columns=["id", "Age", "BMI", "WHR", "Weight", "Sex_male", "Height"])

# Data splitting
X_train, X_val, y_train, y_val = train_test_split(X_few_features, y_few_features, random_state=SEED,test_size=0.10, shuffle=True)


# Define model
xgb_model_few_features = XGBRegressor(objective='reg:squarederror', random_state=SEED) # The "objective" is needed for regression task

# Define the GridSearch to perfome hyperparmenter tunning
xgb_grid_few_features = GridSearchCV(
    estimator=xgb_model_few_features,
    param_grid=xgb_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

xgb_grid_few_features.fit(X_train, y_train)


lgb_model_few_features = LGBMRegressor(random_state=SEED)

lgb_grid_few_features = GridSearchCV(
    estimator=lgb_model_few_features,
    param_grid=lgb_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

lgb_grid_few_features.fit(X_train, y_train)


cat_model_few_features = CatBoostRegressor(random_seed=SEED, verbose=0)

cat_grid_few_features = GridSearchCV(
    estimator=cat_model_few_features,
    param_grid=cat_param_grid,
    cv=5,
    scoring=rmsle_scorer,
    n_jobs=-1,
    verbose=0
)

cat_grid_few_features.fit(X_train, y_train)


print("Best XGBoost Params - All Features:", xgb_grid_all_features.best_params_)
print("Best RMSLE:", -xgb_grid_all_features.best_score_) # negate because we used greater_is_better=False

print("\nBest XGBoost Params - Fewer Features:", xgb_grid_few_features.best_params_)
print("Best RMSLE:", -xgb_grid_few_features.best_score_)

print("\n-------------------------\n")

print("Best LightGBM Params - All Features:", lgb_grid_all_features.best_params_)
print("Best RMSLE:", -lgb_grid_all_features.best_score_)

print("\nBest LightGBM Params - Fewer Features:", lgb_grid_few_features.best_params_)
print("Best RMSLE:", -lgb_grid_few_features.best_score_)

print("\n-------------------------\n")

print("Best CatBoost Params - All Features:", cat_grid_all_features.best_params_)
print("Best RMSLE:", -cat_grid_all_features.best_score_)

print("\nBest CatBoost Params - Fewer Features:", cat_grid_few_features.best_params_)
print("Best RMSLE:", -cat_grid_few_features.best_score_)


print("(XGBoost Params - All Features) - (Best LightGBM Params - All Features) = ", (-xgb_grid_all_features.best_score_) - (-lgb_grid_all_features.best_score_))


# Make predictions
final_pred = xgb_grid_all_features.predict(X_test_all_features)
final_pred = np.maximum(0, final_pred)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': final_pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission

