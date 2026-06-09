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


og=pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')


df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


df.info()


og.info()


og=og.drop(columns='User_ID')


og['Gender'] = og['Gender'].map({'female': 0, 'male': 1})


og.rename(columns={'Gender': 'Sex'}, inplace=True)



og.head()


df=df.drop(columns='id')


df.head()


import pandas as pd

# assuming df is your DataFrame
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})



df.head()


df = pd.concat([df, og], axis=0)



df.shape


df=df.drop_duplicates()


df.shape


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Sex', data=df)
plt.title("Sex Distribution")
plt.show()



sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()



# sns.scatterplot(x='Body_Temp', y='Heart_Rate', hue='Sex', data=df)
# plt.title("Heart Rate vs. Body Temperature")
# plt.show()



# sns.scatterplot(x='Duration', y='Calories', hue='Sex', data=df)
# plt.title("Calories Burned vs. Duration")
# plt.show()



# sns.boxplot(x='Sex', y='Calories', data=df)
# plt.title("Calories Burned by Sex")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.scatterplot(x='Body_Temp', y='Calories', data=df, hue='Sex')
# plt.title("Body Temperature vs. Calories")
# plt.xlabel("Body Temperature")
# plt.ylabel("Calories Burned")
# plt.show()



# sns.scatterplot(x='Heart_Rate', y='Calories', data=df, hue='Sex')
# plt.title("Heart Rate vs. Calories")
# plt.xlabel("Heart Rate")
# plt.ylabel("Calories Burned")
# plt.show()



df.head()


df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2



df.head()


df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young', 'Adult', 'Mid-age', 'Senior'])



df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young', 'Adult', 'Mid-age', 'Senior'])

df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'] + 0.000001)



df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young', 'Adult', 'Mid-age', 'Senior'])

df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'] + 0.000001)

df['Temp_deviation'] = df['Body_Temp'] - 37



df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young', 'Adult', 'Mid-age', 'Senior'])

df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'] + 0.000001)

df['Temp_deviation'] = df['Body_Temp'] - 37

def calculate_bmr(row):
    if row['Sex'] == 1: #Male
        return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] + 5
    else:  # Female
        return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] - 161

df['BMR'] = df.apply(calculate_bmr, axis=1)



# sns.scatterplot(x='BMI', y='Calories', data=df, hue='Sex')
# plt.title("BMI vs. Calories")
# plt.xlabel("BMI")
# plt.ylabel("Calories Burned")
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr(numeric_only=True)
corr = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()



df.shape


# import pandas as pd
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
# from xgboost import XGBRegressor

# # 2. Split into features and target
# X = df.drop(columns=['Calories'])
# y = df['Calories']

# # 3. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Train models
# models = {
#     "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
#     "LightGBM": LGBMRegressor(random_state=42),
#     "XGBoost": XGBRegressor(verbosity=0, random_state=42)
# }

# # 5. Train and evaluate each model
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Calculate performance metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     # Calculate RMSLE
#     rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

#     # Print evaluation metrics
#     print(f"\n{name} Model:")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"R-squared (R²): {r2:.4f}")
#     print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.4f}")



# import catboost
# from catboost import CatBoostRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
# import numpy as np

# # Assuming df is your dataframe and has been preprocessed

# # 2. Split into features and target
# X = df.drop(columns=['Calories'])
# y = df['Calories']

# # 3. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Initialize CatBoostRegressor model
# model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, random_seed=42, verbose=200)

# # 5. Train the model
# model.fit(X_train, y_train)

# # 6. Make predictions
# y_pred = model.predict(X_test)

# # 7. Calculate performance metrics
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

# # 8. Print evaluation metrics
# print(f"CatBoost Model Evaluation:")
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print(f"R-squared (R²): {r2:.4f}")
# print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.4f}")



import catboost
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import numpy as np

# 1. Log1p transform on target
df['Calories_log'] = np.log1p(df['Calories'])

# 2. Identify categorical features by column names or indices
categorical_features = ['Age_Group']  # Add other categorical cols if present, e.g., 'Age_Group'

# 3. Split into features and transformed target
X = df.drop(columns=['Calories', 'Calories_log'])  # Use raw features
y = df['Calories_log']                             # Use log-transformed target

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=200
)

# 6. Train the model with cat_features
model.fit(X_train, y_train, cat_features=categorical_features)

# 7. Predict and inverse transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)        # Invert log1p
y_true = np.expm1(y_test)            # Invert true log1p target

# 8. Evaluation metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

# 9. Print results
print(f"CatBoost Model Evaluation (Log1p transformed target):")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"RMSLE: {rmsle:.4f}")



# model_f = CatBoostRegressor(
#     iterations=1000,
#     learning_rate=0.1,
#     depth=6,
#     random_seed=42,
#     verbose=200
# )

# # 6. Train the model with cat_features
# model_f.fit(X, y, cat_features=categorical_features)


# %%capture
# !pip install flaml


# from flaml import AutoML


# aml = AutoML()
# aml.fit(df.drop(columns='Calories'), np.log1p(df['Calories']), task='regression', metric='rmse', time_budget=600)





# import numpy as np
# import pandas as pd
# from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_log_error

# # 1. Log1p transform
# df['Calories_log'] = np.log1p(df['Calories'])
# X = df.drop(columns=['Calories', 'Calories_log'])
# y = df['Calories_log'].values

# # 2. Initialize 5-fold CV
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# rmsle_scores = []

# # 3. Cross-validation loop
# for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model = CatBoostRegressor(
#         iterations=1000,
#         learning_rate=0.1,
#         depth=6,
#         random_seed=42,
#         verbose=0
#     )

#     model.fit(X_train, y_train)
#     y_pred_log = model.predict(X_val)

#     # Convert predictions back from log1p scale
#     y_pred = np.expm1(y_pred_log)
#     y_true = np.expm1(y_val)

#     # RMSLE
#     rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
#     rmsle_scores.append(rmsle)

#     print(f"Fold {fold + 1} RMSLE: {rmsle:.4f}")

# # 4. Average RMSLE
# print(f"\nMean RMSLE across 5 folds: {np.mean(rmsle_scores):.4f}")



dt=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


dt.info()


ids=dt['id']


dt=dt.drop(columns='id')


dt['Sex'] = dt['Sex'].map({'female': 0, 'male': 1})


dt['BMI'] = dt['Weight'] / (dt['Height'] / 100) ** 2



dt['Age_Group'] = pd.cut(dt['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['Teen', 'Young', 'Adult', 'Mid-age', 'Senior'])

dt['HR_per_min'] = dt['Heart_Rate'] / (dt['Duration'] + 0.000001)

dt['Temp_deviation'] = dt['Body_Temp'] - 37

def calculate_bmr(row):
    if row['Sex'] == 1: #Male
        return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] + 5
    else:  # Female
        return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] - 161

dt['BMR'] = dt.apply(calculate_bmr, axis=1)



dt.head()


preds=model.predict(dt)


PREDS=np.expm1(preds)


sub=pd.DataFrame({
    'id' : ids,
    'Calories' : PREDS
})


sub


import seaborn as sns
import matplotlib.pyplot as plt

# Plot KDE of the 'Calories' column
plt.figure(figsize=(8, 6))
sns.kdeplot(df['Calories'], shade=True)
plt.title('KDE of Calories')
plt.xlabel('Calories')
plt.ylabel('Density')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Plot KDE of the 'Calories' column
plt.figure(figsize=(8, 6))
sns.kdeplot(sub['Calories'], shade=True)
plt.title('KDE of Calories')
plt.xlabel('Calories')
plt.ylabel('Density')
plt.show()



sub.to_csv('submission.csv',index=False)

