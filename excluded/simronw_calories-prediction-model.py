# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


# check for null values
print(train_df.isnull().sum())
print("\n")
print(test_df.isnull().sum())


print(train_df.shape)
print("\n")
print(test_df.shape)



train_df.describe()


test_df.describe()


numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
skew_values = train_df[numeric_cols].skew().sort_values(ascending=False)
print("\nSkewness:\n", skew_values)



plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(train_df[col], kde=True, bins=40, color='skyblue')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()



plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=train_df, y=col, color='lightcoral')
    plt.title(f'{col} Boxplot')
plt.tight_layout()
plt.show()



sns.histplot(train_df['Calories'], bins=50, kde=True, color='green')
plt.title("Target Distribution: Calories")
plt.xlabel("Calories")
plt.show()



plt.figure(figsize=(10, 6))
corr = train_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])


train_df.head()


# Pearson (default)
train_df.corr(method='pearson')

# # Kendall (good for ordinal, non-linear)
# train_df.corr(method='kendall')

# # Spearman (good for monotonic relationships)
train_df.corr(method='spearman')



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

corr = train_df.corr()

# Create mask to hide upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 6))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()



sns.pairplot(train_df[['Calories', 'Duration', 'Heart_Rate', 'Body_Temp']], diag_kind='kde')
plt.show()



pip install phik



X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories']
X_test = test_df.drop(['id'], axis=1)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)



X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)



model2 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model2.fit(X_train, y_train)



val_preds = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds.clip(0)))
print("Validation RMSLE:", rmsle)



val_preds2 = model2.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds2.clip(0)))
print("Validation RMSLE:", rmsle)


test_preds = model.predict(X_test_scaled)
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_preds.clip(0)
})
submission.to_csv("submission.csv", index=False)



df = pd.read_csv("submission.csv")


df.head()


# Example features
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)

train_df['Heart_Rate_per_min'] = train_df['Heart_Rate'] / train_df['Duration']
test_df['Heart_Rate_per_min'] = test_df['Heart_Rate'] / test_df['Duration']



from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators': [300, 500, 800],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
search = RandomizedSearchCV(xgb, param_distributions=params, scoring='neg_mean_squared_log_error', cv=3, n_iter=30, verbose=1)
search.fit(train_df.drop(['id', 'Calories'], axis=1), np.log1p(train_df['Calories']))



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Split before fitting the model
X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply log1p transform on y
y_train_log = np.log1p(y_train)

# Fit the model using search
search.fit(X_train, y_train_log)

# Predict on validation set (still in log scale)
y_pred_log = search.best_estimator_.predict(X_val)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print("RMSLE on validation set:", rmsle)



from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Model 1: XGBoost
model1 = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    objective='reg:squarederror'
)

# Model 2: Random Forest
model2 = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Model 3: Gradient Boosting
model3 = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

# Combine into Voting Regressor
ensemble = VotingRegressor(
    estimators=[('xgb', model1), ('rf', model2), ('gb', model3)],
    n_jobs=-1
)

# Fit the ensemble on log-transformed target
ensemble.fit(X_train, y_train_log)

# Predict and inverse-transform
preds_log = ensemble.predict(X_test)
preds = np.expm1(preds_log)  # because you used log1p for training



y_train_log = np.log1p(train_df['Calories'])



preds = np.expm1(model.predict(X_test))



from sklearn.ensemble import VotingRegressor

model1 = XGBRegressor(...)
model2 = RandomForestRegressor(...)
model3 = GradientBoostingRegressor(...)

ensemble = VotingRegressor([('xgb', model1), ('rf', model2), ('gb', model3)])
ensemble.fit(X_train, y_train_log)





