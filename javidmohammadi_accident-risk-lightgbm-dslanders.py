# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# Import Data
train_file = "/kaggle/input/playground-series-s5e10/train.csv"
test_file = "/kaggle/input/playground-series-s5e10/test.csv"
sample_file = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

df = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_sample = pd.read_csv(sample_file)
pd.set_option('display.max_columns', None)

df.head()


df_test.head()


df_sample.head()


df.shape, df_test.shape, df_sample.shape


df.info()


# Drop ID column
df = df.drop(columns=['id'])


df.describe()


df.isna().sum().sum()


df.duplicated().sum(), df.shape


df = df.drop_duplicates()
df.shape


categorical_cols = df.select_dtypes(exclude=['number']).columns

fig, axes = plt.subplots(2, 4, figsize=(12,8))
axes = axes.flatten()

for col, ax in zip(categorical_cols, axes):
    sns.countplot(
        x=col,
        data=df,
        order=df[col].value_counts().index,
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Countplot: {col}")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("")

plt.tight_layout()
plt.show()


# Draw Histogram
num_cols = df.select_dtypes(include=['number']).columns.to_list()

fig, axs = plt.subplots(2, 3, figsize=(8, 4))
axs = axs.flatten()

for ax, col in zip(axs, num_cols):
  sns.histplot(df[col], kde=True, bins=20, ax=ax, color="darkcyan")

plt.tight_layout()
plt.show()


df['num_lanes'].value_counts()


df['speed_limit'].value_counts()


df['num_reported_accidents'].value_counts()


df.sample(10)


# Draw Boxplot
fig, axs = plt.subplots(2, 3, figsize=(14, 4))
axs = axs.flatten()

for ax, col in zip(axs, num_cols):
  sns.boxplot(x=col, data=df, ax=ax)

plt.tight_layout()
plt.show()


corr_matrix = df[num_cols].corr()

plt.figure(figsize=(5,4))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True
)
plt.show()


# Log transform
log_cols = ['num_lanes', 'num_reported_accidents']
for col in log_cols:
    df[col] = np.log1p(df[col])


df.sample(10)


# Feature Engineering
df['risk_index'] = df['curvature'] * df['speed_limit'] / df['num_lanes']
df['accident_density'] = df['num_reported_accidents'] / df['num_lanes']
df['dangerous_weather'] = (df['weather'] == 'foggy') | (df['weather'] == 'rainy')
df['night_light_factor'] = (df['lighting'] == 'night') | (df['lighting'] == 'dim')



# Poly features
poly_features = ['speed_limit', 'curvature', 'risk_index']
for col in poly_features:
    df[f'{col}**2'] = df[col]



df.info()


# Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


df_encoded.head()


# Assign X, y
X = df_encoded.drop(columns=['accident_risk'])
y = df_encoded['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model init (found by Random Search)
best_model = LGBMRegressor(
    subsample=0.75,
    reg_lambda=1.5,
    reg_alpha=0.15,
    num_leaves=90,
    n_estimators=1100,
    max_depth=10,
    learning_rate=0.01,
    colsample_bytree=0.8,
    device='gpu',
    random_state=42,
    verbose=0,
)


# Train the model
best_model.fit(X_train_scaled, y_train)


# Prediction
y_pred = best_model.predict(X_test_scaled)


# Scores
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.5f}")
print(f" MSE: {mse:.5f}")
print(f"  R2: {r2:.5f}")


# Prediction & Score on Train set
y_pred = best_model.predict(X_train_scaled)

mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f"RMSE: {rmse:.5f}")
print(f" MSE: {mse:.5f}")
print(f"  R2: {r2:.5f}")


# Scaling
X_scaled = scaler.fit_transform(X)


# Fit model on whole data
best_model.fit(X_scaled, y)



# Prediction
y_pred = best_model.predict(X_scaled)


# Final Scores
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.5f}")
print(f" MSE: {mse:.5f}")
print(f"  R2: {r2:.5f}")


df_test.head()


df_test.info()


# Drop ID column
df_test = df_test.drop(columns=['id'])


# Log Transform
for col in log_cols:
    df_test[col] = np.log1p(df_test[col])


# Feature Engineering
df_test['risk_index'] = df_test['curvature'] * df_test['speed_limit'] / df_test['num_lanes']
df_test['accident_density'] = df_test['num_reported_accidents'] / df_test['num_lanes']
df_test['dangerous_weather'] = (df_test['weather'] == 'foggy') | (df_test['weather'] == 'rainy')
df_test['night_light_factor'] = (df_test['lighting'] == 'night') | (df_test['lighting'] == 'dim')



# Poly features
poly_features = ['speed_limit', 'curvature', 'risk_index']
for col in poly_features:
    df_test[f'{col}**2'] = df_test[col]



# Encoding
df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)


df_test.head()


# Standard Scaler
df_test_scaled = scaler.fit_transform(df_test)


# Prediction
y_test_pred = best_model.predict(df_test_scaled)


df_sample.head()


# Make submission DataFrame
submission = pd.DataFrame({
    "id": df_sample.id,
    "accident_risk": y_test_pred
})


submission.head()


# Save submission
submission.to_csv("submission.csv", index=False)




