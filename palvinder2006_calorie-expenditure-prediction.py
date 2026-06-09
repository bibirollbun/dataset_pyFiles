import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_df.head()


train_df.shape


train_df.info()


train_df.describe()


sb.scatterplot(x='Height', y='Weight', data=train_df)
plt.show()


features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = train_df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()


features = train_df.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(train_df[col])
plt.tight_layout()
plt.show()


test_df.head()


train_df.replace({'male': 0, 'female': 1}, inplace=True)
test_df.replace({'male': 0, 'female': 1}, inplace=True)


to_remove = ['Weight', 'Duration']
train_df.drop(to_remove, axis=1, inplace=True)
test_df.drop(to_remove, axis=1, inplace=True)


train_df.head()


test_df.head()


X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories'].values
X_test = test_df.drop(['id'], axis=1)


X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train, y_train)
val_preds = lr.predict(X_val)
print("Linear Regression")
print("MAE:", mean_absolute_error(y_val, val_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("R2 Score:", r2_score(y_val, val_preds))

lr_test_preds = np.maximum(lr.predict(X_test_scaled), 0)


lasso = Lasso()
lasso.fit(X_train, y_train)
val_preds = lasso.predict(X_val)
print("Lasso Regression")
print("MAE:", mean_absolute_error(y_val, val_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("R2 Score:", r2_score(y_val, val_preds))

lasso_test_preds = np.maximum(lasso.predict(X_test_scaled), 0)


ridge = Ridge()
ridge.fit(X_train, y_train)
val_preds = ridge.predict(X_val)
print("Ridge Regression")
print("MAE:", mean_absolute_error(y_val, val_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("R2 Score:", r2_score(y_val, val_preds))

ridge_test_preds = np.maximum(ridge.predict(X_test_scaled), 0)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
val_preds = rf.predict(X_val)
print("Random Forest")
print("MAE:", mean_absolute_error(y_val, val_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("R2 Score:", r2_score(y_val, val_preds))

rf_test_preds = rf.predict(X_test_scaled)


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
val_preds = xgb.predict(X_val)
val_preds_clipped = np.maximum(val_preds, 0)  # Clip negatives
print("XGBoost")
print("MAE:", mean_absolute_error(y_val, val_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("R2 Score:", r2_score(y_val, val_preds))

# Predict for submission (also clip negatives)
xgb_test_preds = xgb.predict(X_test_scaled)
xgb_test_preds = np.maximum(xgb_test_preds, 0)


base_path = "/kaggle/working/"

for name, preds in {
    "LinearRegression": lr_test_preds,
    "Lasso": lasso_test_preds,
    "Ridge": ridge_test_preds,
    "RandomForest": rf_test_preds,
    "XGBoost": xgb_test_preds,
    #"SVR": svr_test_preds,
}.items():
    sub = sample_submission.copy()
    sub['Calories'] = preds
    sub.to_csv(f"{base_path}Calories_submission_{name}.csv", index=False)

print("All submissions saved.")




