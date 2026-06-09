import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor



train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')


print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


train.info()


train.describe()


plt.figure(figsize=(8,5))
sns.histplot(train['price'], bins=50, kde=True, color='green')
plt.title("Distribution of Target: Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

train['price'].skew()


missing = train.isnull().mean().sort_values(ascending=False)
missing[missing > 0] * 100


categorical_cols = train.select_dtypes(include='object').columns.tolist()
numerical_cols = train.select_dtypes(include=['int64', 'float64']).drop(columns=['id', 'price']).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)


plt.figure(figsize=(10,8))
corr = train[numerical_cols + ['price']].corr()
sns.heatmap(corr[['price']].sort_values(by='price', ascending=False), annot=True, cmap='viridis')
plt.title("Correlation of Features with Price")
plt.show()


# Example: mileage vs price
sns.scatterplot(data=train, x='milage', y='price')
plt.title('Mileage vs Price')
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(10,4))
    sns.boxplot(x=col, y='price', data=train)
    plt.xticks(rotation=45)
    plt.title(f"Price by {col}")
    plt.show()


drop_cols = ['id']
if 'year' in train.columns:
    drop_cols.append('year')

train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)


categorical_cols = train.select_dtypes(include='object').columns
train[categorical_cols] = train[categorical_cols].fillna('Missing')
test[categorical_cols] = test[categorical_cols].fillna('Missing')

numeric_cols = train.select_dtypes(include=['int64', 'float64']).drop(columns=['price']).columns
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].median())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].median())


test.head()


for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


train['log_price'] = np.log1p(train['price'])


X = train.drop(columns=['price', 'log_price'])
y = train['log_price']  # log-transformed price

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


lr = LinearRegression()
scores = cross_val_score(lr, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
print("Linear Regression RMSE:", -scores.mean())


ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
print("Ridge Regression RMSE:", -scores.mean())


lasso = Lasso(alpha=0.01)
scores = cross_val_score(lasso, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
print("Lasso Regression RMSE:", -scores.mean())


X = train.drop(columns=['price', 'log_price'])
y = train['log_price']


xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

scores = cross_val_score(xgb_model, X, y, scoring='neg_root_mean_squared_error', cv=5)
print("XGBoost RMSE:", -scores.mean())


lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

scores = cross_val_score(lgb_model, X, y, scoring='neg_root_mean_squared_error', cv=5)
print("LightGBM RMSE:", -scores.mean())


cat_model = CatBoostRegressor(verbose=0, random_state=42)
scores = cross_val_score(cat_model, X, y, scoring='neg_root_mean_squared_error', cv=5)
print("CatBoost RMSE:", -scores.mean())


xgb_model.fit(X, y)

test_preds = xgb_model.predict(test)

final_preds = np.expm1(test_preds)

submission['price'] = final_preds
submission.to_csv('submission.csv', index=False)




