import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
test = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/test.csv')


print(f"Train Dataset Columns: {train.columns}")
print(f"Test Dataset Columns: {test.columns}")


train.describe()


train.info()


print("Dataset shape:", train.shape)
print(train.head())


# Price Distribution
plt.figure(figsize=(6, 4))
sns.histplot(train['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(np.log1p(train["price"]), bins=50, kde=True)
plt.title("Log(Price) Distribution")
plt.show()


# Scatter plot: carat vs price
plt.figure(figsize=(6, 4))
sns.scatterplot(x="carat", y="price", data=train, alpha=0.5)
plt.title("Carat vs Price")
plt.show()


# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Ordinal encoding maps

cut_map = {
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Premium": 4,
    "Ideal": 5,
    "Signature Ideal": 6
}


color_map = {
    "J": 1,
    "I": 2,
    "H": 3,
    "G": 4,
    "F": 5,
    "E": 6,
    "D": 7
}


clarity_map = {
    "I3": 1, "I2": 2, "I1": 3,
    "SI2": 4, "SI1": 5,
    "VS2": 6, "VS1": 7,
    "VVS2": 8, "VVS1": 9, "IF": 10
}


train['cut_ord'] = train['cut'].map(cut_map)
train['color_ord'] = train['color'].map(color_map)
train['clarity_ord'] = train['clarity'].map(clarity_map)


# Handle Invalid Dimensions
train['missing_dim'] = ((train['x'] <= 0) | (train['y'] <= 0) | (train['z'] <= 0)).astype(int)
train.loc[train['x'] <= 0, 'x'] = np.nan
train.loc[train['y'] <= 0, 'y'] = np.nan
train.loc[train['z'] <= 0, 'z'] = np.nan


# Derived Features
train['volume'] = train['x'] * train['y'] * train['z']
train['l_w_ratio'] = train['x'] / train['y']
train['log_price'] = np.log1p(train['price'])


features = [
    "carat", "cut_ord", "color_ord", "clarity_ord",
    "depth", "table", "x", "y", "z", "volume", "l_w_ratio", "missing_dim"
]
target = "log_price"


X = train[features]
y = train[target]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ))
])


print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")


y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
print(f"RMSE on test set (price scale): {rmse:.2f}")


rf = model.named_steps['regressor']
importances = rf.feature_importances_
fi = pd.DataFrame({
    'features': features,
    'importance': importances
})
fi = fi.sort_values(by='importance', ascending=False)
print("\nFeature importances:")
print(fi)


test["cut_ord"] = test["cut"].map(cut_map)
test["color_ord"] = test["color"].map(color_map)
test["clarity_ord"] = test["clarity"].map(clarity_map)

test["missing_dim"] = ((test["x"] <= 0) | (test["y"] <= 0) | (test["z"] <= 0)).astype(int)
test.loc[test["x"] <= 0, "x"] = np.nan
test.loc[test["y"] <= 0, "y"] = np.nan
test.loc[test["z"] <= 0, "z"] = np.nan

test["volume"] = test["x"] * test["y"] * test["z"]
test["l_w_ratio"] = test["x"] / test["y"]


X_final = test[features]
test_pred_log = model.predict(X_final)
test_pred = np.expm1(test_pred_log)


submission = pd.DataFrame({
    "id": test['id'],
    "price": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")




