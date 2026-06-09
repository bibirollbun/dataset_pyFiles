import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col='id')

print(train.shape)
print(train.info())
train.head()


sns.histplot(train['accident_risk'], bins=30, kde=True, color='blue')
plt.title("Accident Risk Distribution")
plt.show()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col, y='accident_risk', data=train, palette='viridis')
    plt.title(f"{col} vs accident_risk")
    plt.xticks(rotation=45)
    plt.show()


num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
sns.heatmap(train[num_cols + ['accident_risk']].corr(), annot=True, cmap="coolwarm")
plt.show()


X = train.drop(['accident_risk'], axis=1)
y = train['accident_risk']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X train shape: {X_train.shape}")
print(f"X validation shape: {X_val.shape}")
print(f"y train shape: {y_train.shape}")
print(f"y validation shape: {y_val.shape}")


model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("Validation RMSE:", rmse)


feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_sorted = feat_importances.sort_values(ascending=True)

colors = cm.rainbow(np.linspace(0, 1, len(feat_sorted)))

plt.figure(figsize=(8, 6))
plt.barh(feat_sorted.index, feat_sorted.values, color=colors)
plt.title("Feature Importances (Rainbow, Ascending Order)")
plt.show()


X_test = pd.get_dummies(test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

preds = model.predict(X_test)

submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
submission['accident_risk'] = preds

submission.to_csv("submission.csv", index=False)
submission.head(10)

