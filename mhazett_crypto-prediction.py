import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/kaggle/input/train-crypto/train_crypto.csv')
print(df.info())
print(df['label'].value_counts())


df['label'].hist(bins=50, figsize=(10, 5))
plt.title("Label Histogram")
plt.xlabel("Label Value")
plt.ylabel("Frequency")
plt.grid(True)


sample_columns = df.columns[1:6]

df[sample_columns].hist(bins=50, figsize=(15, 8))
plt.suptitle("Sample Feature Distributions")
plt.tight_layout()


x = df['X100']
y = df['label']

plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.3, label='Data Points')

slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope * x + intercept, color='red', label='Regression Line')

plt.xlabel("X100")
plt.ylabel("Label")
plt.title("Label and X100 using regression line")
plt.legend()
plt.grid(True)


sns.regplot(x='X400', y='label', data=df, scatter_kws={'alpha':0.3}, line_kws={"color":"red"})
plt.title("X400 vs Label")
plt.grid(True)


if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

X = df.drop(columns=['label'])
y = df['label']
X = X.dropna(axis=1, how='all')

X = X.fillna(X.mean(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


slope, intercept = np.polyfit(y_test, y_pred, 1)
regression_line = slope * y_test + intercept

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(y_test, regression_line, color='red', label='Regression Line')
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.title("Actual vs Predicted Labels")
plt.grid(True)

