import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(f"Training dataset: {train.shape} | Test dataset: {test.shape}")
print(f"Missing values: {train.isnull().sum().sum()}")
print(f"Numerical variables: {train.select_dtypes(include=['int64', 'float64']).columns.tolist()}")
print(f"Categorical variables: {train.select_dtypes(include=['object']).columns.tolist()}")


plt.figure(figsize=(10, 4))
sns.histplot(train['Calories'], kde=True)
plt.title('Calories Expenditure Distribution')
plt.show()


numeric_cols = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col not in ['id', 'Calories']]
plt.figure(figsize=(10, 8))
corr = train[numeric_cols + ['Calories']].corr()
sns.heatmap(corr, annot=True, cmap='winter', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


X = train[numeric_cols]
y = train['Calories']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
features = pd.DataFrame({'Feature': numeric_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)


plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=features.head(5))
plt.title('Top 5 Important Features')
plt.show()


top_feature = features.iloc[0]['Feature']
plt.figure(figsize=(10, 5))
sns.scatterplot(x=top_feature, y='Calories', data=train)
plt.title(f'Calori vs {top_feature}')
plt.show()


print("\nEDA Summary:")
print(f"1. Target variable distribution: Mean={train['Calories'].mean():.2f}, Std={train['Calories'].std():.2f}")
print(f"2. Variable with highest correlation to calories: {corr['Calories'].sort_values(ascending=False).index[1]}")
print(f"3. Most important feature according to Random Forest: {features.iloc[0]['Feature']}")
print(f"4. Variable pair with highest correlation: {pd.DataFrame(np.where(np.triu(corr.values, 1) > 0.7), dtype=int).T.apply(lambda x: corr.columns[x[0]] + ' - ' + corr.columns[x[1]], axis=1).values}")

