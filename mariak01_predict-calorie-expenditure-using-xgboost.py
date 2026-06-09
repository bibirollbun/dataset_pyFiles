# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, r2_score


# load data
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
tdf = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# top 5 rows
df.head()


# shape
print(f'rows - {df.shape[0]} \ncolumns - {df.shape[1]}')


# data info
df.info()


# check null value
df.isnull().sum()


# check duplicates
df.duplicated().sum()


# statistical summary
df.describe()


# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# univariate analysis (frequency of distribution)
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

plt.figure(figsize=[12, 8])

for i, feature in enumerate(features):
  plt.subplot(3, 3, i+1)
  sns.histplot(data=df, x=feature, bins=15, kde=True)
  plt.title(f'distribution of {feature}')
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()


# checking skewness

skewness = df[features].skew()
print(skewness)


# univariate analysis (outliers)

plt.figure(figsize=(10, 8))

for i, feature in enumerate(features):
  plt.subplot(3, 3, i+1)
  sns.boxplot(data=df, x=feature)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()


# bivariate analysis (numerical vs numerical)
n_features = ['Duration', 'Body_Temp', 'Heart_Rate', 'Weight', 'Age']

plt.figure(figsize=(12, 8))

for i, feature in enumerate(n_features):
  plt.subplot(2, 3, i+1)
  sns.scatterplot(data=df, x=feature, y='Calories')
  plt.title(f'{feature} vs Calories')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# bivariate analysis (categorical vs numerical)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Sex vs Calories")
plt.xlabel("Sex")
plt.ylabel("Calories")
sns.boxplot(data=df, x='Sex', y='Calories', palette='Dark2')

plt.subplot(1, 2, 2)
plt.title("Sex vs Weight")
plt.xlabel("Sex")
plt.ylabel("Weight")
sns.barplot(data=df, x='Sex', y='Weight', errorbar='sd', palette='flare')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# Correlation

col = df.select_dtypes(include='number')
correlation = col.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f')
plt.xticks(rotation=45, ha='right')
plt.show()


# converting datatypes
le = LabelEncoder()

# for train data
df['Sex'] = le.fit_transform(df['Sex'])

# for test data
tdf['Sex'] = le.transform(tdf['Sex'])


# outlier using IQR

Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

condition = df[(df[features] < (Q1 - 1.5*IQR)) | (df[features] > (Q3 + 1.5*IQR))]
outliers = df[condition.any(axis=1)]
print(outliers.value_counts())


# remove outliers
df = df[~condition.any(axis=1)]
df


# test dataset
tdf.head()


# X and y

X = df.drop('Calories', axis=1)
y = df['Calories']


# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []
rmsle_scores = []

# splitting
for train_index, val_index in kf.split(X):
  X_train_k, X_val_k = X.iloc[train_index], X.iloc[val_index]
  y_train_k, y_val_k = y.iloc[train_index], y.iloc[val_index]

# model building & prediction
  model_k = XGBRegressor(random_state=42)
  model_k.fit(X_train_k, y_train_k)
  y_pred_k = model_k.predict(X_val_k)
  y_pred_k = np.clip(y_pred_k, 0, None)
    
# evaluation
  r2 = r2_score(y_val_k, y_pred_k)
  r2_scores.append(r2)

  rmsle = np.sqrt(mean_squared_log_error(y_val_k, y_pred_k))
  rmsle_scores.append(rmsle)

print(f'RMSLE: {np.mean(rmsle_scores): .3f}')
print(f'R2 Score: {np.mean(r2_scores): .3f}')


# Visualization of actual vs predicted data (Validation Set)

plt.figure(figsize=(8, 5))

sns.scatterplot(x=y_val_k, y=y_pred_k, color='mediumseagreen', alpha=0.6, s=60)
sns.lineplot(x=[y_val_k.min(), y_val_k.max()], y=[y_pred_k.min(), y_pred_k.max()], color='crimson', linestyle='--', label='Prediction line')

plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.tight_layout()
plt.grid(True)
plt.show()


# model building & training

model = XGBRegressor(random_state=42)
model.fit(X, y)


# final prediction on test data

final_pred = model.predict(tdf)
final_pred = np.clip(final_pred, 0, None)
final_pred


# submission

submission = pd.DataFrame({
    'id': tdf['id'],
    'Calories': final_pred
})

submission.head()


# save submission to csv

submission.to_csv('submission.csv', index=False)

