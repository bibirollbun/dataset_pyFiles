import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


train= pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.drop(columns='id',inplace=True)

test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


print('shape : ',train.shape)
train.info()


# Convert Sex to numeric if not already
sex_map = {'female':0, 'male':1}
train['Sex'] = train['Sex'].map(sex_map).astype(int)
test['Sex'] = test['Sex'].map(sex_map).astype(int)


train.describe()


# Boxplot to detect outliers
plt.figure(figsize=(10,4))
sns.boxplot(x=train['Calories'], color='orange')
plt.title('Boxplot of Calories Burned')
plt.show()


# Distribution of target
plt.figure(figsize=(10,6))
sns.histplot(train['Calories'], bins=50, kde=True, color='coral')
plt.title('Distribution of Calories Burned')
plt.show()


# Log transform visualization
plt.figure(figsize=(10,6))
sns.histplot(np.log1p(train['Calories']), bins=50, kde=True, color='green')
plt.title('Log-Transformed Distribution of Calories')
plt.show()


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

plt.figure(figsize=(16,12))
for i, col in enumerate(features):
    plt.subplot(3,3,i+1)
    sns.histplot(train[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Sex vs Calories
plt.figure(figsize=(8,6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories Burned by Sex')
plt.show()


# Age vs Calories
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Calories', data=train, alpha=0.3)
plt.title('Calories vs Age')
plt.show()


# Height & Weight vs Calories
fig, axes = plt.subplots(1, 2, figsize=(16,6))
sns.scatterplot(x='Height', y='Calories', data=train, alpha=0.3, ax=axes[0])
axes[0].set_title('Calories vs Height')
sns.scatterplot(x='Weight', y='Calories', data=train, alpha=0.3, ax=axes[1])
axes[1].set_title('Calories vs Weight')
plt.show()


# Duration vs Calories
plt.figure(figsize=(8,6))
sns.scatterplot(x='Duration', y='Calories', data=train, alpha=0.3, color='purple')
plt.title('Calories vs Duration')
plt.show()


# Heart Rate vs Calories
plt.figure(figsize=(8,6))
sns.scatterplot(x='Heart_Rate', y='Calories', data=train, alpha=0.3, color='teal')
plt.title('Calories vs Heart Rate')
plt.show()


# Body Temp vs Calories
plt.figure(figsize=(8,6))
sns.scatterplot(x='Body_Temp', y='Calories', data=train, alpha=0.3, color='brown')
plt.title('Calories vs Body Temp')
plt.show()


plt.figure(figsize=(10,8))
corr = train.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()


# Sample 1000 rows for pairplot (to keep it fast)
sampled = train.sample(1000, random_state=42)
sns.pairplot(sampled, vars=['Calories', 'Age', 'Weight', 'Duration', 'Heart_Rate'])
plt.show()


# Prepare the data
X = train.drop(columns='Calories')
y = train['Calories']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Log-transform the target variable
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                             eval_metric='rmse', 
                             random_state=42,
                             learning_rate =0.05,
                             max_depth = 9,
                             n_estimators = 200,
                             subsample =0.8 ,
                             colsample_bytree = 0.8
                            )


xgb_model.fit(X_train, y_train_log)

# Predictions on train and validation sets
y_train_pred_log = xgb_model.predict(X_train)
y_val_pred_log = xgb_model.predict(X_val)

# Calculate RMSLE for train and validation sets
train_rmsle = np.sqrt(mean_squared_log_error(y_train_log, y_train_pred_log))
val_rmsle = np.sqrt(mean_squared_log_error(y_val_log, y_val_pred_log))

print(f"Train RMSLE: {train_rmsle:.4f}, Validation RMSLE: {val_rmsle:.4f}")


X_test = test.drop(columns='id')

# Predict on test set (log-scale)
y_test_pred_log = xgb_model.predict(X_test)

# Inverse transform predictions
y_test_pred = np.expm1(y_test_pred_log)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_test_pred
})

submission.to_csv('submission_xgboost.csv', index=False)
submission.head()

