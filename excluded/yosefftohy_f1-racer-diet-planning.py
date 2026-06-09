# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/f-1-racer-diet-planning/train.csv")
test = pd.read_csv("/kaggle/input/f-1-racer-diet-planning/test.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head(10)


train.isnull().sum()


train.describe()



import matplotlib.pyplot as plt

plt.hist(train['Calories'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Calories Burned")
plt.ylabel("Frequency")
plt.title("Distribution of Target Variable: Calories")
plt.show()


gender_counts =train['Sex'].value_counts()
genders = gender_counts.index
counts = gender_counts.values

# 2. Plot the data
plt.figure(figsize=(8, 6))
plt.bar(genders, counts, color=['blue', 'pink'])
plt.title('Number of Males vs. Females')
plt.xlabel('Gender')
plt.ylabel('Number of Individuals')

max_count = max(counts)
plt.ylim(0, max_count + max_count * 0.1) 

for i, v in enumerate(counts):
    plt.text(i, v + max_count * 0.02, str(v), ha='center', fontsize=12)

plt.show()



import seaborn as sns

numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
corr = train[numeric_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Numeric Features)")
plt.show()



fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.scatterplot(x='Weight', y='Calories', data=train, ax=axes[0,0])
sns.scatterplot(x='Height', y='Calories', data=train, ax=axes[0,1])
sns.scatterplot(x='Duration', y='Calories', data=train, ax=axes[0,2])
sns.scatterplot(x='Heart_Rate', y='Calories', data=train, ax=axes[1,0])
sns.scatterplot(x='Body_Temp', y='Calories', data=train, ax=axes[1,1])
sns.scatterplot(x='Age', y='Calories', data=train, ax=axes[1,2])

plt.tight_layout()
plt.show()



plt.figure(figsize=(6,5))
sns.scatterplot(data=train, x='Height', y='Weight', hue='Sex')
plt.title("Height vs Weight by Sex")
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(data=train, x='Duration', y='Heart_Rate')
plt.title("Heart Rate vs Duration")
plt.show()



plt.figure(figsize=(6,4))
sns.histplot(train['Body_Temp'], bins=30, kde=True)
plt.axvline(37, color='red', linestyle='--', label='Normal Temp (37°C)')
plt.title("Body Temperature Distribution")
plt.legend()
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(data=train, x='Age', y='Heart_Rate')
plt.title("Heart Rate vs Age")
plt.show()


for df in [train, test]:
    df['Sex'] = (df['Sex'] == 'male').astype(int)
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['Heart_Workload'] = df['Heart_Rate'] * df['Duration']
    df['Temp_Stress'] = (df['Body_Temp'] - 37) * df['Duration']
    df['HR_Ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Wt_per_Height'] = df['Weight'] / df['Height']


plt.figure(figsize=(6,4))
sns.histplot(train['BMI'], bins=30, kde=True, color='teal')
plt.axvline(18.5, color='red', linestyle='--', label='Underweight')
plt.axvline(25, color='orange', linestyle='--', label='Overweight')
plt.axvline(30, color='purple', linestyle='--', label='Obese')
plt.title("BMI Distribution with Health Ranges")
plt.legend()
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(data=train, x='Duration', y='Heart_Workload', hue='Sex')
plt.title("Heart Workload vs Duration")
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(data=train, x='Duration', y='Temp_Stress', hue='Sex')
plt.axhline(0, color='red', linestyle='--', label='Normal Temp Stress')
plt.title("Temperature Stress Over Duration")
plt.legend()
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(train['HR_Ratio'], bins=30, kde=True, color='green')
plt.axvline(0.7, color='red', linestyle='--', label='High Intensity (>70% max HR)')
plt.title("Heart Rate Ratio to Max Possible HR")
plt.legend()
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(train['Wt_per_Height'], bins=30, kde=True, color='blue')
plt.title("Weight per Height Ratio Distribution")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


lgb_model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.05,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=3, subsample=0.8, random_seed=42,
    verbose=200, early_stopping_rounds=50
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))


xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbose=100
)


val_preds_lgb = lgb_model.predict(X_val)
val_preds_cat = cat_model.predict(X_val)
val_preds_xgb = xgb_model.predict(X_val)

val_preds_avg = (val_preds_lgb + val_preds_cat + val_preds_xgb) / 3
rmse = np.sqrt(mean_squared_error(y_val, val_preds_avg))
print(f"Validation RMSE (Ensemble): {rmse:.4f}")


preds_test = (
    lgb_model.predict(X_test) +
    cat_model.predict(X_test) +
    xgb_model.predict(X_test)
) / 3


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': preds_test
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission saved to /kaggle/working/submission.csv")





