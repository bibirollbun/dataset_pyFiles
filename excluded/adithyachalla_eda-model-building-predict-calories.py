# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_df


test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df


print(train_df.describe())


train_df.isnull().sum()


train_df['Sex'].value_counts()


sns.countplot(x='Sex', data=train_df)
plt.title("Distribution of Sex")
plt.show()



# Plot histogram with KDE
sns.histplot(train_df['Age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution with KDE")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


sns.histplot(train_df['Calories'], kde=True)
plt.title('Calories Burned Distribution')
plt.show()



sns.boxplot(x='Sex', y='Calories', data=train_df)
plt.title('Calories Burned by Sex')
plt.show()




sns.scatterplot(x='Duration', y='Calories', data=train_df, hue='Sex', alpha=0.5)
plt.title('Calories vs Duration')
plt.show()



sns.lmplot(x='Heart_Rate', y='Calories', data=train_df, hue='Sex')
plt.title('Heart Rate vs Calories Burned')
plt.show()



print(train_df.groupby('Sex')['Calories'].mean())


train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0, 20, 40, 60, 80], labels=['0-20','21-40','41-60','61-80'])

sns.boxplot(x='Age_Group', y='Calories', data=train_df)
plt.title('Calories by Age Group')
plt.show()



plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    train_df['Height'], train_df['Weight'],
    c=train_df['Calories'], cmap='viridis', alpha=0.6
)
plt.colorbar(scatter, label='Calories Burned')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight Colored by Calories Burned')
plt.grid(True)
plt.show()



stats.probplot(train_df['Age'].dropna(), dist="norm", plot=plt)
plt.title("QQ Plot of Age")
plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])  # male=1, female=0 (or vice versa)


train_df


from sklearn.model_selection import train_test_split

X = train_df.drop(['Calories', 'id','Age_Group'], axis=1)
y = train_df['Calories']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    print(f'{model.__class__.__name__} RMSLE: {rmsle:.4f}')



models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    LGBMRegressor(n_estimators=100, random_state=42),
    CatBoostRegressor(verbose=0, random_state=42)
]

for model in models:
    evaluate_model(model, X_train, y_train, X_val, y_val)



test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})
test_df


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Define base models
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
    ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
    ('catboost', CatBoostRegressor(verbose=0, random_state=42))
]

# Meta-model
meta_model = LinearRegression()

# Create stacking regressor
stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,  # includes original features in meta-model
    cv=5,
    n_jobs=-1
)



evaluate_model(stack_model, X_train, y_train, X_val, y_val)


# Drop 'id' column for prediction
X_test = test_df.drop(columns=['id'])

# Predict using the trained stacking model
y_test_pred = stack_model.predict(X_test)

# Clip any negative predictions (to avoid invalid calorie values)
y_test_pred = np.maximum(0, y_test_pred)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': y_test_pred
})

# Export to CSV
submission.to_csv("submission.csv", index=False)



# Combine training and validation data for final training
X_full = np.vstack((X_train, X_val))
y_full = np.concatenate((y_train, y_val))

best_model = CatBoostRegressor(verbose=0, random_state=42)
best_model.fit(X_full, np.ravel(y_full))



X_test = test_df.drop(columns=['id'])
y_test_pred = best_model.predict(X_test)

# Clip negative predictions (especially for RMSLE or real-world targets like Calories)
y_test_pred = np.maximum(0, y_test_pred)



submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': y_test_pred
})

submission.to_csv("submission1.csv", index=False)


