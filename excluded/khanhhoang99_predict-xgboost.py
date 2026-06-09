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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


df_train.head()


df_train.shape


df_test.head()


def preprocess(df):
    df = df.copy()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    df['Workout_Intensity'] = df['Heart_Rate'] / df['Duration'].replace(0, 1)
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0
    df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Height_to_Weight'] = df['Height'] / df['Weight']
    df['Sex_Age_Interaction'] = df['Sex'] * df['Age']
    return df


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


df_train = preprocess(df_train)
df_test = preprocess(df_test)


X_train = df_train.drop(columns=['id', 'Calories'])
y_train = df_train['Calories']
X_test = df_test.drop(columns=['id'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}


grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)


grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test_scaled)


df_test['Calories'] = y_pred
df_test[['id', 'Calories']].to_csv('/kaggle/working/calories_predictions.csv', index=False)


import matplotlib.pyplot as plt
import seaborn as sns


feature_names = X_train.columns
importances = best_model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='mako')
plt.title('ğŸ”� XGBoost Feature Importance')
plt.tight_layout()
plt.show()

