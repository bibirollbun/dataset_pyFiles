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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
#submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
train.head()


print("Train Shape:", train.shape)


train.info()


train.describe()


plt.figure(figsize=(8, 4))
sns.histplot(train['Listening_Time_minutes'], bins=50, kde=True, color='b')
plt.title('Distribution of Listening Time (Target)')
plt.xlabel("Listening Time (minutes)")
plt.show()


missing = train.isnull().sum()
print("Missing values in train:\n", missing[missing > 0])


for df in [train, test]:
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].mode()[0])


plt.figure(figsize=(8, 4))
sns.boxplot(data=train, x='Genre', y='Listening_Time_minutes')
plt.title('Listening Time by Genre')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=train, x='Episode_Length_minutes', y='Listening_Time_minutes', hue='Genre', alpha=0.6)
plt.title('Episode Length vs Listening Time')
plt.show()


sns.heatmap(train.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()


train.Publication_Time.value_counts()


for df in [train, test]:
    df['Is_Morning'] = df['Publication_Time'].apply(lambda x: 1 if x == 'Morning' else 0)
    df['Is_Afternoon'] = df['Publication_Time'].apply(lambda x: 1 if x == 'Afternoon' else 0)
    df['Is_Evening'] = df['Publication_Time'].apply(lambda x: 1 if x == 'Evening' else 0)
    df['Is_Night'] = df['Publication_Time'].apply(lambda x: 1 if x == 'Night' else 0)
    df['Guest_vs_Host_Popularity'] = df['Guest_Popularity_percentage'] - df['Host_Popularity_percentage']
    df['Host_Guest_Combined'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']


categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le


# Prepare Features and Target

X = train.drop(columns=['id', 'Listening_Time_minutes'])
y = train['Listening_Time_minutes']
X_test = test.drop(columns=['id'])



# Model Evaluation and Selection
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ]),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=0)
}


import joblib
import os

# Create a directory to save models
os.makedirs("saved_models", exist_ok=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}


for name, model in models.items():
    try:
        model_path = f"saved_models/{name.replace(' ', '_')}.pkl"
        if os.path.exists(model_path):
            print(f"Loading pre-trained {name} model from disk...")
            trained_model = joblib.load(model_path)
        else:
            print(f"Training {name} model...")
            trained_model = model.fit(X, y)
            joblib.dump(trained_model, model_path)

        scores = cross_val_score(trained_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        mean_rmse = -np.mean(scores)
        results[name] = mean_rmse
        print(f"{name} CV RMSE (log target): {mean_rmse:.4f}")

    except Exception as e:
        print(f"{name} failed: {e}")


# Final Model Training and Prediction
best_model_name = min(results, key=results.get)
model_path = f"saved_models/{best_model_name.replace(' ', '_')}.pkl"
print(f"\nUsing best model: {best_model_name}")
best_model = joblib.load(model_path)

preds = best_model.predict(X_test)


# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f'{best_model_name} - Feature Importances')
    plt.show()


# Submission

submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")



# Submission 2
submission2 = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission2.to_csv('submission2.csv', index=False)
print("Submission2 saved as submission2.csv")

