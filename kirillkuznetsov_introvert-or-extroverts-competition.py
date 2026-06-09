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


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import optuna
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


for col in train:
    print(col, ": ", train[col].isna().sum(), "/", len(train[col]))


def feature_engineering(df):
    df = df.copy()

    # Social Engagement Score (interaction term)
    df['Social_score'] = (df['Social_event_attendance'] + df['Going_outside'] + df['Friends_circle_size'])
    # Introvert-Tendency Proxy
    df['Introvert_score'] = (df['Time_spent_Alone'] - df['Social_score'])
    df['Inp']=df['Introvert_score']-df['Post_frequency']
    df['set']=df['Social_event_attendance']-df['Time_spent_Alone']
    df['In_ex']=df['Stage_fear']+df['Drained_after_socializing']
    df['tsd']=df['Time_spent_Alone']+df['Stage_fear']+df['Drained_after_socializing']

    def team(a):
        if a<0:
            return 0
        else:
            return 1
    df['team']=df['set'].apply(team)

    def intro(a):
        if a>0:
            return 1
        else:
            return 0
    df['intro']=df['In_ex'].apply(intro)

    def tsd_filter(a):
        if a==5:
            return 0
        elif a<5:
            return 0
        else:
            return 1
    df['tsd_filter']=df['tsd'].apply(tsd_filter)
    
    return df

def encode_binary_yes_no(df):
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def handle_missing_values(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=['number']).columns
    
    for col in num_cols:
        if df[col].isna().sum() > 0:

            if set(df[col].dropna().unique()) <= {0, 1}:
                fill_val = df[col].mode()[0]
            elif 'time' in col.lower() or 'hours' in col.lower():
                fill_val = df[col].median()
            elif df[col].dtype == 'int64':
                fill_val = int(df[col].median())
            else:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                fill_val = (q1*0.4 + q3*0.6)
            
            df[col] = df[col].fillna(fill_val)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            if df[col].nunique() == 2:
                fill_val = df[col].mode()[0]
            else:
                fill_val = 'Unknown'
            
            df[col] = df[col].fillna(fill_val)
    
    if 'Friends_circle_size' in df.columns:
        df['Friends_circle_size'] = df['Friends_circle_size'].fillna(
            df.groupby('Social_event_attendance')['Friends_circle_size'].transform('median')
        )
    
    if 'Post_frequency' in df.columns:
        df['Post_frequency'] = df['Post_frequency'].fillna(
            df.groupby(['Social_event_attendance', 'Friends_circle_size'])['Post_frequency'].transform('median')
        )
    
    return df

def preprocess_data(df, is_train=True):
    df = df.copy()

    df = encode_binary_yes_no(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    
    if 'id' in df.columns:
        ids = df['id']
        df.drop('id', axis=1, inplace=True)
    
    if is_train:
        le = LabelEncoder()
        df['Personality'] = le.fit_transform(df['Personality'])
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if is_train:
        num_cols.remove('Personality')
    
    return df, cat_cols, num_cols, (ids if 'id' in locals() else None)


train_processed, cat_cols, num_cols, _ = preprocess_data(train)
test_processed, _, _, test_ids = preprocess_data(test, is_train=False)


X = train_processed.drop('Personality', axis=1)
y = train_processed['Personality']
X_test = test_processed


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


X_train.describe()


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


print("Best trial:")
trial = study.best_trial

print(f"  Accuracy: {trial.value}")
best_params = {}

print("Best hyperparameters:")
for key, value in trial.params.items():
    best_params[key] = value
    print(f"  {key}: {value}")


model=RandomForestClassifier(**best_params)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)


print(f"Средняя точность по {n_splits} фолдам: {np.mean(accuracies):.4f}")

model.fit(X, y)

final_predictions = model.predict(X)
print("Итоговые метрики по всему датасету:\n", classification_report(y, final_predictions))


test_pred = model.predict(X_test)


le = LabelEncoder()
le.fit(train['Personality'])
test_pred_labels = le.inverse_transform(test_pred)


submission = pd.DataFrame({
    'id': test['id'].values,
    'Personality': test_pred_labels
})
submission.to_csv('submission.csv', index=False)


print("Проверка submission файла:")
print(submission.head())




