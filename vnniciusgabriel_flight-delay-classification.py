import warnings
import time
from functools import wraps
from collections import Counter


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore")


DATA_PATH = "/kaggle/input/flight-delays-fall-2018"

train_df = pd.read_csv(f"{DATA_PATH}/flight_delays_train.csv.zip")
test_df = pd.read_csv(f"{DATA_PATH}/flight_delays_test.csv.zip")


train_df.info()


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Month'] = df['Month'].str.split('-', expand=True)[1].astype(int)
    df['DayofMonth'] = df['DayofMonth'].str.split('-', expand=True)[1].astype(int)
    df['DayOfWeek'] = df['DayOfWeek'].str.split('-', expand=True)[1].astype(int)
    df['Flight'] = df['Origin'] + '-' + df['Dest']
    
    df['DepTime'] = df['DepTime'].astype(str)
    df['DepTime'] = pd.to_datetime(df['DepTime'], format='%H%M', errors='coerce')
    df['DepTime'] = df['DepTime'].dt.time

    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    
    return df

train_df_preprocessed = preprocessing_data(train_df)


train_df_preprocessed.head()


num_cols = ['Distance']
cat_cols = list(train_df_preprocessed.drop(columns=num_cols + ['dep_delayed_15min', 'DepTime'], axis=1))

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
])

X_train_transformed = pipeline.fit_transform(train_df_preprocessed.drop('dep_delayed_15min', axis=1))
y = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0})


y.value_counts()


y.hist()


sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X_train_transformed, y)


y.hist()


Counter(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.2f} seconds")
        return result
    return wrapper


@timeit
def train_and_plot_cm(model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    cm.plot()
    plt.show()


train_and_plot_cm(LogisticRegression())


train_and_plot_cm(RandomForestClassifier())


train_and_plot_cm(XGBClassifier())


train_and_plot_cm(CatBoostClassifier(verbose=0))


model = {
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(),
    'CatBoostClassifier': CatBoostClassifier(verbose=0)
}

params = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    'XGBClassifier': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.3],
        'subsample': [0.8, 1.0]
    },
    'CatBoostClassifier': {
        'iterations': [100, 200],
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }
}

@timeit
def hypertune_models():
    return  {
        name: GridSearchCV(model, params[name], cv=5, n_jobs=-1, scoring='accuracy').fit(X_train, y_train).best_estimator_
        for name, model in model.items()
    }

best_models = hypertune_models()


for name, model in best_models.items():
    
    y_pred = model.predict(X_test)
    print("=" * 60)
    print(f"Modelo: {name}\n")
    print(classification_report(y_test, y_pred))
    print("=" * 60, "\n")


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, (name, model) in zip(axes, best_models.items()):

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
    ax.set_title(f"confusion matrix {name}")
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))

for name, model in best_models.items():

    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"{name} AUC: {auc_score:.2f}")

plt.plot([0, 1], [0, 1], '--k', lw=2)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve comparision')
plt.legend(loc='lower right')
plt.show()


model = best_models['RandomForestClassifier']

test_df_preprocessed = preprocessing_data(test_df)
X_test_tranformed = pipeline.transform(test_df_preprocessed)

test_df_preprocessed['dep_delayed_15min'] = model.predict_proba(X_test_tranformed)[:, 1].round(3)
test_df


df_result = test_df_preprocessed[['dep_delayed_15min']].copy()

df_result.reset_index(inplace=True)
df_result.rename(columns={'index': 'id'}, inplace=True)

df_result.to_csv('/kaggle/working/submission.csv', index=False)

