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
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder

import optuna
import optuna.logging

from scipy.stats import chi2_contingency

import warnings
import contextlib
import io



!pip install optuna


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


df.duplicated().sum()


df.isnull().sum()


df.nunique()


numeric_cols = [
    'Temparature',
    'Humidity',
    'Moisture',
    'Nitrogen',
    'Potassium',
    'Phosphorous'
]


nominal_cols = [
    'Soil Type',
    'Crop Type'
]


target_col = 'Fertilizer Name'


blue_shade = '#004DAB'

for col in nominal_cols + [target_col]:
    plt.figure(figsize=(8, 6))
    df[col].value_counts().plot(
        kind='bar',
        color=blue_shade,
        edgecolor=None
    )
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def cramers_v_matrix(df, cols):
    matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                confusion = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(confusion)
    return matrix.astype(float)

cramers_matrix = cramers_v_matrix(df, nominal_cols + [target_col])

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Cramér's V Correlation Heatmap")
plt.tight_layout()
plt.show()


warnings.filterwarnings('ignore', category=FutureWarning)

for feature in numeric_cols:
    plt.figure(figsize=(8, 5))
    
    sns.histplot(df[feature], kde=True, bins=30, color=blue_shade)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()


for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=target_col, y=col, data=df, inner='box', palette='pastel')
    plt.title(f'{col} by {target_col}', fontsize=14)
    plt.xlabel('Fertilizer Name')
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


X = pd.concat([df[numeric_cols], pd.get_dummies(df[nominal_cols], drop_first=True)], axis=1)
le = LabelEncoder()
y = le.fit_transform(df[target_col])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def objective(trial):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y)),
                'metric': 'multi_logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'verbosity': -1
            }

            model = LGBMClassifier(**params)

            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        early_stopping(stopping_rounds=10),
                        log_evaluation(0)
                    ]
                )
            except Exception:
                raise TrialPruned()

            prob = model.predict_proba(X_val)
            top_3_preds = np.argsort(prob, axis=1)[:, -3:][:, ::-1]
            return mapk(y_val, top_3_preds.tolist(), k=3)


optuna.logging.set_verbosity(optuna.logging.ERROR)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)


best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)
top_3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_val, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().drop(index='accuracy')

plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report")
plt.show()

map3 = mapk(y_val, top_3_preds.tolist(), k=3)

plt.figure(figsize=(4, 3))
plt.bar(['MAP@3'], [map3])
plt.ylim(0, 1)
plt.title(f"MAP@3: {map3:.3f}")
plt.show()

