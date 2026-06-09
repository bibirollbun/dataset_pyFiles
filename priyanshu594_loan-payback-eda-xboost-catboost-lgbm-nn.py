import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.display import display, HTML
import plotly.express as px
from scipy.stats import f_oneway, kruskal
import itertools
import colorsys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import colorsys
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from IPython.core.display import display, HTML
from warnings import filterwarnings
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np

import catboost
print(catboost.__version__)
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,KFold

import catboost as cb
from catboost import CatBoostClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import shap
import math

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  # Print datase

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
from tqdm.auto import tqdm
from xgboost import XGBClassifier, callback as xgb_callback
warnings.filterwarnings("ignore", category=FutureWarning)
filterwarnings('ignore')
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import warnings 
warnings.filterwarnings('ignore')



df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')




BLUE_BOLD = "\033[1;34m"
RESET = "\033[0m"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None)  

print(f"{BLUE_BOLD}ğŸ“� Shape of the DataFrame:{RESET} {df_train.shape}")
print(f"{BLUE_BOLD}â�¡ï¸� Rows:{RESET} {df_train.shape[0]}  {BLUE_BOLD}| Columns:{RESET} {df_train.shape[1]}")

print(f"\n{BLUE_BOLD}ğŸ”� Preview of the DataFrame (df.head(2)):{RESET}")
print(df_train.head(2))

print(f"\n{BLUE_BOLD}ğŸ•³ï¸� Missing Values in Each Column:{RESET}")
print(df_train.isnull().sum())

print(f"\n{BLUE_BOLD}ğŸ”¢ Unique Values in Each Column:{RESET}")
print(df_train.nunique().sort_values())


print(f"\n{BLUE_BOLD}â„¹ï¸� DataFrame Info:{RESET}")
df_train.info()


pd.reset_option('display.max_columns')
pd.reset_option('display.width')



def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['Duplicate'] = df.duplicated().sum()
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['avg'] = desc['mean'].values
    summ['std dev'] = desc['std'].values
    summ['top value'] = desc['top'].values
    summ['Freq'] = desc['freq'].values

    
    return summ

summary(df_train).style.background_gradient()


num_cols = ['credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_amount']
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']
target_col = 'loan_paid_back'
df_train.head()


!pip install -q git+https://github.com/priyanshu5943/Graphium.git > /dev/null 2>&1
import graphium as gh



gh.ip_cat_univariate(df_train)


gh.sp_num_univariate(df_train)


gh.ip_cat_vs_target(df_train, target_col='loan_paid_back')


gh.sp_feature_target_analysis(df_train, target_col='loan_paid_back',plot_type=2)


gh.sp_feature_target_analysis(df_train, target_col='credit_score',plot_type=3)


gh.sp_feature_target_analysis(df_train, target_col='annual_income',plot_type=3)


gh.sp_feature_target_analysis(df_train, target_col='loan_amount',plot_type=3)


gh.sp_feature_target_analysis(df_train, target_col='debt_to_income_ratio',plot_type=3)


gh.sp_feature_target_analysis(df_train, target_col='interest_rate',plot_type=3)


gh.mv_num_cat_vs_target_grid(
    df=df_train,
    num_cols=num_cols,      # numeric columns
    cat_cols=cat_cols,        # categorical columns
    target_col= target_col,                   # hue target
    plot_type='box',                               # 'violin', 'box', or 'swarm'
    n_cols=2                                       # plots per row
)



gh.mv_corr_heatmap(df_train)


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print('Train Shape:', train.shape)
train.head(3)


TARGET = 'loan_paid_back'
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]
print(len(FEATURES), 'features.')


X = train[FEATURES]
y = train[TARGET]


from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': True,
}


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')


import seaborn as sns
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'feature': FEATURES, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 20))
sns.barplot(x='importance', 
            y='feature', 
            data=importance_df.head(50)) 
plt.title('Feature Importance (Fold5 model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(f'submission_xgb_{overall_auc}.csv', index=False)


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")




def create_features(df):
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
    return df

def delete_features(df):
    cols_to_drop = ['grade_subgrade','gender','marital_status']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def one_hot_encode(df):
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    return pd.get_dummies(df, columns=obj_cols, drop_first=False)

def bool_to_int(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


# 3. APPLY PREPROCESSING

train = create_features(train)
test  = create_features(test)

train = delete_features(train)
test  = delete_features(test)

train = one_hot_encode(train)
test  = one_hot_encode(test)

# align train/test columns
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0


test = test[train.columns]


test = test.drop(columns=['loan_paid_back'])

train = bool_to_int(train)
test  = bool_to_int(test)



def add_target_count_features(train, test, target_col, n_splits=10):
    BASE = [c for c in train.columns if c != target_col]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_train = pd.DataFrame(index=train.index)
    count_train = pd.DataFrame(index=train.index)
    mean_test = pd.DataFrame(index=test.index)
    count_test = pd.DataFrame(index=test.index)

    for col in BASE:
        if train[col].isnull().all():  
            continue

        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            m = tr_fold.groupby(col)[target_col].mean()
            mean_encoded[val_idx] = val_fold[col].map(m)

        mean_train[f'mean_{col}'] = mean_encoded
        global_mean = train.groupby(col)[target_col].mean()
        mean_test[f'mean_{col}']  = test[col].map(global_mean)

        count_map = train[col].value_counts()
        count_train[f'count_{col}'] = train[col].map(count_map)
        count_test[f'count_{col}']  = test[col].map(count_map)

    train = pd.concat([train, mean_train, count_train], axis=1)
    test  = pd.concat([test, mean_test, count_test], axis=1)

    return train, test

train, test = add_target_count_features(train, test, 'loan_paid_back')


def split_data(df):
    X = df.drop(columns=['id','loan_paid_back'])
    y = df['loan_paid_back']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(train)
test_X = test.drop(columns=['id']).copy()


def build_catboost_model(iterations=1000, depth=5, learning_rate=0.1):
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=42,
        eval_metric="AUC",
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False
    )

model = build_catboost_model()
model.fit(X_train, y_train)


def evaluate_metrics(y_true, y_pred_proba):
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'Accuracy':  accuracy_score(y_true, y_pred_bin),
        'Precision': precision_score(y_true, y_pred_bin, zero_division=0),
        'Recall':    recall_score(y_true, y_pred_bin),
        'F1':        f1_score(y_true, y_pred_bin),
        'AUC':       roc_auc_score(y_true, y_pred_proba)
    }

    return pd.DataFrame([metrics])


pred_probs = model.predict_proba(X_test)[:, 1]
results = evaluate_metrics(y_test, pred_probs)
display(results)



import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import numpy as np
def plot_confusion_interactive(y_true, y_pred, normalize=True):
    # Compute confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize=('true' if normalize else None)
    )

    # Labels
    labels = ["0", "1"]


    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        hoverongaps=False,
        text=np.round(cm, 3),
        texttemplate="%{text}",
        colorscale="Blues"
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=500,          # smaller width
        height=300,         # smaller height
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=10) , # smaller font
 
    )
    fig.show()
plot_confusion_interactive(y_test, (pred_probs >= 0.5).astype(int))





test_probs = model.predict_proba(test_X)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_probs
})

submission.head()
submission.isnull().sum()



submission.to_csv("submission_catboost.csv")

