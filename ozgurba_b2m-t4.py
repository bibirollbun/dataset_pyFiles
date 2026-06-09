# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install -q lofo-importance # uzun bi çıktı istemiyorum

import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from collections import Counter
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import gc
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
import shap
import sklearn
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.impute import SimpleImputer
from lofo import LOFOImportance, Dataset, plot_importance
import re
import lightgbm as lgbm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings("ignore")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sample_submission = pd.read_csv("/kaggle/input/home-credit-default-risk/sample_submission.csv")
install_df = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")
app_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
app_test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
subm = pd.read_csv("../input/home-credit-default-risk/sample_submission.csv")
pos_cash = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
credit_card = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
previous_app = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
install_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')

def load_datasets():
    paths = {
        "train_dataf": "/kaggle/input/home-credit-default-risk/application_train.csv",
        "test_dataf": "/kaggle/input/home-credit-default-risk/application_test.csv",
        "sample_submission": "/kaggle/input/home-credit-default-risk/sample_submission.csv",
        "previous_application": "/kaggle/input/home-credit-default-risk/previous_application.csv",
        "bureau": "/kaggle/input/home-credit-default-risk/bureau.csv",
        "bureau_balance": "/kaggle/input/home-credit-default-risk/bureau_balance.csv"
    }
    datasets = {name: pd.read_csv(path, index_col=0 if name == "sample_submission" else None)
                for name, path in paths.items()}
    return (datasets["train_dataf"], datasets["test_dataf"], datasets["sample_submission"],
            datasets["previous_application"], datasets["bureau"], datasets["bureau_balance"])

train_dataf, test_dataf, sample_submission, previous_application, bureau, bureau_balance = load_datasets()


train_dataf.head()


test_dataf.head()


balanceddf = pd.concat((train_dataf[train_dataf["TARGET"]==1], train_dataf[train_dataf["TARGET"]==0].sample(n=train_dataf[train_dataf["TARGET"]==1].shape[0])))


column_types = train_dataf.dtypes.astype(str).value_counts().reset_index()
column_types.columns = ['Data Type', 'Count']

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Data Type', y='Count', data=column_types)

for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x() + p.get_width() / 2,
        y=height,
        s=f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

plt.title('Data Types of Columns')
plt.xlabel('Data Type')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(data=train_dataf, 
             x='TARGET',
             hue='TARGET',
             multiple="stack",
             kde=False,
             bins=15)
plt.title('Target-Class Distribution')
plt.xlabel('TARGET')
plt.ylabel('Count')

plt.show()


train_dataf.describe().T.style.background_gradient(cmap='coolwarm')


def genel_bakis(df):
    metrics = {
        'Number of rows': lambda df: df.shape[0],
        'Number of columns': lambda df: df.shape[1],
        'Number of columns with missing values': lambda df: df.isna().sum().astype(bool).sum(),
        'Number of duplicate rows': lambda df: df.duplicated().sum(),
        'Number of Float columns': lambda df: df.select_dtypes(include='float').shape[1],
        'Number of Int columns': lambda df: df.select_dtypes(include='int').shape[1],
        'Number of Object columns': lambda df: df.select_dtypes(include='object').shape[1],
        'Number of data points': lambda df: df.size - df.isna().sum().sum(),
        'Number of missing values': lambda df: df.isna().sum().sum(),
        'Percentage of missing values': lambda df: (df.isna().sum().sum() / df.size) * 100,
        'Memory Usage (in MB)': lambda df: df.memory_usage(deep=True).sum() / (1024 ** 2)
    }

    data = {'Metrics': [metric(df) for metric in metrics.values()]}

    result_df = pd.DataFrame(data, index=metrics.keys())

    def format_value(x):
        if isinstance(x, float):
            if abs(x) >= 1e6:
                return f"{x:,.0f}"
            elif x >= 1:
                return f"{x:,.2f}"
            else:
                return f"{x:.2%}"
        return f"{x:,}"

    formatted = result_df['Metrics'].apply(format_value)
    formatted_df = pd.DataFrame({'Metrics': formatted})
    print(formatted_df)

genel_bakis(train_dataf)


weekday_groupby = train_dataf.groupby(["WEEKDAY_APPR_PROCESS_START"])["TARGET"].agg([np.mean, np.size])
pd.DataFrame(weekday_groupby).style.background_gradient(cmap = "Pastel1")


AMT_target_groupby = train_dataf.groupby(["TARGET"])[["AMT_CREDIT","AMT_INCOME_TOTAL"]].agg([np.mean, np.std])
pd.DataFrame(AMT_target_groupby).style.background_gradient(cmap = "Pastel1")


def kolonlarda_missing_sayi(df):
    missing_values = df.isna().sum()
    
    missing_values = missing_values[missing_values > 0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('KOLONLARDA SAYISAL OLARAK EKSİK DEĞERLER')
    plt.xlabel('KOLONLAR')
    plt.ylabel('MİSSİNG VALUE SAYILARI')
    plt.xticks(rotation=90) 
    plt.show()

kolonlarda_missing_sayi(train_dataf)


def kolonlarda_missing_yuzde(df):
    missing_value_percentages = (df.isna().sum() / df.shape[0]) * 100
    missing_value_percentages = missing_value_percentages[missing_value_percentages > 0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_value_percentages.index, y=missing_value_percentages.values)

    plt.title('KOLONLARDA YÜZDELİK OLARAK EKSİK DEĞERLER')
    plt.xlabel('KOLONLAR')
    plt.ylabel('MİSSİNG VALUE YÜZDELERİ')
    plt.xticks(rotation=90)
    plt.show()

kolonlarda_missing_yuzde(train_dataf)


train_dataf.select_dtypes('object').apply(pd.Series.nunique,
                                          axis = 0)


for h in train_dataf.select_dtypes("object").columns:
    unique_count = train_dataf[h].nunique()
    
    if unique_count < 100:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        sns.histplot(train_dataf[h].dropna(),
                     color='#3b8bba',
                     edgecolor='black',
                     ax=ax)

        fig.text(0.1, 0.95, f'{h} (Unique: {unique_count})',
                 fontsize=18,
                 fontweight='bold',
                 fontfamily='Arial')

        ax.set_xlabel('Value', fontsize=14, fontfamily='Arial')
        ax.set_ylabel('Count', fontsize=14, fontfamily='Arial')

        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()


print("Histograms with Classes portions")

for h in train_dataf.select_dtypes("object").columns:
    if train_dataf[h].nunique() < 100:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        sns.histplot(
            x=balanceddf[h].dropna(), 
            hue=balanceddf["TARGET"], 
            multiple="dodge", 
            shrink=0.8,
            palette="viridis",
            edgecolor='black',
            ax=ax
        )

        fig.text(0.1, 0.95, f'{h}',
                 fontsize=18,
                 fontweight='bold',
                 fontfamily='Arial')

        ax.set_xlabel('Değerler', fontsize=14, fontfamily='Arial')
        ax.set_ylabel('Adet', fontsize=14, fontfamily='Arial')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()


train_dataf.info()


all_data = pd.concat([train_dataf, test_dataf], axis = 0)


all_data = pd.get_dummies(all_data , columns = ["NAME_HOUSING_TYPE", #5
                                                "WALLSMATERIAL_MODE", #7
                                                "FLAG_OWN_REALTY", 
                                                "FLAG_OWN_CAR",
                                                "CODE_GENDER"])


# --- EXT_SOURCE_1'i EXT_SOURCE_2 ve EXT_SOURCE_3 ile tahmin et ---
ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

data_ext1 = all_data[ext_cols].dropna()
X_ext1 = data_ext1[["EXT_SOURCE_2", "EXT_SOURCE_3"]]
y_ext1 = data_ext1["EXT_SOURCE_1"]

model_ext1 = make_pipeline(StandardScaler(), Ridge(alpha=0.001))
model_ext1.fit(X_ext1, y_ext1)

mask_ext1 = (
    all_data["EXT_SOURCE_1"].isna() &
    all_data["EXT_SOURCE_2"].notna() &
    all_data["EXT_SOURCE_3"].notna()
)
X_pred_ext1 = all_data.loc[mask_ext1, ["EXT_SOURCE_2", "EXT_SOURCE_3"]]

if not X_pred_ext1.empty:
    all_data.loc[mask_ext1, "EXT_SOURCE_1"] = model_ext1.predict(X_pred_ext1)

# --- EXT_SOURCE_3'ü EXT_SOURCE_1 ve EXT_SOURCE_2 ile tahmin et ---
data_ext3 = all_data[ext_cols].dropna()
X_ext3 = data_ext3[["EXT_SOURCE_1", "EXT_SOURCE_2"]]
y_ext3 = data_ext3["EXT_SOURCE_3"]

model_ext3 = make_pipeline(StandardScaler(), Ridge(alpha=0.001))
model_ext3.fit(X_ext3, y_ext3)

mask_ext3 = (
    all_data["EXT_SOURCE_3"].isna() &
    all_data["EXT_SOURCE_1"].notna() &
    all_data["EXT_SOURCE_2"].notna()
)
X_pred_ext3 = all_data.loc[mask_ext3, ["EXT_SOURCE_1", "EXT_SOURCE_2"]]

if not X_pred_ext3.empty:
    all_data.loc[mask_ext3, "EXT_SOURCE_3"] = model_ext3.predict(X_pred_ext3)


# median ile doldur - burdakiler asimetrik değişkenler
all_data["EXT_SOURCE_1"] = all_data["EXT_SOURCE_1"].fillna(all_data["EXT_SOURCE_1"].median())
all_data["EXT_SOURCE_2"] = all_data["EXT_SOURCE_2"].fillna(all_data["EXT_SOURCE_2"].median())
all_data["EXT_SOURCE_3"] = all_data["EXT_SOURCE_3"].fillna(all_data["EXT_SOURCE_3"].median())
all_data["DAYS_LAST_PHONE_CHANGE"] = all_data["DAYS_LAST_PHONE_CHANGE"].fillna(all_data["DAYS_LAST_PHONE_CHANGE"].median())

# ortalama ile doldur - simetrik 
all_data["AMT_ANNUITY"] = all_data["AMT_ANNUITY"].fillna(all_data["AMT_ANNUITY"].mean())
all_data["AMT_GOODS_PRICE"] = all_data["AMT_GOODS_PRICE"].fillna(all_data["AMT_GOODS_PRICE"].mean())
all_data["AMT_REQ_CREDIT_BUREAU_YEAR"] = all_data["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(all_data["AMT_REQ_CREDIT_BUREAU_YEAR"].mean())

# çok fazla eksik veri içeren sütunları sil - yüzde 60
columns_to_drop = [
    "FLOORSMIN_MODE", "FLOORSMIN_MEDI", "FLOORSMIN_AVG",
    "FLOORSMAX_MODE", "FLOORSMAX_MEDI", "FLOORSMAX_AVG",
    "APARTMENTS_MODE", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "ELEVATORS_AVG", "ELEVATORS_MEDI", "ELEVATORS_MODE",
    "LIVINGAREA_AVG", "LIVINGAREA_MEDI", "LIVINGAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "OWN_CAR_AGE", "TOTALAREA_MODE"
]

all_data = all_data.drop(columns=[col for col in columns_to_drop if col in all_data.columns])


most_freg_col = ["EXT_SOURCE_2",
                 "AMT_ANNUITY", 
                 "AMT_GOODS_PRICE",
                 "NAME_TYPE_SUITE",
                 "DAYS_LAST_PHONE_CHANGE"]

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='most_frequent')  

for key in tqdm(most_freg_col):
    all_data[key] = imputer.fit_transform(all_data[[key]]).ravel()


data2 = pd.DataFrame() 

for key in all_data.keys():
    if all_data[key].dtype == "object": 
        new_key = key + "_cat"
        data2[new_key] = all_data[key].astype("category").cat.codes
        
    else:
        data2[key] = all_data[key]
        
data_ridge = data2


def merge_bureau(col):
    
    bureau_merge = pd.DataFrame(bureau.groupby(["SK_ID_CURR"])[col].agg([np.mean, np.size,
                                                                                   np.sum, np.std]))
    bureau_merge.rename(columns={'mean': str(col) + '_bureau_mean',
                                 'size': str(col) + '_bureau_count',
                                 'sum': str(col)  + '_bureau_sum',
                                 'std': str(col)  + '_bureau_std'    }, inplace=True)
    global data2
    data2 = data2.merge(bureau_merge, how='left', on='SK_ID_CURR')

columns_bur = ["DAYS_CREDIT",
               "AMT_CREDIT_SUM_DEBT",
               "AMT_CREDIT_MAX_OVERDUE"]
for column in columns_bur:
    merge_bureau(column)


def merge_previous(col):
    
    previous_application_merge = pd.DataFrame(previous_application.groupby(["SK_ID_CURR"])[col].agg([np.size]))
    
    previous_application_merge.rename(columns={'size': str(col) + '_previous_count'}, inplace=True)
    
    global data2
    data2 = data2.merge(previous_application_merge, how='left', on='SK_ID_CURR')
    

columns_pre = ["SK_ID_PREV","AMT_CREDIT","AMT_GOODS_PRICE"]
for column in columns_pre:
    merge_previous(column)


# Credit ratios
data2['CREDIT_TO_ANNUITY_RATIO'] = data2['AMT_CREDIT'] / data2['AMT_ANNUITY']
data2['CREDIT_TO_GOODS_RATIO'] = data2['AMT_CREDIT'] / data2['AMT_GOODS_PRICE']
# Income ratios
data2['CREDIT_TO_INCOME_RATIO'] = data2['AMT_CREDIT'] / data2['AMT_INCOME_TOTAL']


columns = [
    "AMT_CREDIT_SUM_DEBT_bureau_mean",
    "AMT_CREDIT_MAX_OVERDUE_bureau_mean",
    "AMT_CREDIT_MAX_OVERDUE_bureau_sum",
    "AMT_CREDIT_SUM_DEBT_bureau_sum",
    "AMT_CREDIT_SUM_DEBT_bureau_std",
    "AMT_INCOME_TOTAL"
]

for column in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data2[column], kde=True, bins=30)

    plt.title(f"Distribution of Column: {column}")
    plt.xlabel("Column Values")
    plt.ylabel("Frequency")
    plt.show()


data2["AMT_CREDIT_SUM_DEBT_bureau_mean"] = np.log(data2["AMT_CREDIT_SUM_DEBT_bureau_mean"])
data2["AMT_CREDIT_MAX_OVERDUE_bureau_mean"] = np.log(data2["AMT_CREDIT_MAX_OVERDUE_bureau_mean"])
data2["AMT_CREDIT_MAX_OVERDUE_bureau_sum"] = np.log(data2["AMT_CREDIT_MAX_OVERDUE_bureau_sum"])

data2["AMT_CREDIT_SUM_DEBT_bureau_sum"] = np.log(data2["AMT_CREDIT_SUM_DEBT_bureau_sum"])
data2["AMT_CREDIT_SUM_DEBT_bureau_std"] = np.log(data2["AMT_CREDIT_SUM_DEBT_bureau_std"])
data2["AMT_INCOME_TOTAL"] = np.log(data2["AMT_INCOME_TOTAL"])


import seaborn as sns
import matplotlib.pyplot as plt

columns = [
    "AMT_CREDIT_SUM_DEBT_bureau_mean",
    "AMT_CREDIT_MAX_OVERDUE_bureau_mean",
    "AMT_CREDIT_MAX_OVERDUE_bureau_sum",
    "AMT_CREDIT_SUM_DEBT_bureau_sum",
    "AMT_CREDIT_SUM_DEBT_bureau_std",
    "AMT_INCOME_TOTAL"
]

for column in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data2[column], kde=True, bins=30)

    plt.title(f"Distribution of Column: {column}")
    plt.xlabel("Column Values")
    plt.ylabel("Frequency")
    plt.show()


data2['app EXT_SOURCE mean'] = data2[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
data2['app EXT_SOURCE std'] = data2[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
data2['app EXT_SOURCE prod'] = data2['EXT_SOURCE_1'] * data2['EXT_SOURCE_2'] * data2['EXT_SOURCE_3']
data2['app EXT_SOURCE_1 * EXT_SOURCE_2'] = data2['EXT_SOURCE_1'] * data2['EXT_SOURCE_2']
data2['app EXT_SOURCE_1 * EXT_SOURCE_3'] = data2['EXT_SOURCE_1'] * data2['EXT_SOURCE_3']
data2['app EXT_SOURCE_2 * EXT_SOURCE_3'] = data2['EXT_SOURCE_2'] * data2['EXT_SOURCE_3']
data2['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = data2['EXT_SOURCE_1'] * data2['DAYS_EMPLOYED']
data2['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = data2['EXT_SOURCE_2'] * data2['DAYS_EMPLOYED']
data2['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = data2['EXT_SOURCE_3'] * data2['DAYS_EMPLOYED']
data2['app EXT_SOURCE_1 / DAYS_BIRTH'] = data2['EXT_SOURCE_1'] / data2['DAYS_BIRTH']
data2['app EXT_SOURCE_2 / DAYS_BIRTH'] = data2['EXT_SOURCE_2'] / data2['DAYS_BIRTH']
data2['app EXT_SOURCE_3 / DAYS_BIRTH'] = data2['EXT_SOURCE_3'] / data2['DAYS_BIRTH']


import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from lofo import LOFOImportance, Dataset, plot_importance
import matplotlib.pyplot as plt

data_ridge = data2.copy()
data_ridge.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = data_ridge.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data_ridge.select_dtypes(include=['object']).columns
imputer_numeric = SimpleImputer(strategy='median')
data_ridge[numeric_cols] = imputer_numeric.fit_transform(data_ridge[numeric_cols])

if len(categorical_cols) > 0:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data_ridge[categorical_cols] = imputer_categorical.fit_transform(data_ridge[categorical_cols])

data_ridge.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in data_ridge.columns]
train_lofo = data_ridge.iloc[:307511, :]

X = train_lofo.drop("TARGET", axis=1)
y = train_lofo["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

model_ridge = RidgeClassifier()
model_ridge.fit(X_train, y_train)
dataset = Dataset(df=train_lofo, target='TARGET', features=[col for col in train_lofo.columns if col != 'TARGET'])
lofo_imp = LOFOImportance(dataset, model=model_ridge, scoring='roc_auc')

importance_df = lofo_imp.get_importance()
plot_importance(importance_df, figsize=(15, 30))


importance_df = lofo_imp.get_importance()

print("Ortalama Importance Mean:", importance_df['importance_mean'].mean())
print("Ortalama Importance Std :", importance_df['importance_std'].mean())

mean_importance = importance_df['importance_mean'].mean()
mean_stddev = importance_df['importance_std'].mean()

columns_to_drop = importance_df[
    (importance_df['importance_mean'] < mean_importance) & 
    (importance_df['importance_std'] < mean_stddev)
]['feature'].tolist()

columns_to_drop = [col for col in columns_to_drop if col in data2.columns]

print(f"Çıkarılacak {len(columns_to_drop)} özellik:")
print(columns_to_drop)

data2 = data2.drop(columns=columns_to_drop)


columns_to_drop = ['LIVE_CITY_NOT_WORK_CITY',
                   'REGION_RATING_CLIENT',
                   'NAME_TYPE_SUITE_cat',
                   'NONLIVINGAPARTMENTS_AVG',
                   'COMMONAREA_AVG',
                   'FLAG_DOCUMENT_13',
                   'EXT_SOURCE_2',
                   'FLOORSMAX_MEDI',
                   'ELEVATORS_MODE',
                   'LIVE_REGION_NOT_WORK_REGION',
                   'WALLSMATERIAL_MODE_Block',
                   'YEARS_BUILD_MODE',
                   'WALLSMATERIAL_MODE_Wooden',
                   'LIVINGAREA_MEDI',
                   'LIVINGAREA_AVG',
                   'LIVINGAREA_MODE',
                   'CODE_GENDER_M',
                   'DAYS_LAST_PHONE_CHANGE',
                   'YEARS_BEGINEXPLUATATION_MEDI',
                   'DAYS_REGISTRATION']

existing_cols_to_drop = [col for col in columns_to_drop if col in data2.columns]
data2 = data2.drop(columns=existing_cols_to_drop)


data2 = data2.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
data_ridge = data2.copy()


from sklearn.impute import SimpleImputer
import numpy as np

data_ridge.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = data_ridge.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data_ridge.select_dtypes(include=['object']).columns

imputer_numeric = SimpleImputer(strategy='median')
data_ridge[numeric_cols] = imputer_numeric.fit_transform(data_ridge[numeric_cols])

if len(categorical_cols) > 0:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data_ridge[categorical_cols] = imputer_categorical.fit_transform(data_ridge[categorical_cols])

ridge_train = data_ridge.iloc[:307511, :]
ridge_test = data_ridge.iloc[307511:, :]


X = ridge_train.drop("TARGET", axis=1)
y = ridge_train["TARGET"]


params = {
    "alpha": 0.10,
    "solver": "auto"
}

# -------------------- CROSS-VALIDATION --------------------
t1 = time.time()
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

score_list = []
test_preds = np.zeros(X.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"[Fold {fold}] Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Train ve validation verileri
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

    # SMOTE ile oversampling
    sm = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Model eğitimi
    model = RidgeClassifier(**params)
    model.fit(X_train_res, y_train_res)

    # Tahmin ve skor
    y_pred = model.predict(X_val_scaled)
    auc = roc_auc_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print(f"ROC AUC Score: {auc:.6f}")
    print(f"Confusion Matrix:\n{cm}\n{'-'*60}")

    score_list.append(auc)
    test_preds[val_idx] = y_pred

    del model, X_train, y_train, X_val, y_val
    gc.collect()

# -------------------- SONUÇLAR --------------------
print(f"\nBest Fold AUC: {np.max(score_list):.6f}")
print(f"Total Time: {(time.time() - t1):.2f} seconds")


X_ridge = ridge_train.drop(["TARGET"], axis = 1)
y_ridge = ridge_train["TARGET"]

def objective(trial):
    
    alpha = trial.suggest_loguniform('alpha', 0.001, 5)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    copy_X = trial.suggest_categorical('copy_X', [True, False])
    max_iter = trial.suggest_int('max_iter', 100, 600)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    score_list = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_ridge, y_ridge), 1):
        X_train, X_val = X_ridge.iloc[train_index], X_ridge.iloc[test_index]
        y_train, y_val = y_ridge.iloc[train_index], y_ridge.iloc[test_index]

        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train))
        X_val_sc = pd.DataFrame(scaler.transform(X_val))

        sm = SMOTE(sampling_strategy=0.6)
        X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train_sc, y_train)

        ridge = RidgeClassifier(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            random_state=1
        )
        ridge.fit(X_train_oversampled, y_train_oversampled)

        y_pred = ridge.predict(X_val_sc)

        score = roc_auc_score(y_val, y_pred)
        score_list.append(score)

    return np.mean(score_list)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best trial:")
best_trial = study.best_trial
print("  Value: {}".format(best_trial.value))
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))


params = {
    "alpha": 0.014054771769335705,
    "fit_intercept": True,
    "copy_X": True,
    "max_iter": 392,
    "solver": "cholesky"
}

t1 = time.time()
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

score_list = []
test_preds = np.zeros(X.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"[Fold {fold}] Train: {len(train_idx)}, Val: {len(val_idx)}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

    sm = SMOTE(sampling_strategy=0.6, random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    model = RidgeClassifier(**params)
    model.fit(X_train_res, y_train_res)

    y_pred_proba = model.decision_function(X_val_scaled)  

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)  
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)

    cm = confusion_matrix(y_val, y_pred_binary)

    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"ROC AUC Score: {auc:.6f}")
    print(f"Confusion Matrix:\n{cm}\n{'-'*60}")

    score_list.append(auc)
    test_preds[val_idx] = y_pred_binary

    del model, X_train, y_train, X_val, y_val
    gc.collect()

print(f"\nBest Fold AUC: {np.max(score_list):.6f}")
print(f"Total Time: {(time.time() - t1):.2f} seconds")


sample_submission.head()


# Test veri setindeki SK_ID_CURR kolonunu doğru türe dönüştürelim
ridge_test['SK_ID_CURR'] = pd.to_numeric(ridge_test['SK_ID_CURR'], errors='coerce', downcast='integer')

# Eğer NaN değerler varsa, onları 0 ile dolduralım veya uygun bir değerle
ridge_test['SK_ID_CURR'].fillna(0, inplace=True)

# 1. NaN ve inf değerlerini temizleyelim
ridge_test.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = ridge_test.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = ridge_test.select_dtypes(include=['object']).columns

# 2. Eksik değerleri dolduralım
imputer_numeric = SimpleImputer(strategy='median')
ridge_test[numeric_cols] = imputer_numeric.fit_transform(ridge_test[numeric_cols])

if len(categorical_cols) > 0:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    ridge_test[categorical_cols] = imputer_categorical.fit_transform(ridge_test[categorical_cols])

# 3. Veriyi ayıralım
X_test = ridge_test.drop("TARGET", axis=1)

# 4. Özellikleri ölçekleyelim
scaler = StandardScaler()
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

# 5. Model parametreleri
params = {
    "alpha": 0.014054771769335705,
    "fit_intercept": True,
    "copy_X": True,
    "max_iter": 392,
    "solver": "auto"  # Ridge regresyonunda solver seçimi
}

# 6. Eğitim verisinde kullanılan parametreler
X = ridge_train.drop("TARGET", axis=1)
y = ridge_train["TARGET"]

# 7. Modeli eğitip, tahminlemeyi yapacağız
t1 = time.time()

# Modeli eğitmek için Stratified KFold kullanıyoruz
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

score_list = []
test_preds = np.zeros(X.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"[Fold {fold}] Train: {len(train_idx)}, Val: {len(val_idx)}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Özellikleri ölçekleme
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

    # Modeli eğitiyoruz
    model = Ridge(**params)
    model.fit(X_train_scaled, y_train)

    # Validasyon seti üzerinde tahminler
    y_pred_proba = model.predict(X_val_scaled)

    # ROC eğrisini hesaplayarak optimal threshold'u belirliyoruz
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)  # TPR - FPR farkını en büyük yapan index
    optimal_threshold = thresholds[optimal_idx]  # Optimal threshold

    # Binary tahminler
    y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)

    # AUC hesaplama
    auc = roc_auc_score(y_val, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_binary)

    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"ROC AUC Score: {auc:.6f}")
    print(f"Confusion Matrix:\n{cm}\n{'-'*60}")

    score_list.append(auc)
    test_preds[val_idx] = y_pred_binary

    del model, X_train, y_train, X_val, y_val
    gc.collect()

print(f"\nBest Fold AUC: {np.max(score_list):.6f}")
print(f"Total Time: {(time.time() - t1):.2f} seconds")

# 8. Test verisi üzerinde tahminleme
model = Ridge(**params)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Modeli eğitip, test verisi üzerinde tahmin yapalım
model.fit(X_train_scaled, y)

# Test verisi üzerinde tahminleme
y_pred_proba_test = model.predict(X_test_scaled)

# Optimal threshold ile binary tahmin
y_pred_binary_test = (y_pred_proba_test > optimal_threshold).astype(int)

# 9. Submission dosyası oluşturma
submission_df = pd.DataFrame({
    'SK_ID_CURR': ridge_test['SK_ID_CURR'],  # Test veri setindeki ID'leri kullanıyoruz
    'TARGET': y_pred_binary_test
})

# 10. Submission dosyasını kaydedelim
submission_df.to_csv('submission_r5.csv', index=False)
print("Submission dosyası oluşturuldu.")


app_train_domain = app_train.copy()
app_test_domain = app_test.copy()


app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# bazı kolonların hedef kolonumuza göre dağılımları
def visualize_new_features(df, target, new_features_list):
    plt.figure(figsize = (8, 6))
    
    for i, feature in enumerate(new_features_list):
        
        plt.subplot(4, 1, i + 1)

        sns.kdeplot(df.loc[df[target] == 0, feature], label = 'target == 0')
        sns.kdeplot(df.loc[df[target] == 1, feature], label = 'target == 1')
    
        plt.title('Distribution of %s by Target Value' % feature)
        plt.xlabel('%s' % feature); plt.ylabel('Density');
    
    plt.tight_layout(h_pad = 2.5)
visualize_new_features(app_train_domain, "TARGET", ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT'])


def merge_dfs(df_main, df_merge):
    df_main = df_main.merge(right=df_merge.reset_index(), how='left', on='SK_ID_CURR')
    return df_main


prev_apps_count = previous_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
previous_app['SK_ID_PREV'] = previous_app['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])

prev_apps_avg = previous_app.groupby('SK_ID_CURR')[previous_app.select_dtypes(include='number').columns].mean()
prev_apps_avg.columns = ['P_' + col for col in prev_apps_avg.columns]

app_train = merge_dfs(app_train, prev_apps_avg)
app_test = merge_dfs(app_test, prev_apps_avg)


bureau_avg = bureau.groupby('SK_ID_CURR')[bureau.select_dtypes(include='number').columns].mean()
bureau_avg['BUREAU_COUNT'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
bureau_avg.columns = ['B_' + col for col in bureau_avg.columns]

app_train = merge_dfs(app_train, bureau_avg)
app_test = merge_dfs(app_test, bureau_avg)


install_count = install_payments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
install_payments['SK_ID_PREV'] = install_payments['SK_ID_CURR'].map(install_count['SK_ID_PREV'])

install_avg = install_payments.groupby('SK_ID_CURR')[install_payments.select_dtypes(include='number').columns].mean()
install_avg.columns = ['I_' + col for col in install_avg.columns]

app_train = merge_dfs(app_train, install_avg)
app_test = merge_dfs(app_test, install_avg)


prev_credit_count = credit_card[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
credit_card['SK_ID_PREV'] = credit_card['SK_ID_CURR'].map(prev_credit_count['SK_ID_PREV'])

avg_credit_bal = credit_card.groupby('SK_ID_CURR')[credit_card.select_dtypes(include='number').columns].mean()
avg_credit_bal.columns = ['CC_B_' + col for col in avg_credit_bal.columns]

app_train = merge_dfs(app_train, avg_credit_bal)
app_test = merge_dfs(app_test, avg_credit_bal)


def find_unique_features(train_set, test_set):
    unique_features = set(test_set.columns) - set(train_set.columns) - {"TARGET"}
    return unique_features

def drop_unique_features(test_set, unique_features):
    test_set = test_set.drop(columns=unique_features)
    return test_set


unique_features_test_set = find_unique_features(app_train_domain, app_test_domain)
app_test_domain = drop_unique_features(app_test_domain, unique_features_test_set)

labels = app_train['TARGET']

app_train_domain.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
app_test_domain.drop(['SK_ID_CURR'], axis=1, inplace=True)

num_features = app_train_domain.select_dtypes(include=['number']).columns
cat_features = app_train_domain.select_dtypes(include=['object']).columns


from sklearn.compose import ColumnTransformer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)


train_processed = preprocessor.fit_transform(app_train_domain)

feature_names = []
for name, transformer, columns in preprocessor.transformers_:
    if hasattr(transformer, 'get_feature_names_out'):
        feature_names.extend(transformer.get_feature_names_out(columns))
    else:
        feature_names.extend(columns)

train_processed = pd.DataFrame(train_processed, columns=feature_names)
test_processed = preprocessor.transform(app_test_domain)
test_processed = pd.DataFrame(test_processed, columns=feature_names)


from sklearn.metrics import roc_curve, auc, roc_auc_score

def cross_validate_model(model, X, y, cv):
    auc_scores = []
    threshold_list = []
    
    for fold, (train_index, val_index) in enumerate(cv.split(X, y)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        if isinstance(model, CatBoostClassifier):
            model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), verbose=False)
            y_val_pred_prob = model.predict_proba(X_val_cv)[:, 1]
        else:
            model.fit(X_train_cv, y_train_cv)
            y_val_pred_prob = model.predict(X_val_cv)

        fpr, tpr, thresholds = roc_curve(y_val_cv, y_val_pred_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        threshold_list.append(optimal_threshold)
        y_val_pred_binary = (y_val_pred_prob >= optimal_threshold).astype(int)
        auc_score = roc_auc_score(y_val_cv, y_val_pred_prob)
        auc_scores.append(auc_score)

        print(f"[Fold {fold + 1}] AUC: {auc_score:.4f}, Optimal Threshold: {optimal_threshold:.4f}")

    print(f"\nMean AUC: {np.mean(auc_scores):.4f}")
    print(f"Average Optimal Threshold: {np.mean(threshold_list):.4f}")
    return np.mean(auc_scores), threshold_list


from catboost import CatBoostClassifier
stk_fold = StratifiedKFold(n_splits=5, 
                           shuffle=True,
                           random_state=42)
cb_model = CatBoostClassifier(verbose=0, random_state=42)

mean_auc, threshold_list = cross_validate_model(cb_model, train_processed, labels, stk_fold)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def lofo_analysis_single_pass(model, X, y, test_size=0.3, random_state=42):
    # Eğitim ve doğrulama ayrımı (tek seferlik)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    auc_scores = {}

    for feature in tqdm(X.columns, desc="Processing Features"):
        X_train_wo = X_train.drop(columns=[feature])
        X_val_wo = X_val.drop(columns=[feature])

        model.fit(X_train_wo, y_train)
        y_pred_prob = model.predict_proba(X_val_wo)[:, 1]

        auc_score = roc_auc_score(y_val, y_pred_prob)
        auc_scores[feature] = auc_score

    return auc_scores

cb_model = CatBoostClassifier(verbose=0, random_state=42)

feature_importances = lofo_analysis_single_pass(cb_model, train_processed, labels)

mean_auc = np.mean(list(feature_importances.values()))
std_auc = np.std(list(feature_importances.values()))

filtered_features = {
    feature: auc for feature, auc in feature_importances.items()
    if auc < mean_auc and abs(auc - mean_auc) < std_auc
}

X_filtered = train_processed.drop(columns=filtered_features.keys())

print(f"\nFiltered out features ({len(filtered_features)}): {list(filtered_features.keys())}")
print(f"New shape after removal: {X_filtered.shape}")

sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
features = [x[0] for x in sorted_features]
importances = [x[1] for x in sorted_features]

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('AUC Score without Feature')
plt.title('LOFO Analysis (Single Pass)')
plt.gca().invert_yaxis()
plt.show()


features_to_remove = [
    'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'FLAG_EMP_PHONE', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START', 'APARTMENTS_AVG', 'COMMONAREA_AVG',
    'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'LANDAREA_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAREA_AVG',
    'ELEVATORS_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAREA_MODE',
    'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
    'COMMONAREA_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'LANDAREA_MEDI', 'LIVINGAREA_MEDI',
    'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_CONTRACT_TYPE_Cash loans', 'NAME_CONTRACT_TYPE_Revolving loans'
]

features_to_remove = [feat for feat in features_to_remove if feat in train_processed.columns]
train_processed_filtered = train_processed.drop(columns=features_to_remove)
test_processed_filtered = test_processed.drop(columns=features_to_remove,
                                              errors='ignore')



import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def objective(trial):
    # Hiperparametre aralıkları
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "verbose": 0,
        "random_state": 42
    }

    model = CatBoostClassifier(**params)

    mean_auc, _ = cross_validate_model(model, train_processed_filtered, labels, stk_fold)

    return mean_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("\nBest Parameters:")
print(study.best_params)
print(f"Best ROC AUC Score: {study.best_value:.4f}")


best_params = study.best_params
best_params.update({"verbose": 0, "random_state": 34})

best_cb_model = CatBoostClassifier(**best_params)
mean_auc_best, threshold_list_best = cross_validate_model(best_cb_model,
                                                          train_processed_filtered,
                                                          labels,
                                                          stk_fold)



from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

# En iyi parametreler
best_params = {"iterations" : 1000,
"learning_rate" : 0.05,
                              "depth" : 7,
                              "l2_leaf_reg" : 40,
                              "bootstrap_type" : 'Bernoulli',
                              "subsample" : 0.7,
                              "scale_pos_weight" : 5,
                              "eval_metric" : 'AUC',
                              "metric_period" : 50,
                              "od_type" : 'Iter',
                              "od_wait" : 45,
                              "random_seed" : 17,
                              "allow_writing_files" : False
}

# Modeli eğit
final_model = CatBoostClassifier(**best_params)
final_model.fit(train_processed_filtered, labels)

# Test verisi için olasılık tahminleri
test_preds_proba = final_model.predict_proba(test_processed_filtered)[:, 1]

# Threshold'u ortalama olarak al
optimal_threshold = np.mean(threshold_list)  # Burada threshold_list kullanılıyor
test_preds_binary = (test_preds_proba >= optimal_threshold).astype(int)

# Submission dosyasını oluştur
submission = pd.DataFrame({
    'SK_ID_CURR': app_test['SK_ID_CURR'],
    'TARGET': test_preds_binary
})

submission.to_csv('cat10.csv', index=False)


#best_cb_model.fit(train_processed, labels)
#explainer = shap.TreeExplainer(best_cb_model)
#shap_values = explainer.shap_values(train_processed)




