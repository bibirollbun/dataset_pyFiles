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
%matplotlib inline
import seaborn as sns

from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold , KFold , cross_val_predict
from sklearn.metrics import roc_auc_score


import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')


train_df = train_data.copy()
test_df = test_data.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Duplicated #####################")
    print(dataframe.duplicated().sum())


def grab_cols(df):

    cat_cols = [col for col in df.columns if df[col].dtype in ['O','category']]

    num_cols = [col for col in df.columns if (df[col].dtype in ["int64", "float64"])]

    num_but_cat = [col for col in df.columns if (df[col].nunique() < 15) & (col in num_cols)]

    cardinals = [col for col in df.columns if (df[col].nunique() > 25) & (col in cat_cols)]


    num_cols = [i for i in num_cols if i not in num_but_cat]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cardinals]

    # SonuÃ§larÄ± yazdÄ±rma
    print('#########cat_cols#########')
    print(cat_cols)
    print('#########num_cols#########')
    print(num_cols)
    print('#########cardinals#########')
    print(cardinals)
    print('#########num_but_cat#########')
    print(num_but_cat)

    return cat_cols, num_cols, cardinals, num_but_cat


# sample_submission.head(2)


train_df.head(2)


test_df.head(2)


print(train_df.shape)
print(test_df.shape)


train_df['is_test'] = 0  
test_df['is_test'] = 1   
df = pd.concat([train_df, test_df], ignore_index=True)


check_df(df)


cat_cols, num_cols, cardinals, num_but_cat = grab_cols(train_df)


from statsmodels.graphics.mosaicplot import mosaic

colors = {'0': '#cb4060', '1': '#4bbeed'}


mosaic(train_df, 
       ['loan_status'], 
       gap=0.01, 
       title='Loan Approval Ratio', 
       properties=lambda key: {'color': colors[str(key[0])]})

print('Not approved : ' ,train_df["loan_status"].value_counts().values[0] , '\napproved : ' ,  train_df["loan_status"].value_counts().values[1])
plt.show()



print("\nðŸ§¾ **Column-Wise Analysis for Numeric Cols:**\n")
for col in num_cols:
    print(f"\nðŸ”¹ **{col}**")
    print(f" - Data Type: {df[col].dtype}")
    print(f" - Unique Values: {df[col].nunique()}")
    print(f" - Sample Values: {df[col].unique()[:5]}")
    print(f" - Null Values: {df[col].isnull().sum()}")
    if df[col].dtype in ['int64', 'float64']:
        print(f" - Mean: {df[col].mean():.2f}, Std Dev: {df[col].std():.2f}, Min: {df[col].min()}, Max: {df[col].max()}")



df[num_cols].describe().T


# def plot_distribution(train, feature, hue="set", palette=None):
#     data_df = train.copy()
#     data_df['set'] = 'train'  # 'train' etiketini ekliyoruz

#     f, axes = plt.subplots(1, 2, figsize=(14, 5))
#     for i, s in enumerate(data_df[hue].unique()):
#         selection = data_df.loc[data_df[hue] == s, feature]
#         # Filter 'selection' to include only the central 95% of the data
#         q_025, q_975 = np.percentile(selection, [2.5, 97.5])
#         selection_filtered = selection[(selection >= q_025) & (selection <= q_975)]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=FutureWarning)
#             sns.histplot(selection_filtered, color=palette[i], ax=axes[0], label=s)
#             # sns.boxplot(x=hue, y=feature, data=data_df, palette=palette, ax=axes[1])
#             sns.violinplot(data = data_df , x = hue , y = feature ,inner = "quart"  , palette=palette, ax=axes[1] )
    
#     axes[0].set_title(f"Distribution of {feature} in Train Set")
#     axes[1].set_title(f"Boxplot of {feature} in Train Set")
#     axes[0].legend()
#     axes[1].legend()
#     plt.show()

# color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]
# for feature in num_cols:
#     if feature != 'id' : 
#         plot_distribution(train_df, feature, palette=color_list)


fig = plt.figure(figsize=(15,15))
for index,column in enumerate(num_cols):
    plt.subplot(4,3,index+1)
    sns.distplot(x = df.loc[:, column], color = '#7008f5')
    plt.title(column, size = 12)
    fig.tight_layout()
    plt.grid(True)
plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):

        outliers = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit )]
        print(f"Column: {col_name}\nOutliers Detected: {len(outliers)}\n-------------------------")
        return True
    else:
        print(f"Column: {col_name}\nNo Outliers\n-------------------------")
        return False





fig, ax = plt.subplots(nrows=len(num_cols), figsize=(10, 1.5 * len(num_cols)))

for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], palette="plasma", ax=ax[i], orient="v")
    ax[i].set_title(col)

plt.tight_layout()
plt.show()


yalan1 = train_df[(train_df['person_age'] - train_df['person_emp_length']) < 14]
yalan2 = train_df[(train_df['person_age'] - train_df['cb_person_cred_hist_length'])< 5]

# bunlarÄ± bÄ±rakÄ±yorum model bu farktan biÅŸeyler Ã¶ÄŸrenebilir belki diye Ã¶zellik olarak ekleyeceÄŸim.


train_df.isnull().sum()


test_df.isnull().sum()


df['cb_person_default_on_file'] = df['cb_person_default_on_file'].replace({'Y': 1, 'N': 0})


sns.set_style('whitegrid')
plt.figure(figsize=(20,12), dpi=150)

plt.subplot(221)
sns.countplot(data=train_df, x='person_home_ownership', palette='pastel');
plt.yscale('log')

plt.subplot(222)
sns.countplot(data=train_df, x='loan_intent', palette='pastel');

plt.subplot(223)
sns.countplot(data=train_df, x='loan_grade', palette='pastel');
plt.yscale('log')

plt.subplot(224)
sns.countplot(data=train_df, x='cb_person_default_on_file', palette='pastel');

plt.tight_layout()
plt.show()


df.head(1)


df['person_age'].describe().T


df["work_start_age"] = df['person_age'] - df['person_emp_length']
df['credit_start_age'] = df['person_age'] - df['cb_person_cred_hist_length']
df["age_cat"] =  pd.cut(df['person_age'] , [0,30,40,50,70] , labels=["genc" ,"orta_yasli" , "orta_ust" , "yasli"] )

df['loan_income_ratio'] =  np.where(df['person_income'] == 0, 0, df['loan_amnt'] / df['person_income'])

df['income_per_year'] = np.where(df['person_emp_length'] == 0, 0, df['person_income'] // df['person_emp_length'])

df['debt_to_credit_ratio'] = np.where(df['cb_person_cred_hist_length'] == 0, 0, df['loan_amnt'] / df['cb_person_cred_hist_length'])

df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']

df['int_to_loan_ratio'] = np.where(df['loan_amnt'] == 0, 0, df['loan_int_rate'] / df['loan_amnt'])


df["income_cat"] = pd.cut(
    df['person_income'], 
    bins=[-float('inf'), 42000, 58000, 75000, 150000 , float('inf')], 
    labels=["asiri_dusuk", "dusuk", "orta", "yuksek" , "cok_yuksek"])

df['risk_flag'] = (np.where((df['cb_person_default_on_file'] == 1) & (df['income_cat'].isin(['asiri_dusuk', 'dusuk', 'cok_yuksek'])), 1, 0))


df.isnull().sum()


cat_cols, num_cols, cardinals, num_but_cat = grab_cols(df)


df2= df.copy()


cat_cols, num_cols, cardinals, num_but_cat = grab_cols(df2)


for_scale = [col for col in num_cols if col != 'id']


from sklearn.preprocessing import RobustScaler


rs = RobustScaler()

transformed_data = rs.fit_transform(df2[for_scale])

df2[for_scale] = transformed_data


from sklearn.preprocessing import LabelEncoder ,OrdinalEncoder

for_label = ['loan_intent','person_home_ownership','age_cat']
for_ordinal = ['income_cat','loan_grade']
for_ohe = ['cb_person_default_on_file' , 'risk_flag']


df2 = pd.get_dummies(df2, columns=for_ohe, drop_first=True, dtype=int)

label_encoder = LabelEncoder()
for col in for_label:
    df2[col] = label_encoder.fit_transform(df2[col])

income_cat_categories = ['asiri_dusuk', 'dusuk', 'orta', 'yuksek', 'cok_yuksek']
loan_grade_categories = ['G', 'F', 'E', 'D', 'C', 'B', 'A']  # tersten sÄ±ralÄ±yorum.

# Create an OrdinalEncoder object
ordinal_encoder = OrdinalEncoder(categories=[income_cat_categories, loan_grade_categories])

df2[for_ordinal] = ordinal_encoder.fit_transform(df2[for_ordinal])


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


train_df = df2[df2['is_test'] == 0].drop('is_test', axis=1)
test_df = df2[df2['is_test'] == 1].drop(['is_test', 'loan_status'], axis=1)

X_train = train_df.drop('loan_status', axis=1)
y_train = train_df['loan_status']

X_test = test_df

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


print(train_df.shape)
print(test_df.shape)


def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
    }

    # LightGBM Dataset
    train_data = lgb.Dataset(X_train_split, label=y_train_split)
    val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data)

    # Modeli eÄŸit
    gbm = lgb.train(param, train_data, valid_sets=[train_data, val_data])

    # Tahmin yap ve AUC hesapla
    y_pred = gbm.predict(X_val_split)
    auc = roc_auc_score(y_val_split, y_pred)

    return auc


# Optuna Ã§alÄ±ÅŸtÄ±r
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# En iyi parametreler
best_params = study.best_params
print("Best Parameters:", best_params)

# LightGBM'i en iyi parametrelerle eÄŸit
final_gbm = lgb.LGBMClassifier(**best_params)
final_gbm.fit(X_train, y_train)

# Test setinde tahmin yap
y_test_pred = final_gbm.predict_proba(X_test)[:, 1]

# SonuÃ§larÄ± kaydetmek iÃ§in bir DataFrame oluÅŸtur
submission = pd.DataFrame({'id': test_df['id'], 'loan_status': y_test_pred})
submission.to_csv('submission.csv', index=False)





