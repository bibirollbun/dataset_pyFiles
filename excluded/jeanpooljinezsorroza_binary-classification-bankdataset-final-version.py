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


"""
import sys
!{sys.executable} -m pip install lightgbm
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from functools import partial

# Columns tranformes libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder

# Split data and scores
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


# Avoid warning messages
import warnings
import logging
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', message="'Threading' parallel backend is not supported")
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sklearnex').setLevel(logging.ERROR)
logging.getLogger("sklearnex").setLevel(logging.WARNING)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
df_full = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')


# Visualization of the data
print(20*'****')
print('DIMENSION OF THE DATA IMPORTED: \n')
print('df_train  = ',df_train.shape)
print('df_test   = ',df_test.shape)
print('df_full   = ',df_full.shape)
print(20*'****')
print('df_train: \n')
print(df_train.head(),'\n')
print(20*'****')
print('df_train DESCRIBE:')
print(df_train.describe())
print(20*'****')


# ------------------------ Columns --------------------------
# ORDINAL variable 
ordinal_col  = ['education']

# Variables categóricas para OneHotEncoder
onehot_cols  = ['job', 'marital', 'contact', 'month', 'poutcome']

# Variables binarias
bin_cols = ["default", "housing", "loan"]

# Variables categoricas
category_cols = ordinal_col+onehot_cols+bin_cols

# Variables numéricas
numeric_cols = ['age', 'balance', 'duration', 
                'campaign', 'day', 'pdays', 
                'previous']
# TARGET
target =["y"]


#-------------------------------------------------------------------------------------------------------------
# Funtions for calculating ratio for columns
def col_ratio(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
#-------------------------------------------------------------------------------------------------------------
# Funtions for calculating our MI-Scores and plotting then
from sklearn.feature_selection import mutual_info_regression
# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize(sort=True)
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
    
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
#-------------------------------------------------------------------------------------------------------------
# Funtion for plotting correlation matrix
def corr_matrix(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]):
        df_encoded[col], _ = df_encoded[col].factorize(sort=True)
    corr = df_encoded.corr().abs()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)
    sns.clustermap(corr, cmap="vlag", annot=False, figsize=(14,12))
    plt.show()
#-------------------------------------------------------------------------------------------------------------
# Funtion for splitting data in X and y
def split_train_test(X, y):
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,     # 10% for testing, 90% for training
        random_state=0,
        stratify=y         # maintains class ratio in both sets :contentReference[oaicite:1]{index=1}
    )
    return X_train, X_test, y_train, y_test
#----------------------------------------------------------------------------------------------
def output(df, pipeline):
    y_pred_prob = pipeline.predict_proba(df)[:, 1]
    output=pd.DataFrame({'id':df.index, 'y':y_pred_prob})
    output.to_csv('submission.csv', index=False)
    output.head()
    return output


# y have values of yes or no in our 'df_full' so we need to convert it on 
# 1 and 0 before the merge of df_train and df_full.
df_full["y"] = df_full["y"].fillna(df_full["y"].mode()[0]).str.lower().map({"yes": 1, "no": 0}).astype(int)
train_full = pd.concat([df_train, df_full], axis=0, ignore_index=True)
print('MERGE OF df_train and df_full == train_full: ',train_full.shape)

# That value 7.302 means that in your dataset 
# the majority class (0 = NO) is approximately 
# 7 times more frequent than the minority class (1 = YES).
num_neg = (train_full.y == 0).sum()
num_pos = (train_full.y == 1).sum()
scale_pos_weight = (num_neg / num_pos).round().astype(int)
print('scale_pos_weight = ',scale_pos_weight)


# Null data and unique values
print('train_full SHAPE: ',train_full.shape)
print('train_full COLUMNS: \n\nCOLUMN  |||  DTYPE  |||  NULL-COUNT  |||  UNIQUE VALUES\n')
for i in train_full.columns:
    print(f'"{i}"  |||  dtype:{train_full[i].dtypes}  |||  Null-count:{train_full[i].isnull().sum()}  |||  unique_values:{train_full[i].nunique()}')
print(20*'****')
# Duplicated Data
duplicated = train_full.duplicated().sum()
print('Sum of duplicated data in the train_full dataset: ', duplicated)
print('\nNull values for columns:')
train_full.isnull().sum().sort_values(ascending=False)


for col in train_full.columns:
    col_ratio(train_full, col)


# Split data in X and y
X = train_full.drop(['y'],axis=1)
y = train_full.y

# Mask with a 30% of the data
X_sample = X.sample(n=238_563, random_state=0)
y_sample = y.sample(n=238_563, random_state=0)

scores=make_mi_scores(X_sample, y_sample)
plot_mi_scores(scores)


mi_series = pd.Series(scores, index=X_sample.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)
print(mi_series_sorted)


# df_sample for a fast visulization of the data with a 30% of the train_full
df_sample = train_full.sample(n=238_563, random_state=0)
corr_matrix(df_sample)


# Distribution of the target
sns.countplot(x="y", data=train_full)
plt.title("Distribución del Target")
plt.show()


# Categorias vs Target
n_cols = 3  # cantidad de gráficos por fila
# definir grid (ej: 2 columnas por fila)
n_rows = math.ceil(len(category_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()  # aplanar para indexar fácil

for i, col in enumerate(category_cols):
    order = train_full[col].value_counts().index
    sns.countplot(x=col, hue="y", data=train_full, order=order, ax=axes[i])
    axes[i].set_title(f"{col} vs Target")
    axes[i].tick_params(axis="x", rotation=45)

# borrar ejes vacíos si sobran
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Numericas vs Target
n_rows = math.ceil(len(numeric_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(
        data=train_full,
        x=col,
        hue="y",
        bins=50,
        kde=True,
        ax=axes[i],
        element="step",    # barras más limpias
        stat="density",    # normalizar densidad
        common_norm=False  # para que cada clase se escale separada
    )
    axes[i].set_title(f"Distribución de {col} según Target")

# eliminar ejes vacíos si sobran
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Trarget Rate for Categories
n_rows = math.ceil(len(category_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(category_cols):
    cat_rate = train_full.groupby(col)["y"].mean().sort_values(ascending=False)
    sns.barplot(x=cat_rate.values, y=cat_rate.index, ax=axes[i], palette="viridis")
    axes[i].set_title(f"Proporción de y=1 por {col}")
    axes[i].set_xlabel("Proporción de y=1")
    axes[i].set_ylabel(col)

# eliminar ejes vacíos si sobran
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------
# ---- 1. Temporadas (ejemplo hemisferio norte: invierno = dic-ene-feb)
def month_season_codification(df, col="month"):
    df = df.copy()
    # Convert months to numbers (jan=1, feb=2, ..., dec=12)
    df[col] = df[col].astype('category')
    df[f"{col}_num"] = df[col].cat.codes.astype(float)
    
    # Codificación cíclica
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[f"{col}_num"] / 12)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[f"{col}_num"] / 12)
    
    def season_mapper(m):
        if m in [12, 1, 2]:
                return 0 #"winter"
        elif m in [3, 4, 5]:
                return 1 #"spring"
        elif m in [6, 7, 8]:
                return 2 #"summer"
        else:
                return 3 #"autumn"
        
    df["season"] = df[f"{col}_num"].apply(season_mapper).astype(int)
    # ---- 2. ¿Es fin de semana?
    df["is_weekend"] = df["day"].isin([6, 7]).astype(int)
    # ---- 3. ¿Comienzo o fin de mes?
    df["is_begin_month"] = (df["day"] <= 5).astype(int)
    df["is_end_month"] = (df["day"] >= 25).astype(int)
    
    return df.drop(columns=['month', 'month_num'])
    
def day_codification(df, col="day"):
    df = df.copy()
    max_day = df[col].max()
    # Codificación cíclica
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_day)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_day)
    
    return df.drop(columns=[col])
#----------------------------------------------------------------------------------------------
def winsorize_numeric(df, winsor_cols=None, lower=0.01, upper=0.99):
    df = df.copy()
    for col in winsor_cols:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def categorize_numeric(df):
    df = df.copy()
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"],
                                bins=[0, 25, 40, 60, 100],
                                labels=["joven", "adulto_joven", "adulto", "mayor"])
    if "balance" in df.columns:
        df["balance_category"] = pd.cut(df["balance"],
                                       bins=[-float("inf"), 0, 500, 2000, float("inf")],
                                       labels=["negativo", "bajo", "medio", "alto"])
    if "duration" in df.columns:
        df["duration_category"] = pd.cut(df["duration"],
                                        bins=[0, 60, 180, 600, float("inf")],
                                        labels=["corta", "media", "larga", "muy_larga"])
    return df
#----------------------------------------------------------------------------------------------
def new_features_num_cols(df):
    df = df.copy()
    # Evitar división entre cero sumando +1 en el denominador
    df["balance_per_duration"] = df["balance"] / (1 + df["duration"])
    df["balance_per_campaign"] = df["balance"] / (1 + df["campaign"])
    df["balance_per_age"] = df["balance"] / (1 + df["age"])
    #...

    # Diferencia simple
    df["duration_minus_campaign"] = df["duration"] - df["campaign"]
    #...
    
    # Contacto total
    df['total_contact']= df['campaign'] + df['previous']
    df['recent_contact_flag'] = [1 if x < 30 else 0 for x in df['pdays']]
    df["pdays"] = [0 if x==-1 else x for x in df["pdays"]]
    df['contact_intensity'] = df['total_contact'] / (1+df['pdays'])
    #...
    
    # Log (se usa log1p para evitar problemas con ceros o negativos)
    df["balance_log"] = np.log1p(df["balance"].clip(lower=0))
    df["duration_log"] = np.log1p(df["duration"].clip(lower=0))
    df["campaign_log"] = np.log1p(df["campaign"].clip(lower=0))
    #...
    
    # Raíz cuadrada (otra forma de suavizar asimetría)
    df["balance_sqrt"] = np.sqrt(df["balance"].clip(lower=0))
    df["duration_sqrt"] = np.sqrt(df["duration"].clip(lower=0))
    df["campaign_sqrt"] = np.sqrt(df["campaign"].clip(lower=0))
    #...

    # Si quieres asegurarte de no tener infinitos o NaN (por divisiones)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
    
def new_features_bin_cols(df):
    df['debt_exposure_ratio'] = (
    df['loan'].eq('yes').astype(int) +
    df['housing'].eq('yes').astype(int) +
    df['default'].eq('yes').astype(int)
    ) / (1 + df["balance"])
    #...
    
    # Si quieres asegurarte de no tener infinitos o NaN (por divisiones)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
#----------------------------------------------------------------------------------------------
def frequency_encoding(df, cols=['job', 'marital']):
    df = df.copy()
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)
    return df.drop(columns=cols)
#----------------------------------------------------------------------------------------------


df_encoded = train_full.copy()

print('df_encoded SHAPE: ', df_encoded.shape)
df_encoded.head()


# New features
df_encoded = new_features_num_cols(df_encoded)
df_encoded = new_features_bin_cols(df_encoded)

# time encoding
df_encoded = month_season_codification(df_encoded)
df_encoded = day_codification(df_encoded)

# frequenty encoding
df_encoded = frequency_encoding(df_encoded) 

# categorize coding
df_encoded = categorize_numeric(df_encoded)

# winsorize funtion for cutting outliers
winsor_cols = ["balance", "duration", "campaign", "pdays", "previous"]
df_encoded = winsorize_numeric(df_encoded, winsor_cols)

print('df_encoded: ', df_encoded.shape)
df_encoded.head()


df_encoded.columns


df_encoded.dtypes


# Split data in X and y
X_encoded = df_encoded.drop(['y'],axis=1)
y_encoded = df_encoded.y

# Mask with a 30% of the data
X_sample_encoded = X_encoded.sample(n=238_563, random_state=0)
y_sample_encoded = y_encoded.sample(n=238_563, random_state=0)

scores=make_mi_scores(X_sample_encoded, y_sample_encoded)
plot_mi_scores(scores)


mi_series = pd.Series(scores, index=X_sample_encoded.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)
print(mi_series_sorted)


# df_sample for a fast visulization of the data with a 30% of the train_full
df_sample_encoded = df_encoded.sample(n=238_563, random_state=0)
corr_matrix(df_sample_encoded)


# -------------- Order of the ordinal encoder ---------------
education_order = ['unknown', 'primary', 'secondary', 'tertiary']
age_order       = ["joven", "adulto_joven", "adulto", "mayor"]
balance_order   = ["negativo", "bajo", "medio", "alto"]
duration_order  = ["corta", "media", "larga", "muy_larga"]
ordinal_orders  = [education_order, age_order, balance_order, duration_order]

# ------------------- FunctionTranformers -------------------
features_encoder_num = FunctionTransformer(new_features_num_cols, validate=False)
features_encoder_bin = FunctionTransformer(new_features_bin_cols, validate=False)

freq_coder = FunctionTransformer(frequency_encoding , validate=False)
categorizer = FunctionTransformer(categorize_numeric, validate=False)

month_season_coder = FunctionTransformer(month_season_codification, validate=False)
day_coder   = FunctionTransformer(day_codification, validate=False)
#---------------------- Winsorize columns ----------------------
winsorizer  = FunctionTransformer(
    func     = winsorize_numeric,
    kw_args  = {'winsor_cols': winsor_cols},
    validate = False
)


# ----------------------- Transformers ----------------------- 
# Ordinal transformer
ordinal_pipeline = Pipeline(steps=[
    ('categorizer', categorizer),   # si quieres categorizar primero
    ('ordinal', OrdinalEncoder(
        categories=[education_order],#<--- education_order es tipo list()
        handle_unknown="use_encoded_value", 
        unknown_value=-1
    ))
])

# OneHot transformer
onehot_transformer = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=True
)

# Binary columns
binary_pipeline = Pipeline(steps=[
    ('feat_bin', features_encoder_bin),   # si quieres categorizar primero
    ('ordinal', OrdinalEncoder(
    handle_unknown='use_encoded_value', 
    unknown_value=-1
     ))
])

# Numeric transformer
numeric_pipeline = Pipeline(steps=[
    ('feat_num', features_encoder_num),
    ('winsor', winsorizer), 
    ('scaler', StandardScaler()) 
])

# Date time transformer
date_pipeline = Pipeline(steps=[
    ('month_coder', month_season_coder),
    ('day_coder', day_coder)
])
# ------------------- ColumnTransformer -------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_pipeline, ordinal_col),
        ('onehot', onehot_transformer, onehot_cols),
        ('bin', binary_pipeline, (numeric_cols+bin_cols)),
        ('numeric', numeric_pipeline, numeric_cols),
        ('freq_enc', freq_coder, ['job', 'marital']),
        ('dates', date_pipeline, ['day', 'month'])
    ],
    remainder='drop'
)


df_final = train_full.copy()

# SPLITTING DATA IN X AND y
X_final=df_final.drop(['y'], axis=1)
y_final=df_final.y

# Using function for Splitting data
X_train_final, X_test_final, y_train_final, y_test_final=split_train_test(X_final, y_final)


param_distributions_xgb = {
    'classifier__n_estimators':[4500],
    'classifier__max_depth': [9],
    'classifier__learning_rate': [0.001],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.4],
    'classifier__min_child_weight': [12],
    'classifier__gamma': [0]
    #Mejor score: 0.9687 (APROX)
}

param_distributions_lgbm= {
    'classifier__n_estimators': 1000,
    'classifier__learning_rate': 0.01,
    'classifier__num_leaves': 100,
    'classifier__subsample': 0.1,
    'classifier__colsample_bytree': 0.5,
    'classifier__reg_lambda': 0.0,
    'classifier__min_child_weight': 15
    #Mejor score: 0.9638491252397774
}

param_distributions_rf= {
    'classifier__n_estimators': 400,
    'classifier__max_depth': 20,
    'classifier__min_samples_split': 10,
    'classifier__min_samples_leaf': 3,
    'classifier__max_features': 'sqrt',
    'classifier__bootstrap': True,
    'classifier__class_weight': 'balanced',
    'classifier__criterion': 'gini'
    #Mejor score: 0.9604631047848057
}


XGB_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier'  , XGBClassifier(
        random_state=0,
        n_jobs=1,
        objective='binary:logistic',    
        scale_pos_weight=scale_pos_weight,           
        use_label_encoder=False,  
        enable_categorical=False,
        eval_metric='auc',
        n_estimators=4500,
        max_depth=9,
        learning_rate=0.001,
        subsample=0.8,
        colsample_bytree=0.4,
        min_child_weight=12,
        gamma=0,
        verbosity=1
    ))
])

LGBM_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=0,
        n_jobs=1,
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        boosting_type='gbdt',
        metric='auc',
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=100,
        subsample=0.1,
        colsample_bytree=0.5,
        reg_lambda=0.0,
        min_child_weight=15,
        verbose=1
    ))
])

RFC_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=0,
        n_jobs=1,
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        criterion='gini'
    ))
])


"""
# Base-estimators
base_estimators = [    
    ('xgb', XGB_pipeline),
    ('lgbm', LGBM_pipeline),
    ('rf', RFC_pipeline)
]

# Meta-modelo (stacker)
meta_model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced')

# Stack
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_model,
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)
"""


"""
final_stack = stack.fit(X_train_final, y_train_final)
"""


"""
output(df_test, final_stack)
"""


"""
# Save the trained model
joblib.dump(final_stack, "final_stack.pkl")

# Load the trained model
loaded_model = joblib.load("final_stack.pkl")

# Use it...
# output(df_test, loaded_model)
"""

