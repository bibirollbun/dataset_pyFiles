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


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#Intel Extension for Scikit‑learn (sklearnex) 
#dramatically improves KNN performance, up to 100x faster
from sklearnex import patch_sklearn
patch_sklearn()

import warnings
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sklearnex').setLevel(logging.ERROR)
logging.getLogger("sklearnex").setLevel(logging.WARNING)


df_train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


def get_outlier_bounds(df, cols):
    bounds = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)
    return bounds

def apply_outlier_bounds(df, bounds):
    df_copy = df.copy()
    for col, (lower, upper) in bounds.items():
        df_copy[col] = np.where(df_copy[col] < lower, lower, df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper, upper, df_copy[col])
    return df_copy

# check the outliers of the dataset
def outliers(df, df_cols):
    df_copy = df.copy()
    # Box plots for outlier visualization
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_cols):
        plt.subplot(len(df_cols) // 3 + 1, 3, i + 1)
        sns.boxplot(y=df_copy[col])
        plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate IQR and identify outliers for each numerical column
    outlier_info = {}  # Store outlier information for each column
    
    for col in df_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]
    
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outliers': len(outliers),
        }
    
        print(f"Column: {col}")
        print(f"Number of outliers: {len(outliers)}")
        print("-" * 20)

# THIS FUNCTION CODE CYCLIC FEATURES AS "month" 
# WITHOUT ELIMINATE THE "month" feature
def mont_codification(df):
    # Convert months to numbers (jan=1, feb=2, ..., dec=12)
    df["month_num"] = df["month"].map({
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    })
    # Codificación cíclica
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df

def split_train_test(X, y):
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,     # 10% for testing, 90% for training
        random_state=0,
        stratify=y         # maintains class ratio in both sets :contentReference[oaicite:1]{index=1}
    )
    return X_train, X_test, y_train, y_test

# THIS FUNCTIONS ARE FOR MANUAL AJUSTMENT FOR THE HIPERPARAMETERS
def pipe_func_rfc(preprocessor, n_estimators, max_depth, min_split, min_leaf, max_features, n_jobs):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=0,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            n_jobs=n_jobs
        ))
    ])
    return pipeline
    
def pipe_func_gbc(preprocessor, n_estimators, max_depth, learning_rate):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        ))
    ])
    return pipeline

# DEFAULT CONFIGURATION
    # random_search(pipeline, param_distributions, 30, 'roc_auc', 5, -1, 2, 42, X_train, y_train)

def random_search(preprocessor, estimator, param_distributions, n_iter, scoring, cv, n_jobs, verbose, X_train, y_train):

    pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('classifier', estimator)
    ])
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    
    print("Mejores parámetros:", random_search.best_params_)
    print("Mejor score:", random_search.best_score_)
    print("Mejor modelo:", best_model)
    return best_model

# DEFAULT CONFIGURATION
    # grid_search(pipeline, param_grid, 'roc_auc', 5, -1, 2, X_train, y_train)
def grid_search(preprocessor, estimator, param_grid, scoring, cv, n_jobs, verbose, X_train, y_train):

    pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('classifier', estimator)
    ])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score:", grid_search.best_score_)
    print("Mejor modelo:", best_model)
    return best_model

def output(df, pipeline):
    tranformed_df_test= df.copy()
    tranformed_df_test= mont_codification(tranformed_df_test)
    tranformed_df_test = apply_outlier_bounds(tranformed_df_test, outlier_bounds)
    
    y_pred_prob = pipeline.predict_proba(tranformed_df_test)[:, 1]
    
    output=pd.DataFrame({'id':df.id, 'y':y_pred_prob})
    output.to_csv('submission.csv', index=False)
    output.head()
    return output


# Variables categóricas para OneHotEncoder
onehot_cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Variable categórica ordinal
ordinal_col = ['education']  # Solo si tiene un orden real
education_order = ['unknown', 'primary', 'secondary', 'tertiary']

# Variables numéricas (se dejan pasar tal cual)
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'month_num', 'month_sin', 'month_cos']

# Ordinal transformer
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[education_order]))
])

# OneHot transformer
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('ord', ordinal_transformer, ordinal_col),
    ('cat', onehot_transformer, onehot_cols)
])


# THE BATCH SIZE IS THE 30% OF THE ORIGINAL DATASET. 
#FOR BETTER TIMES IN THE TRAINNING
df_sample = df_train.sample(n=225_000, random_state=0)

# -------------------    CYCLIC "month" CODING     --------------------
df_sample=mont_codification(df_sample)

# SPLITTING DATA IN X AND y
X=df_sample.drop(['y'], axis=1)
y=df_sample.y

# Using function for Splitting data
X_train, X_test, y_train, y_test=split_train_test(X, y)

# ------------------------     Treated OUTLIERS    ----------------------------
outlier_bounds = get_outlier_bounds(X_train, numeric_cols)

X_train_clean=apply_outlier_bounds(X_train, outlier_bounds)
X_test_clean=apply_outlier_bounds(X_test, outlier_bounds)


param_distributions_rfc = [
    {  # RandomForest
        'classifier': [RandomForestClassifier(random_state=0)],
        'classifier__n_estimators': [5000], 
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [15,20,30]
    }
]

param_distributions_gbc = [
    {  # GradientBoosting
        'classifier': [GradientBoostingClassifier(random_state=0)],
        'classifier__n_estimators': [600],
        'classifier__learning_rate': [0.20, 0.25],
        'classifier__max_depth': [3, 4]
    }
]


# Instance of the Estiamtor RFC
rfc = RandomForestClassifier(random_state=0)
# Instance of the Estiamtor GBC
gbc= GradientBoostingClassifier(random_state=0)


"""best_model_random_search_rfc = random_search(
                                    preprocessor, 
                                    rfc, 
                                    param_distributions_rfc, 
                                    30, 
                                    'roc_auc', 
                                    3, 
                                    -1, 
                                    2, 
                                    X_train_clean, 
                                    y_train)"""


"""best_model_random_search_gbc = random_search(
                                    preprocessor, 
                                    gbc, 
                                    param_distributions_gbc, 
                                    30, 
                                    'roc_auc', 
                                    3, 
                                    -1, 
                                    2, 
                                    X_train_clean, 
                                    y_train)"""


df=df_train.copy()

df=mont_codification(df)

X=df.drop(['y'],axis=1)
y=df.y

X_train, X_test, y_train, y_test=split_train_test(X, y)

outlier_bounds = get_outlier_bounds(X_train, numeric_cols)

X_train_clean=apply_outlier_bounds(X_train, outlier_bounds)
X_test_clean=apply_outlier_bounds(X_test, outlier_bounds)


pipeline_rfc = pipe_func_rfc(
    preprocessor,
    5000,
    None,
    30,
    2,
    'sqrt',
    -1)

# Train the complete pipeline (including preprocessing)
pìpe_rfc = pipeline_rfc.fit(X_train_clean, y_train)


# Predicciones de probabilidad en test
y_prob_rfc = pìpe_rfc.predict_proba(X_test_clean)[:, 1]
# Calcular AUC – ROC
auc_rfc = roc_auc_score(y_test, y_prob_rfc)

print(f"ROC AUC score RandomForestClassifier: {auc_rfc:.5f}")


pipeline_gbc = pipe_func_gbc(
    preprocessor, 
    600, 
    4, 
    0.25 
    )

# Train the complete pipeline (including preprocessing)
pìpe_gbc = pipeline_gbc.fit(X_train_clean, y_train)


# Predicciones de probabilidad en test
y_prob_gbc = pìpe_gbc.predict_proba(X_test_clean)[:, 1]
# Calcular AUC – ROC
auc_gbc = roc_auc_score(y_test, y_prob_gbc)

print(f"ROC AUC score RandomForestClassifier: {auc_gbc:.5f}")


'output(df_test, pìpe_rfc)'
output(df_test, pìpe_gbc)

