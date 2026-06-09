%%capture
import gc
gc.collect()
!pip install duckdb
!pip install optbinning


import sys
import os
import missingno as msno
import joblib

import warnings
warnings.filterwarnings('ignore')

import random
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import duckdb
from optbinning import OptimalBinning

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cat

import shap
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, train_test_split, cross_validate  
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering  
from sklearn.preprocessing import StandardScaler 
from scipy.cluster.hierarchy import linkage, dendrogram , fcluster
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

import matplotlib.pyplot as plt
from plotnine import * 
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from sklearn.pipeline import Pipeline

import duckdb
duckdb.query('PRAGMA disable_progress_bar;')
from pprint import pprint


RANDOM_STATE = 1999
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(RANDOM_STATE)

class Shhh:
    # some of these models are still quite chatty even after disabling logging
    # we use this to swallow the printed output.
    # see: https://stackoverflow.com/questions/72346178/how-to-suppress-automatically-generated-output-from-a-python-code
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
%matplotlib inline
%config InlineBackend.figure_format='retina'

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#_ = pd.read_parquet('/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet')


#print(f'in the test data set, the max date is {_.date.max()}, and the min date is {_.date.min()}')


# Disables the progress bar for DuckDB queries.
duckdb.query('PRAGMA disable_progress_bar;')

BASEPATH = '/kaggle/input/neo-bank-non-sub-churn-prediction/'

train_path = BASEPATH + 'train_*.parquet'
test_path = BASEPATH + 'test.parquet'

# Creates a view named 'train_data' by over the Parquet files 
# matching the pattern in tran_path
# The 'filename=true' option adds a column with the source filename.
duckdb.query(f'''
    create or replace view train_data as
    SELECT 
        *
    FROM read_parquet('{train_path}', union_by_name=true, filename=true) t
    ''')

duckdb.query(f'''
    create or replace view test_data as
    SELECT 
        * EXCLUDE (Usage)
    FROM read_parquet('{test_path}', union_by_name=true, filename=true) t
    ''')

# combine train and test:
duckdb.query(f'''
    create or replace view all_data as
    select tr.*, 'train' as rowtype from train_data tr
    union all
    select ts.*, 'test' as rowtype from test_data ts
    '''
)


    current_date = "2024-01-01"
    # these are mostly null. csat 99.5% and touchpoints 96.5%
    # for now, ignore. There might be some tiny signal here.
    duckdb.query(f'''
        create or replace table csat_unnest as
        select
            csat_scores, 
            phone as csat_phone, 
            whatsapp as csat_whatsapp, 
            appointment as csat_appointment, 
            email as csat_email
        from (
            select csat_scores, unnest(csat_scores, max_depth:=1) as tp
            from all_data 
            group by csat_scores)''')

    duckdb.query(f'''
        create or replace table tp_unnest as
        with tp as (
            select touchpoints, unnest(touchpoints) as touchpoint
            from all_data 
            group by touchpoints)
        select * from tp
        PIVOT (
          SUM(CASE WHEN touchpoint IS NOT NULL THEN 1 ELSE 0 END) 
          FOR touchpoint IN (
              'phone' as tp_phone, 
              'whatsapp' as tp_whatsapp, 
              'appointment' as tp_appointment, 
              'email' as tp_email))
            ''')
# all_data_new
    duckdb.query(f'''
        create or replace table all_data_new as
        SELECT 
            t.* exclude (date_of_birth), 
            csat_unnest.* exclude (csat_scores), 
            tp_unnest.* exclude (touchpoints),
            datediff('year', date_of_birth, date) as age,
            datediff('day', date, '{current_date}') as date_distence
        FROM 
            --read_parquet('{train_path}', union_by_name=true, filename=true) t
            all_data t
            left join csat_unnest on csat_unnest.csat_scores = t.csat_scores
            left join tp_unnest on tp_unnest.touchpoints = t.touchpoints
        ''')


duckdb.query(f'''
    CREATE OR REPLACE TABLE DATE_BASE AS
    SELECT
        *,
        ABS(DATEDIFF('day', date, LAG(date)  OVER (PARTITION BY customer_id ORDER BY date DESC))) as date_diff,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY date) AS login_count,
        CASE
            WHEN 
                ABS(DATEDIFF('day', date, LAG(date)  OVER (PARTITION BY customer_id ORDER BY date DESC))) > 547 OR
                churn_due_to_fraud = 1 OR
                customer_id NOT IN (
                    SELECT
                        DISTINCT customer_id
                    FROM all_data_new
                    WHERE date BETWEEN '2022-06-01' AND '2023-12-31' 
                )THEN 1
            ELSE 0
        END AS churn
    FROM all_data_new
    ORDER BY date ASC''')


duckdb.query(f'''
    SELECT
        churn
    FROM DATE_BASE''').fetchdf().churn.value_counts() 


if False:
    _ = duckdb.query(f'''
        WITH date_diff AS (
        SELECT 
            customer_id,
            date_distence,
            date_distence - LAG(date_distence) OVER (PARTITION BY customer_id ORDER BY date_distence) AS date_diff
        FROM all_data_new
        WHERE date <= '2023-12-31')
        SELECT
            customer_id, 
            MIN(date_distence) as recent_days,
            AVG(date_diff) AS day_intervalle
        FROM
            date_diff
        GROUP BY 
            customer_id''').fetchdf()


if False:
    print(f'the 80%ist of clients was on ling within {_.recent_days.quantile(0.8)/365.25} years, and they have a on line habbit in average of {_.day_intervalle.quantile(0.8)} days')


if False : 
    # Exemple des données (remplacez "_" par vos données réelles)
    seuil_recent = _['recent_days'].quantile(0.7)
    bins = int((_['recent_days'].max() - _['recent_days'].min()) / (365.25 / 4))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Distribution for day_recent and the day_avg", fontsize=16)
    
    # Premier graphique : Distribution de "recent_days"
    sns.histplot(data=_, x="recent_days", ax=axes[0], color="skyblue", bins=bins, kde=True, alpha=0.7)
    axes[0].set_title("Distribution of Recent Days", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Recent days".replace("_", " ").capitalize())
    axes[0].axvline(seuil_recent, color="r", linestyle="dashed", linewidth=1, label=f"Seuil of 70% : {seuil_recent:.2f} days")  # Ligne verticale
    axes[0].legend()  # Afficher la légende pour le seuil
    
    seuil_day_fréquence = _['day_intervalle'].quantile(0.7)  # Quantile 70%
    seuil_day_moyenne = _['day_intervalle'].mean()           # Moyenne
    seuil_day_medianne = _['day_intervalle'].median()        # Médiane
    
    # Deuxième graphique : Distribution de "day_intervalle"
    sns.histplot(data=_, x="day_intervalle", ax=axes[1], color="skyblue", kde=True, alpha=0.7)
    axes[1].set_title("Distribution of Intervalle Average", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Day intervalle".replace("_", " ").capitalize())
    
    # Ajout des lignes verticales avec différentes couleurs et étiquettes
    axes[1].axvline(seuil_day_fréquence, color="r", linestyle="dashed", linewidth=1, label=f"Seuil 70% : {seuil_day_fréquence:.2f} days")
    axes[1].axvline(seuil_day_moyenne, color="b", linestyle="dashed", linewidth=1, label=f"Moyenne : {seuil_day_moyenne:.2f} days")
    axes[1].axvline(seuil_day_medianne, color="g", linestyle="dashed", linewidth=1, label=f"Médiane : {seuil_day_medianne:.2f} days")
    
    # Ajout de la légende
    axes[1].legend()
    
    # Ajustement des marges
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()
    
    
    # Ajustement des marges
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()


duckdb.query(f'''
    CREATE OR REPLACE TABLE c_score AS
    WITH test AS (
        SELECT 
            customer_id,
            unnest(csat_scores, max_depth := 1)
        FROM all_data_new
    ),
    score_union AS (
    SELECT customer_id, appointment AS score FROM test WHERE appointment IS NOT NULL 
    UNION ALL

    SELECT customer_id, email AS score FROM test WHERE email IS NOT NULL
    UNION ALL

    SELECT customer_id, phone AS score FROM test WHERE phone IS NOT NULL
    UNION ALL

    SELECT customer_id, whatsapp AS score FROM test WHERE whatsapp IS NOT NULL)
    
    SELECT 
        customer_id,
        ROUND(AVG(score)) AS m_score,
        CASE WHEN COUNT(score) = 1 THEN 0 ELSE ROUND(STDDEV(score)) END as stdev_score
    FROM score_union
    GROUP BY customer_id
''')


duckdb.query(f'''
CREATE OR REPLACE TABLE date_diff AS
WITH train_with_tenure AS (
    SELECT 
        customer_id,
        date_distence,
        MAX(tenure) OVER (PARTITION BY customer_id) AS max_tenure
    FROM all_data_new
),
date_diff_calculated AS (
    SELECT 
        customer_id,
        date_distence - LAG(date_distence) OVER (PARTITION BY customer_id ORDER BY date_distence) AS date_diff,
        'A' AS groupe
    FROM train_with_tenure
    WHERE max_tenure > 0

    UNION ALL

    SELECT
        customer_id,
        date_distence AS date_diff,
        'B' AS groupe
    FROM train_with_tenure
    WHERE max_tenure = 0
)

SELECT
    customer_id,
    ROUND(AVG(date_diff)) AS day_interval_avg,
    CASE 
        WHEN groupe = 'A' THEN ROUND(STDDEV(date_diff))
        ELSE 0
    END AS day_interval_stdd,
    CASE 
        WHEN groupe = 'A' THEN ROUND(AVG(date_diff) + STDDEV(date_diff))
        ELSE 0
    END AS day_tolérer
FROM date_diff_calculated 
WHERE date_diff IS NOT NULL
GROUP BY 
    customer_id, groupe
''')



duckdb.query(f'''
    create or replace table data_model as
    WITH develop_tp AS (
        SELECT 
            tr.customer_id,
            unnest(touchpoints, max_depth:=1) as tp
        FROM all_data_new tr
        WHERE tr.date <= '2023-12-31'
    ),
    counted_tp as (
        SELECT 
            d_tp.customer_id,
            count(distinct d_tp.tp) as tp_canneau,
            count(d_tp.tp) as tp_times_nb
        FROM develop_tp d_tp
        GROUP BY d_tp.customer_id
    )
    SELECT
        t.*,
        s.m_score,
        s.stdev_score,
        tp.tp_canneau,
        tp.tp_times_nb
    FROM DATE_BASE t
    LEFT JOIN counted_tp tp ON t.customer_id = tp.customer_id
    LEFT JOIN c_score s ON t.customer_id = s.customer_id
    LEFT JOIN date_diff day ON t.customer_id = day.customer_id''')


conn = duckdb.connect(database=':memory:')  

conn.execute("DROP TABLE IF EXISTS date_diff")
conn.execute("DROP TABLE IF EXISTS all_data")
conn.execute("DROP TABLE IF EXISTS all_data_new")
conn.execute("DROP TABLE IF EXISTS data_base")
conn.execute("DROP TABLE IF EXISTS c_score")
conn.execute("DROP TABLE IF EXISTS tp_unnest")
conn.execute("DROP TABLE IF EXISTS csat_unnest")


data_model = duckdb.query(f"""select * from data_model""").fetchdf()


conn.execute("DROP TABLE IF EXISTS data_model")


cols_1 = ['Id','customer_id','interest_rate','country',#'name','address','date',
 'atm_transfer_in',
 'atm_transfer_out',
 'bank_transfer_in',
 'bank_transfer_out',
 'crypto_in',
 'crypto_out',
 'bank_transfer_in_volume',
 'bank_transfer_out_volume',
 'crypto_in_volume',
 'crypto_out_volume',
 'complaints',
 #'touchpoints',
 #'csat_scores',
 'tenure',
 'from_competitor',
 'job',
 'churn_due_to_fraud',
 'model_predicted_fraud',
 'filename',
 'rowtype',
 #'csat_phone',
 #'csat_whatsapp',
 #'csat_appointment',
 #'csat_email',
 #'tp_phone',
 #'tp_whatsapp',
 #'tp_appointment',
 #'tp_email',
 'age',
 'date_distence',
 'date_diff', #0
 'login_count',
 'churn',
 #'m_score', #mean
 'stdev_score', #0
 'tp_canneau',#0
 'tp_times_nb']#0
cols_2 = ['m_score'] # mean
cols_drop =  ['touchpoints', 'csat_scores',
              'csat_phone','csat_whatsapp','csat_appointment',
              'csat_email','tp_phone','tp_whatsapp','tp_appointment',
              'tp_email','name','address','date']
data_model[cols_1] = data_model[cols_1].fillna(0)
data_model[cols_2] = data_model[cols_2].fillna(data_model[cols_2].mean())
data_model = data_model.drop(columns = cols_drop)


data_model.churn.value_counts() 


data = data_model[data_model.rowtype == 'train']
target = data_model[data_model.rowtype == 'test']


VARIABLE = "country"
NBINS = 15
optb = OptimalBinning(name=VARIABLE, dtype="categorical", solver="cp", max_n_bins=NBINS)
optb.fit(data[VARIABLE], data.churn)
bin_table = optb.binning_table
bin_table.build()
print(bin_table.build())
optb.binning_table.plot()


cat_maps = []
cat_map = {}
for i, lst in enumerate(bin_table.build().Bin[:-3]):
    for cat in lst:
        cat_map[cat] = VARIABLE + str(i)
cat_maps.append((cat_map, VARIABLE))

def map_to_cat(x):
    try:
        return cat_map[x]
    except:
        return NBINS
        
data = pd.concat([data, pd.get_dummies(data[VARIABLE].apply(map_to_cat), 
                                   drop_first=True)], axis=1)
data.head()


cat_maps = []
cat_map = {}
for i, lst in enumerate(bin_table.build().Bin[:-3]):
    for cat in lst:
        cat_map[cat] = VARIABLE + str(i)
cat_maps.append((cat_map, VARIABLE))

def map_to_cat(x):
    try:
        return cat_map[x]
    except:
        return NBINS
        
target = pd.concat([target, pd.get_dummies(target[VARIABLE].apply(map_to_cat), 
                                   drop_first=True)], axis=1)


VARIABLE = "job"
NBINS = 15
optb = OptimalBinning(name=VARIABLE, dtype="categorical", solver="cp", max_n_bins=NBINS)
optb.fit(data[VARIABLE], data.churn)
bin_table = optb.binning_table
bin_table.build()
print(bin_table.build())
optb.binning_table.plot()


cat_maps = []
cat_map = {}
for i, lst in enumerate(bin_table.build().Bin[:-3]):
    for cat in lst:
        cat_map[cat] = VARIABLE + str(i)
cat_maps.append((cat_map, VARIABLE))

def map_to_cat(x):
    try:
        return cat_map[x]
    except:
        return NBINS
        
data = pd.concat([data, pd.get_dummies(data[VARIABLE].apply(map_to_cat), 
                                   drop_first=True)], axis=1)
data.head()


cat_maps = []
cat_map = {}
for i, lst in enumerate(bin_table.build().Bin[:-3]):
    for cat in lst:
        cat_map[cat] = VARIABLE + str(i)
cat_maps.append((cat_map, VARIABLE))

def map_to_cat(x):
    try:
        return cat_map[x]
    except:
        return NBINS
        
target = pd.concat([target, pd.get_dummies(target[VARIABLE].apply(map_to_cat), 
                                   drop_first=True)], axis=1)


drop_columns = ['Id','customer_id', 'country', 'job', 'filename','rowtype']  
data = data.drop(columns=drop_columns)
drop_columns2 = ['customer_id', 'country', 'job', 'filename','rowtype']  
target = target.drop(columns=drop_columns2)


data_model = []


data.columns


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Sélection des variables
numeric_features = data.select_dtypes(include=['float64','int64']).columns  
binary_features = data.select_dtypes(include=['bool']).columns 

# Train et target
X = data.drop(columns=['churn']) 
y = data['churn']  

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construction du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Normalisation des variables numériques
        ('binary', 'passthrough', binary_features)    # Passer directement les variables booléennes
    ])

# Pipeline avec LogisticRegression
model = Pipeline([
    ('preprocessor', preprocessor),  
    ('classifier', LogisticRegression(solver='lbfgs', random_state=42))  # Utilisation de LogisticRegression
])

# Entraîner le modèle
model.fit(X_train, y_train)


if False : 
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    print(f"Scores de la cross-validation: {scores}")
    print(f"Précision moyenne: {scores.mean():.2f}")


logistic_model = model.named_steps['classifier']
feature_names = numeric_features.tolist() + binary_features.tolist()

# Visualize the important feature
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance':  logistic_model.coef_[0]
}).sort_values(by='Importance', ascending=False)

#plt.figure(figsize=(10, 6))
#plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
#plt.xlabel('Importance')
#plt.ylabel('Features')
#plt.title('Feature Importance in Logistic Regression')
#plt.gca().invert_yaxis() 
#plt.show()


feature_importance_min = duckdb.query(f'''
    SELECT
        *
    FROM feature_importance
    WHERE ABS(Importance) > 0.2
''').df()

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_min["Feature"], feature_importance_min["Importance"], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Filtered Feature Importance (|Importance| ≥ 0.2)")
plt.gca().invert_yaxis()
plt.show()


# Définir les variables conservées
keep_feature = feature_importance_min.Feature.tolist()

# Filtrer les colonnes des jeux d'entraînement et de test
X_train_1 = X_train[keep_feature]
X_test_1 = X_test[keep_feature]
X_train = []
X_test = []

# Définir les variables cibles
y_train_1 = y_train  # Variable cible associée au X_train original
y_test_1 = y_test    # Variable cible associée au X_test original

# Identifier les types de colonnes (numériques et booléennes)
numeric_features = X_train_1.select_dtypes(include=['float64', 'int64']).columns.tolist()
binary_features = X_train_1.select_dtypes(include=['bool']).columns.tolist()

# Créer le préprocesseur pour normaliser les données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Normalisation des variables numériques
        ('binary', 'passthrough', binary_features)    # Passer directement les variables booléennes
    ]
)

# Pipeline avec LogisticRegression
model = Pipeline([
    ('preprocessor', preprocessor),  
    ('classifier', LogisticRegression(solver='lbfgs', random_state=42))  # Utilisation de LogisticRegression
])

# Entraîner le modèle sur le nouveau jeu de données filtré
model.fit(X_train_1, y_train_1)

# Évaluer le modèle sur le jeu de test
accuracy = model.score(X_test_1, y_test_1)
print(f"Accuracy du modèle après filtrage des variables : {accuracy:.2f}")



from scipy.stats import loguniform, randint


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


model.get_params()


%%time
param_distributions = {
    "classifier__C": loguniform(1e-6, 1e6),
    "classifier__penalty": ["l1","l2"],  
    "classifier__max_iter": [200, 500, 800], 
    "classifier__class_weight": [None, "balanced"], 
    "classifier__tol": [1e-4, 1e-5],
}

model_random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    verbose=1,
    n_jobs=-1,
)


model_random_search.fit(X_train_1, y_train_1)
#joblib.dump(model, 'logistic_model.pkl')  # Le fichier sera créé avec ce nom
#print("Modèle sauvegardé.")
#loaded_model = joblib.load('logistic_model.pkl')

accuracy = model_random_search.score(X_test_1, y_test_1)

print(f"The test accuracy score of the best model is {accuracy:.2f}")
from pprint import pprint

print("The best parameters are:")
pprint(model_random_search.best_params_)


if False :
    best_params = {
        'classifier__C': 363710.74215429317,
        'classifier__class_weight': None,
        'classifier__max_iter': 200,
        'classifier__penalty': 'l2',
        'classifier__tol': 1e-06
    }            
    
    submission_sample = pd.read_csv('/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv')
    
    
    final_model = model.set_params(**best_params)
    X_test_final = target.loc[:, keep_feature]
    col_id = target['Id']


best_params = model_random_search.best_params_
submission_sample = pd.read_csv('/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv')


final_model = model.set_params(**best_params)
X_test_final = target.loc[:, keep_feature]
col_id = target['Id']


y_pred = final_model.predict_proba(X_test_final)[:, 1]


submission = pd.DataFrame({
    'Id': col_id,  # 确保 ID 直接从 sample 提取，格式无误
    'churn': y_pred    
})

submission.columns = ['Id', 'churn']

submission.to_csv("submission.csv", index=False)

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Submission file created!")


assert submission_sample.shape[0] == submission['Id'].shape[0], "IDs do not match!"


duckdb.query(f'''
    SELECT
        Id
    FROM
        submission
    WHERE
    Id NOT IN (SELECT Id FROM submission_sample)''')


pd.read_csv('/kaggle/working/submission.csv').head()


if False :
# 读取数据
    Id = pd.read_csv('/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv')
    

    best_params = model_random_search.best_params_
    

    final_model = model.set_params(**best_params)
    

    X = target[:, keep_feature + ['Id']].fillna(0)
    
    y_pred = final_model.predict_proba(X)[:, 1] 

    
    submission = pd.DataFrame({
        'Id': Id['Id'],
        'churn': y_pred    
    })
    
    submission.to_csv("submission.csv", index=False)
    
    for dirname, _, filenames in os.walk('/kaggle/working'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            print("Submission file created!")


if False :
    duckdb.query(f'''
        create or replace table data_model as
        WITH develop_tp AS (
            SELECT 
                tr.customer_id,
                unnest(touchpoints, max_depth:=1) as tp
            FROM all_data_new tr
            WHERE tr.date <= '2023-12-31'
        ),
        counted_tp as (
            SELECT 
                d_tp.customer_id,
                count(distinct d_tp.tp) as tp_canneau,
                count(d_tp.tp) as tp_times_nb
            FROM develop_tp d_tp
            GROUP BY d_tp.customer_id
        )
        SELECT 
            t.customer_id,
            t.age,
            t.country,
            t.filename,
            day.day_interval_avg,
            day.day_interval_stdd,
            CASE 
                WHEN MAX(t.date) >= '2022-06-01' THEN 0 
                ELSE 1
            END AS churn,
            ROUND(avg(t.interest_rate)) as interest_avg,
            ROUND(STDDEV(t.interest_rate)) as interest_stdev,
            ROUND(avg(t.atm_transfer_in)) as atm_in_avg,
            ROUND(STDDEV(t.atm_transfer_in)) as atm_in_stdev,
            ROUND(avg(t.atm_transfer_out)) as atm_in_avg,
            ROUND(STDDEV(t.atm_transfer_out)) as atm_in_stdev,
            ROUND(avg(t.bank_transfer_in)) as bk_in_avg,
            ROUND(STDDEV(t.bank_transfer_in)) as bk_in_stdev,
            ROUND(avg(t.bank_transfer_out)) as bk_out_avg,
            ROUND(STDDEV(t.bank_transfer_out)) as bk_out_avg,
            ROUND(avg(t.crypto_in)) as cpt_in_avg,
            ROUND(STDDEV(t.crypto_in)) as cpt_in_stdev,
            ROUND(avg(t.crypto_out)) as cpt_out_avg,
            ROUND(STDDEV(t.crypto_out)) as cpt_out_stdev,
            ROUND(avg(t.bank_transfer_in_volume)) as V_bk_in_avg,
            ROUND(STDDEV(t.bank_transfer_in_volume)) as V_bk_in_stdev,
            ROUND(avg(t.bank_transfer_out_volume)) as V_bk_out_avg,
            ROUND(STDDEV(t.bank_transfer_out_volume)) as V_bk_out_stdev,
            ROUND(avg(t.crypto_in_volume)) as V_cpt_out_avg,
            ROUND(STDDEV(t.crypto_in_volume)) as V_cpt_out_stdev,
            ROUND(avg(t.complaints)) as frq_complaints,
            tp.tp_canneau,
            tp.tp_times_nb,
            score.stdev_score,
            score.m_score
        FROM all_data_new t
        LEFT JOIN counted_tp tp ON t.customer_id = tp.customer_id
        LEFT JOIN c_score score ON t.customer_id = score.customer_id
        LEFT JOIN date_diff day ON t.customer_id = day.customer_id
        WHERE t.date <= '2023-12-31'
        GROUP BY 
            t.customer_id,
            t.age,
            t.country,
            t.filename,
            day.day_interval_avg,
            day.day_interval_stdd,
            day.day_tolérer,
            tp.tp_canneau,
            tp.tp_times_nb,
            score.stdev_score,
            score.m_score
    ''')
    data_model = duckdb.query(f"""select * from data_model""").fetchdf()
    cols_1 = ['customer_id', 'age', 'filename', 'day_interval_avg',
           'day_interval_stdd', 'interest_avg', 'interest_stdev',
           'atm_in_avg', 'atm_in_stdev', 'atm_in_avg_1', 'atm_in_stdev_1',
           'bk_in_avg', 'bk_in_stdev', 'bk_out_avg', 'bk_out_avg_1', 'cpt_in_avg',
           'cpt_in_stdev', 'cpt_out_avg', 'cpt_out_stdev', 'V_bk_in_avg',
           'V_bk_in_stdev', 'V_bk_out_avg', 'V_bk_out_stdev', 'V_cpt_out_avg',
           'V_cpt_out_stdev', 'frq_complaints', 'tp_canneau', 'tp_times_nb',
           'stdev_score']#, 'country'
    cols_2 = ['m_score']
    
    data_model[cols_1] = data_model[cols_1].fillna(0)
    data_model[cols_2] = data_model[cols_2].fillna(data_model[cols_2].mean())


if False :
    VARIABLE = "country"
    NBINS = 20
    optb = OptimalBinning(name=VARIABLE, dtype="categorical", solver="cp", max_n_bins=NBINS)
    optb.fit(data_model[VARIABLE], data_model.churn)
    bin_table = optb.binning_table
    bin_table.build()
    print(bin_table.build())
    optb.binning_table.plot()


if False :
    cat_maps = []
    cat_map = {}
    for i, lst in enumerate(bin_table.build().Bin[:-3]):
        for cat in lst:
            cat_map[cat] = VARIABLE + str(i)
    cat_maps.append((cat_map, VARIABLE))
    
    def map_to_cat(x):
        try:
            return cat_map[x]
        except:
            return NBINS
            
    data_model = pd.concat([data_model, pd.get_dummies(data_model[VARIABLE].apply(map_to_cat), 
                                       drop_first=True)], axis=1)
    data_model.head()


if False :
    data_model.churn.value_counts() 

#data_model.churn.value_counts() CASE WHEN MIN(t.date_distence) > day.day_tolérer AND MAX(t.date) > '2022-06-01' THEN 1 ELSE 0 END AS churn,
#churn
#0    371367
#1     83763
#data_model.churn.value_counts() #CASE WHEN MIN(t.date_distence) > day.day_tolérer OR MAX(t.date) > '2022-06-01' THEN 1 ELSE 0 END AS churn,
#churn
#1    454675
#0       455
#data_model.churn.value_counts() #MAX(t.date) > '2022-06-01' THEN 1 ELSE 0 END AS churn,
#churn
#0    331910
#1    123220
#data_model.churn.value_counts() #CASE WHEN MIN(t.date_distence) > day.day_tolérer THEN 1 ELSE 0 END AS churn,
#churn
#1    415218
#0     39912


if False :
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Supprimer les colonnes non utiles
    drop_columns = ['customer_id', 'country', 'filename']  
    data = data_model.drop(columns=drop_columns)
    
    # Sélection des variables
    numeric_features = data.select_dtypes(include=['float64']).columns  
    binary_features = data.select_dtypes(include=['bool']).columns 
    
    # Train et target
    X = data.drop(columns=['churn']) 
    y = data['churn']  
    
    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Construction du préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # Normalisation des variables numériques
            ('binary', 'passthrough', binary_features)    # Passer directement les variables booléennes
        ])
    
    # Pipeline avec LogisticRegression
    model = Pipeline([
        ('preprocessor', preprocessor),  
        ('classifier', LogisticRegression(solver='lbfgs', random_state=42))  # Utilisation de LogisticRegression
    ])
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    ## Tester le modèle
    #y_pred = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"Taux de correction: {accuracy:.2f}")



if False :
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    print(f"Scores de la cross-validation: {scores}")
    print(f"Précision moyenne: {scores.mean():.2f}")


if False :
    logistic_model = model.named_steps['classifier']
    
    feature_names = numeric_features.tolist() + binary_features.tolist()
    
    # Visualize the important feature
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance':  logistic_model.coef_[0]
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance in Logistic Regression')
    plt.gca().invert_yaxis() 
    plt.show()


if False :
    feature_importance_min01 = duckdb.query(f'''
        SELECT
            *
        FROM feature_importance
        WHERE ABS(Importance) > 0.1
    ''').df()
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_min01["Feature"], feature_importance_min01["Importance"], color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Filtered Feature Importance (|Importance| ≥ 0.1)")
    plt.gca().invert_yaxis()
    plt.show()


if False :
    # Définir les variables conservées
    keep_feature = feature_importance_min01.Feature.tolist()
    
    # Filtrer les colonnes des jeux d'entraînement et de test
    X_train_1 = X_train[keep_feature].copy()
    X_test_1 = X_test[keep_feature].copy()
    
    # Définir les variables cibles
    y_train_1 = y_train  # Variable cible associée au X_train original
    y_test_1 = y_test    # Variable cible associée au X_test original
    
    # Identifier les types de colonnes (numériques et booléennes)
    numeric_features = X_train_1.select_dtypes(include=['float64', 'int64']).columns.tolist()
    binary_features = X_train_1.select_dtypes(include=['bool']).columns.tolist()
    
    # Créer le préprocesseur pour normaliser les données
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # Normalisation des variables numériques
            ('binary', 'passthrough', binary_features)    # Passer directement les variables booléennes
        ]
    )
    
    # Pipeline avec LogisticRegression
    model = Pipeline([
        ('preprocessor', preprocessor),  
        ('classifier', LogisticRegression(solver='lbfgs', random_state=42))  # Utilisation de LogisticRegression
    ])
    
    # Entraîner le modèle sur le nouveau jeu de données filtré
    model.fit(X_train_1, y_train_1)
    
    # Évaluer le modèle sur le jeu de test
    accuracy = model.score(X_test_1, y_test_1)
    print(f"Accuracy du modèle après filtrage des variables : {accuracy:.2f}")



if False :
    from scipy.stats import loguniform, randint
      
    class loguniform_int:
        """Integer valued version of the log-uniform distribution"""
    
        def __init__(self, a, b):
            self._distribution = loguniform(a, b)
    
        def rvs(self, *args, **kwargs):
            """Random variable sample"""
            return self._distribution.rvs(*args, **kwargs).astype(int)


if False :
    %%time
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform, randint
    
    param_distributions = {
        "classifier__C": loguniform(1e-6, 1e6),
        "classifier__penalty": ["l1","l2"],  
        "classifier__max_iter": [100, 200, 500, 1000], 
        "classifier__class_weight": [None, "balanced"], 
        "classifier__tol": [1e-4, 1e-5, 1e-6],
    }
    
    model_random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        verbose=1,
    )
    
    model_random_search.fit(X_train_1, y_train_1)
    #joblib.dump(model, 'logistic_model.pkl')  # Le fichier sera créé avec ce nom
    #print("Modèle sauvegardé.")
    #loaded_model = joblib.load('logistic_model.pkl')


if False :
    accuracy = model_random_search.score(X_test_1, y_test_1)
    print(f"The test accuracy score of the best model is {accuracy:.2f}")


if False :
    from pprint import pprint
    print("The best parameters are:")
    pprint(model_random_search.best_params_)


if False :
    duckdb.query(f'''
        create or replace table data_test as
        WITH develop_tp AS (
            SELECT 
                tr.customer_id
                unnest(touchpoints, max_depth:=1) as tp
            FROM 
                read_parquet('{test_path}', union_by_name=true, filename=true) tr
        ),
        counted_tp as (
            SELECT 
                d_tp.customer_id,
                count(distinct d_tp.tp) as tp_canneau,
                count(d_tp.tp) as tp_times_nb
            FROM develop_tp d_tp
            GROUP BY d_tp.customer_id
        )
        SELECT 
            t.customer_id,
            datediff('year', date_of_birth, date) as age,
            t.country,
            t.filename,
            day.day_interval_avg,
            day.day_interval_stdd,
            ROUND(avg(t.interest_rate)) as interest_avg,
            ROUND(STDDEV(t.interest_rate)) as interest_stdev,
            ROUND(avg(t.atm_transfer_in)) as atm_in_avg,
            ROUND(STDDEV(t.atm_transfer_in)) as atm_in_stdev,
            ROUND(avg(t.atm_transfer_out)) as atm_in_avg,
            ROUND(STDDEV(t.atm_transfer_out)) as atm_in_stdev,
            ROUND(avg(t.bank_transfer_in)) as bk_in_avg,
            ROUND(STDDEV(t.bank_transfer_in)) as bk_in_stdev,
            ROUND(avg(t.bank_transfer_out)) as bk_out_avg,
            ROUND(STDDEV(t.bank_transfer_out)) as bk_out_avg,
            ROUND(avg(t.crypto_in)) as cpt_in_avg,
            ROUND(STDDEV(t.crypto_in)) as cpt_in_stdev,
            ROUND(avg(t.crypto_out)) as cpt_out_avg,
            ROUND(STDDEV(t.crypto_out)) as cpt_out_stdev,
            ROUND(avg(t.bank_transfer_in_volume)) as V_bk_in_avg,
            ROUND(STDDEV(t.bank_transfer_in_volume)) as V_bk_in_stdev,
            ROUND(avg(t.bank_transfer_out_volume)) as V_bk_out_avg,
            ROUND(STDDEV(t.bank_transfer_out_volume)) as V_bk_out_stdev,
            ROUND(avg(t.crypto_in_volume)) as V_cpt_out_avg,
            ROUND(STDDEV(t.crypto_in_volume)) as V_cpt_out_stdev,
            ROUND(avg(t.complaints)) as frq_complaints,
            tp.tp_canneau,
            tp.tp_times_nb,
            score.stdev_score,
            score.m_score
        FROM read_parquet('{test_path}', union_by_name=true, filename=true) t
        LEFT JOIN counted_tp tp ON t.customer_id = tp.customer_id
        LEFT JOIN c_score score ON t.customer_id = score.customer_id
        LEFT JOIN date_diff day ON t.customer_id = day.customer_id
        GROUP BY 
            t.customer_id,
            t.country,
            t.filename,
            t.date,
            t.date_of_birth,
            day.day_interval_avg,
            day.day_interval_stdd,
            day.day_tolérer,
            tp.tp_canneau,
            tp.tp_times_nb,
            score.stdev_score,
            score.m_score
    ''')
    data_test = duckdb.query(f"""select * from data_test""").fetchdf()
    cols_1 = ['customer_id', 'age', 'filename', 'day_interval_avg',
           'day_interval_stdd', 'interest_avg', 'interest_stdev',
           'atm_in_avg', 'atm_in_stdev', 'atm_in_avg_1', 'atm_in_stdev_1',
           'bk_in_avg', 'bk_in_stdev', 'bk_out_avg', 'bk_out_avg_1', 'cpt_in_avg',
           'cpt_in_stdev', 'cpt_out_avg', 'cpt_out_stdev', 'V_bk_in_avg',
           'V_bk_in_stdev', 'V_bk_out_avg', 'V_bk_out_stdev', 'V_cpt_out_avg',
           'V_cpt_out_stdev', 'frq_complaints', 'tp_canneau', 'tp_times_nb',
           'stdev_score']#, 'country'
    cols_2 = ['m_score']
    
    data_test[cols_1] = data_test[cols_1].fillna(0)
    data_test[cols_2] = data_test[cols_2].fillna(data_test[cols_2].mean())


if False :
    cat_maps = []
    cat_map = {}
    for i, lst in enumerate(bin_table.build().Bin[:-3]):
        for cat in lst:
            cat_map[cat] = VARIABLE + str(i)
    cat_maps.append((cat_map, VARIABLE))
    
    def map_to_cat(x):
        try:
            return cat_map[x]
        except:
            return NBINS
            
    data_test = pd.concat([data_test, pd.get_dummies(data_test[VARIABLE].apply(map_to_cat), 
                                       drop_first=True)], axis=1)
    data_test.head()


if False :
    if pd.read_parquet('/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet').shape[0] == data_test.shape[0] :
        # Get best params
        best_params = model_random_search.best_params_
        
        # Use best params to predic
        final_model = model.set_params(**best_params)
        
        
        features = data_test.drop(['country', 'filename'], axis=1)
    
        # Matrice 
        X = features[keep_feature]
        
        y_pred = final_model.predict_proba(X)[:, 1] 
        #y_pred = np.round(y_pred, 1)
        
        
        submission = pd.DataFrame({
            'Id': features['Id'],
            'churn': y_pred    
        })
        
        submission.to_csv("submission.csv", index=False)
        for dirname, _, filenames in os.walk('/kaggle/working'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                print("Submission file created!")


if False :
    a = pd.read_csv('/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv')
    a.columns


if False :
    pd.read_csv('/kaggle/working/submission.csv').churn.dtype


if False :
    test = pd.read_parquet("/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet")
    submission = pd.read_csv("/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv")
    # Make sure the IDs are aligned
    assert test['Id'].tolist() == submission['Id'].tolist(), "IDs do not match!"

