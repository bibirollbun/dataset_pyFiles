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


# Installation des dépendances
# ✅ Revenir à l'environnement Kaggle stable et compatible
!pip install -q -U numpy==1.26.4 scikit-learn==1.4.2 imbalanced-learn==0.12.3 xgboost==2.0.3 umap-learn==0.5.5
!pip uninstall -y cesium



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.compose import make_column_transformer, make_column_selector
import gc
from sklearn.preprocessing import OrdinalEncoder


def reduce_mem_usage(df):
    """Réduit l'utilisation mémoire d'un DataFrame."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Mémoire initiale : {start_mem:.2f} Mo")
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Mémoire réduite : {end_mem:.2f} Mo ({100 * (start_mem - end_mem) / start_mem:.1f}% gagné)")
    return df

# ====================================================
# Chargement des fichiers
# ====================================================
# Chargement complet (pour Kaggle, GPU conseillé)
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

train_df = train_transaction.merge(train_identity, how='left', on='TransactionID')
test_df = test_transaction.merge(test_identity, how='left', on='TransactionID')

# Libérer la mémoire
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

# ====================================================
# Réduction mémoire
# ====================================================
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# ====================================================
# Vérification
# ====================================================
print(f"Dimensions du train fusionné : {train_df.shape}")
print(f"Dimensions du test fusionné : {test_df.shape}")
print("Aperçu des données :")
display(train_df.head())


train_df.info()


train_df.duplicated().sum()


train_df.isnull().sum()


threshold=0.7
cols_to_drop=train_df.columns[train_df.isnull().mean()>threshold]
train_df=train_df.drop(columns=cols_to_drop)
print(f"Colonnes supprimées : {len(cols_to_drop)}")
print(f"Dimensions du dataset après nettoyage : {train_df.shape}")


num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_df.select_dtypes(include=['object']).columns

# Imputation
# Numériques → médiane
for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# Catégorielles → "Missing"
for col in cat_cols:
    train_df[col] = train_df[col].fillna("Missing")


# Distribution de la fraude 
plt.figure(figsize=(10,6))
sns.countplot(data=train_df, x="isFraud", color="Purple")
plt.title("Distribution de la variable cible (Fraude vs Non Fraude)")
plt.xlabel("Fraude (1) / Non Fraude (0)")
plt.ylabel("Nombre d'observations")
plt.show()

print(f"Pourcentage de fraudes : {train_df['isFraud'].mean():.2%}")



#Distribution générale des montants de transaction
plt.figure(figsize=(8,5))
sns.histplot(train_df["TransactionAmt"], bins=100, kde=True, color="purple")
plt.title("Distribution générale des montants de transaction")
plt.show()


# Distribution des montants de transaction selon la fraude
fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

sns.boxplot(y="TransactionAmt", data=train_df[train_df["isFraud"]==0], ax=axes[0], color="skyblue")
axes[0].set_title("Transactions normales")

sns.boxplot(y="TransactionAmt", data=train_df[train_df["isFraud"]==1], ax=axes[1], color="salmon")
axes[1].set_title("Transactions frauduleuses")

plt.ylim(0, 2000)
plt.show()


# Répartition des types de produits selon la fraude
plt.figure(figsize=(10,6))
sns.countplot(data=train_df, x="ProductCD", hue="isFraud", palette="Purples")
plt.title("Répartition des types de produits selon la fraude")
plt.xlabel("Type de produit")
plt.ylabel("Nombre de transactions")
plt.legend(title="Fraude")
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(train_df["TransactionDT"], bins=100, kde=False, color="purple")
plt.title("Distribution des transactions dans le temps")
plt.xlabel("TransactionDT (temps relatif)")
plt.ylabel("Nombre de transactions")
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(train_df["card1"], bins=50, kde=False, color="purple")
plt.title("Distribution des identifiants card1")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="card4", data=train_df, hue="isFraud", palette="Purples")
plt.title("Distribution du type de carte selon la fraude")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="card6", data=train_df, hue="isFraud", palette="Purples")
plt.title("Distribution du type de carte (débit/crédit) selon la fraude")
plt.show()



plt.figure(figsize=(10,6))
sns.countplot(x="addr1", data=train_df, order=train_df["addr1"].value_counts().iloc[:10].index)
plt.title("Top 10 des adresses addr1")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="P_emaildomain", data=train_df, order=train_df["P_emaildomain"].value_counts().iloc[:10].index)
plt.title("Top 10 des domaines email (payer)")
plt.show()



cols = ["isFraud","TransactionAmt"] + [col for col in train_df.columns if "card" in col]
num_cols = train_df[cols].select_dtypes(include=["number"])  # garde que les colonnes numériques

corr = num_cols.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Corrélations avec la variable cible (fraude)")
plt.show()



def feature_engineering_pipeline(train_df, test_df):
    """
    Applique le feature engineering complet pour IEEE-CIS Fraud Detection
    en évitant la fuite de données et les problèmes de fragmentation mémoire.
    """

    # -----------------------------
    # 1. Variables temporelles
    # -----------------------------
    START_DATE = pd.to_datetime('2017-11-30')

    def add_time_features(df):
        if not np.issubdtype(df['TransactionDT'].dtype, np.datetime64):
            df['TransactionDT'] = pd.to_timedelta(df['TransactionDT'], unit='s') + START_DATE

        time_features = pd.DataFrame({
            'hour': df['TransactionDT'].dt.hour,
            'day': df['TransactionDT'].dt.day,
            'weekday': df['TransactionDT'].dt.weekday,
            'is_weekend': (df['TransactionDT'].dt.weekday >= 5).astype(int)
        })
        return pd.concat([df, time_features], axis=1)

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)

    # -----------------------------
    # 2. Variables financières simples
    # -----------------------------
    def add_amount_features(df):
        amount_features = pd.DataFrame({
            'amt_cents': df['TransactionAmt'] % 1,
            'amt_digits': df['TransactionAmt'].astype(str).str.split('.').str[-1].str.len(),
            'amt_per_hour': df['TransactionAmt'] / (df['hour'] + 1)
        })
        return pd.concat([df, amount_features], axis=1)

    train_df = add_amount_features(train_df)
    test_df = add_amount_features(test_df)

    # -----------------------------
    # 3. Variables Email / Device
    # -----------------------------
    def add_email_features(df):
        email_features = pd.DataFrame({
            'P_emaildomain_bin': df['P_emaildomain'].apply(
                lambda x: str(x).split('.')[-1] if pd.notnull(x) else x
            ),
            'is_corporate_email': df['P_emaildomain'].isin(['gmail.com', 'yahoo.com']).astype(int)
        })
        return pd.concat([df, email_features], axis=1)

    train_df = add_email_features(train_df)
    test_df = add_email_features(test_df)

    # -----------------------------
    # 4. Variables statistiques basées sur le train
    # -----------------------------
    group_cols = ['card1', 'card2', 'addr1']

    for col in group_cols:
        stats = train_df.groupby(col)['TransactionAmt'].agg(['mean', 'median', 'std', 'count']).reset_index()
        stats.columns = [col, f'{col}_mean_amt', f'{col}_median_amt', f'{col}_std_amt', f'{col}_freq']

        # Merge sur train/test
        train_df = train_df.merge(stats, on=col, how='left')
        test_df = test_df.merge(stats, on=col, how='left')

        # Remplir NaN pour le test (catégories non vues dans le train)
        for new_col in [f'{col}_mean_amt', f'{col}_median_amt', f'{col}_std_amt', f'{col}_freq']:
            test_df[new_col] = test_df[new_col].fillna(train_df[new_col].mean())

    # -----------------------------
    # 5. Défragmenter les DataFrames
    # -----------------------------
    train_df = train_df.copy()
    test_df = test_df.copy()

    return train_df, test_df



train_df, test_df = feature_engineering_pipeline(train_df, test_df)

X = train_df.drop('isFraud', axis=1)
y = train_df['isFraud']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Sélection des colonnes
numerical_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)

# Pipelines de prétraitement
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_features),
    (categorical_pipeline, categorical_features)
)

# ⚙️ Modèle XGBoost optimisé (compatible GPU XGBoost 2.x)
xgb_base = XGBClassifier(
    tree_method="hist",   
    device="cpu",       
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1, # pour éviter la saturation GPU
    max_bin=256
)

# Pipeline complet
pipeline = imbPipeline([
    ('preprocessor', preprocessor),
    ('smoteen', SMOTEENN(random_state=42)),
    ('model', xgb_base)
])

# Grille de recherche allégée (pour Kaggle)
param_distributions = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [4, 6, 8],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.3],
    'model__scale_pos_weight': [1, 2, 5]
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=8,            # garde un peu de marge mémoire
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

import gc
del df_chunk
gc.collect()


# Entraînement
random_search.fit(X_train, y_train)

print("Meilleurs paramètres :", random_search.best_params_)
print("Meilleur score AUC :", random_search.best_score_)



best_model = random_search.best_estimator_

y_pred = best_model.predict(X_valid)
y_proba = best_model.predict_proba(X_valid)[:, 1]

print("ROC AUC sur validation :", roc_auc_score(y_valid, y_proba))
print(classification_report(y_valid, y_pred))



from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report


# --- 1. Prétraitement identique à XGBoost ---
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_valid_preprocessed = preprocessor.transform(X_valid)
X_test_preprocessed  = preprocessor.transform(X_test)

# --- 2. Entraîner IsolationForest ---
iso = IsolationForest(
    n_estimators=200,
    contamination='auto',  # proportion estimée d'anomalies
    random_state=42
)
iso.fit(X_train_preprocessed)

# --- 3. Scores d'anomalie ---
# score_samples donne des valeurs plus grandes = plus "normal", donc on inverse
s_train = -iso.score_samples(X_train_preprocessed)
s_valid = -iso.score_samples(X_valid_preprocessed)
s_test  = -iso.score_samples(X_test_preprocessed)

# --- 4. Normaliser les scores entre 0 et 1 ---
scaler = MinMaxScaler()
s_train_norm = scaler.fit_transform(s_train.reshape(-1,1)).ravel()
s_valid_norm = scaler.transform(s_valid.reshape(-1,1)).ravel()
s_test_norm  = scaler.transform(s_test.reshape(-1,1)).ravel()


from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_valid, s_valid_norm)
auprc   = average_precision_score(y_valid, s_valid_norm)

print(f"IsolationForest ROC-AUC: {roc_auc:.3f}, AUPRC: {auprc:.3f}")



from sklearn.metrics import precision_recall_fscore_support
X_test=test_df
# --- 6. Optimisation alpha (poids XGBoost / IsolationForest) sur validation ---
alphas = np.linspace(0, 1, 11)  # de 0 à 1 par pas de 0.1
best_alpha = 0
best_f1 = 0
best_threshold = 0

for alpha in alphas:
    combined_score = alpha * y_proba + (1 - alpha) * s_valid_norm
    
    # Chercher le meilleur seuil pour maximiser F1 sur validation
    thresholds = np.linspace(0, 1, 101)
    for thr in thresholds:
        y_pred_combined = (combined_score >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_valid, y_pred_combined, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_threshold = thr

print(f"Meilleur alpha: {best_alpha:.2f}, meilleur seuil: {best_threshold:.2f}, F1 sur validation: {best_f1:.3f}")

# --- 7. Évaluation finale sur test set ---
combined_test = best_alpha * best_model.predict_proba(X_test)[:,1] + (1 - best_alpha) * s_test_norm
y_test_pred = (combined_test >= best_threshold).astype(int)

roc_auc_test = roc_auc_score(y_test, combined_test)
auprc_test   = average_precision_score(y_test, combined_test)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')

print("\n--- Évaluation finale sur test set ---")
print(f"ROC-AUC: {roc_auc_test:.3f}, AUPRC: {auprc_test:.3f}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")






import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# --- 1. Probabilités / scores sur test set ---
y_proba_xgb = best_model.predict_proba(X_test)[:,1]
y_proba_iso = s_test_norm
y_proba_hybrid = combined_test

# --- 2. Courbes ROC ---
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
fpr_iso, tpr_iso, _ = roc_curve(y_test, y_proba_iso)
fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_proba_hybrid)

roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_iso = auc(fpr_iso, tpr_iso)
roc_auc_hyb = auc(fpr_hyb, tpr_hyb)

plt.figure(figsize=(8,6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_xgb:.3f})')
plt.plot(fpr_iso, tpr_iso, label=f'IsolationForest (AUC={roc_auc_iso:.3f})')
plt.plot(fpr_hyb, tpr_hyb, label=f'Hybride (AUC={roc_auc_hyb:.3f})')
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbes ROC - Test Set')
plt.legend()
plt.grid()
plt.show()

# --- 3. Courbes Precision-Recall ---
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
prec_iso, rec_iso, _ = precision_recall_curve(y_test, y_proba_iso)
prec_hyb, rec_hyb, _ = precision_recall_curve(y_test, y_proba_hybrid)

auprc_xgb = auc(rec_xgb, prec_xgb)
auprc_iso = auc(rec_iso, prec_iso)
auprc_hyb = auc(rec_hyb, prec_hyb)

plt.figure(figsize=(8,6))
plt.plot(rec_xgb, prec_xgb, label=f'XGBoost (AUPRC={auprc_xgb:.3f})')
plt.plot(rec_iso, prec_iso, label=f'IsolationForest (AUPRC={auprc_iso:.3f})')
plt.plot(rec_hyb, rec_hyb, label=f'Hybride (AUPRC={auprc_hyb:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbes Precision-Recall - Test Set')
plt.legend()
plt.grid()
plt.show()


