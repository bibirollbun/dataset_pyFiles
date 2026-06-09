import numpy as np 
import pandas as pd 

#elaborazione dei dati
from sklearn.preprocessing import OneHotEncoder #serve per risolvere le variabili categoriche
from sklearn.preprocessing import StandardScaler #serve per normalizzare le variabili
from sklearn.model_selection import train_test_split #per splittare i dati 

from sklearn.metrics import roc_auc_score, roc_curve #punteggio che viene usato all'interno della competizione, mi conviene usare questo

from sklearn.linear_model import LogisticRegression #modello di regressione lineare
from sklearn.ensemble import RandomForestClassifier #secondo modello che provo ad usare
from xgboost import XGBClassifier #terzo modello per arrivare top 10% 
from sklearn.metrics import accuracy_score, confusion_matrix #serve per valutare il modello e plottarlo

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

import time
from typing import Tuple

print('completato')


def engineer_features_numeric(
    train: pd.DataFrame, 
    test: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete numeric feature engineering pipeline for bank dataset.
    
    Creates only numerical features:
    - Ratios (balance/age, duration/campaign)
    - Interactions (age*balance, campaign*duration)
    - Aggregations (mean balance by job/education)
    - Domain features (is_solvent, campaign_fatigue)
    - Log transforms (skewed variables)
    - Statistical features (rolling means, differences)
    
    Args:
        train: Training DataFrame (without id/target)
        test: Test DataFrame (without id)
        verbose: Print progress messages
    
    Returns:
        Tuple of (train_enhanced, test_enhanced) with only numeric features
    """
    
    if verbose:
        print("\n" + "="*60)
        print("ğŸ”§ FEATURE ENGINEERING (NUMERIC ONLY)")
        print("="*60)
    
    # Store original shapes
    original_train_shape = train.shape
    original_test_shape = test.shape
    
    # Combine train+test for consistent transformations
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    n_train = len(train)
    
    # ============================================
    # 1. RATIO FEATURES
    # ============================================
    
    if verbose:
        print("ğŸ“Š Creating ratio features...")
    
    # Balance ratios
    if 'balance' in combined.columns and 'age' in combined.columns:
        combined['balance_per_age'] = combined['balance'] / (combined['age'] + 1)
    
    if 'balance' in combined.columns and 'duration' in combined.columns:
        combined['balance_per_duration'] = combined['balance'] / (combined['duration'] + 1)
    
    # Age ratios
    if 'age' in combined.columns and 'campaign' in combined.columns:
        combined['campaign_per_age'] = combined['campaign'] / (combined['age'] + 1)
    
    if 'age' in combined.columns and 'previous' in combined.columns:
        combined['previous_per_age'] = combined['previous'] / (combined['age'] + 1)
    
    # Campaign efficiency
    if 'duration' in combined.columns and 'campaign' in combined.columns:
        combined['avg_call_duration'] = combined['duration'] / (combined['campaign'] + 1)
    
    # Contact history ratio
    if 'previous' in combined.columns and 'pdays' in combined.columns:
        combined['previous_per_pdays'] = combined['previous'] / (combined['pdays'].replace(-1, 999) + 1)
    
    # ============================================
    # 2. INTERACTION FEATURES
    # ============================================
    
    if verbose:
        print("ğŸ”€ Creating interaction features...")
    
    # Age interactions
    if 'age' in combined.columns and 'balance' in combined.columns:
        combined['age_x_balance'] = combined['age'] * combined['balance']
    
    if 'age' in combined.columns and 'duration' in combined.columns:
        combined['age_x_duration'] = combined['age'] * combined['duration']
    
    # Campaign interactions
    if 'campaign' in combined.columns and 'duration' in combined.columns:
        combined['campaign_x_duration'] = combined['campaign'] * combined['duration']
    
    if 'campaign' in combined.columns and 'previous' in combined.columns:
        combined['campaign_x_previous'] = combined['campaign'] * combined['previous']
    
    # Contact history interactions
    if 'previous' in combined.columns and 'pdays' in combined.columns:
        combined['previous_x_pdays'] = combined['previous'] * (combined['pdays'] + 1)
    
    # ============================================
    # 3. DOMAIN-SPECIFIC FEATURES (NUMERIC FLAGS)
    # ============================================
    
    if verbose:
        print("ğŸ�¦ Creating domain features...")
    
    # Financial health
    if 'balance' in combined.columns:
        combined['is_solvent'] = (combined['balance'] > 0).astype(int)
        combined['high_balance'] = (combined['balance'] > combined['balance'].median()).astype(int)
        combined['balance_positive_ratio'] = np.maximum(combined['balance'], 0) / (np.abs(combined['balance']) + 1)
    
    # Contact recency
    if 'pdays' in combined.columns:
        combined['was_contacted_before'] = (combined['pdays'] != -1).astype(int)
        combined['recent_contact'] = (combined['pdays'] < 7).astype(int)
        combined['very_recent_contact'] = (combined['pdays'] < 3).astype(int)
        combined['pdays_log'] = np.log1p(combined['pdays'].replace(-1, 999))
    
    # Previous success
    if 'previous' in combined.columns:
        combined['has_previous_success'] = (combined['previous'] > 0).astype(int)
        combined['multiple_previous'] = (combined['previous'] > 1).astype(int)
    
    # Age segments
    if 'age' in combined.columns:
        combined['is_working_age'] = ((combined['age'] >= 18) & (combined['age'] <= 65)).astype(int)
        combined['is_senior'] = (combined['age'] >= 60).astype(int)
        combined['is_young'] = (combined['age'] < 30).astype(int)
        combined['age_squared'] = combined['age'] ** 2
    
    # Campaign fatigue
    if 'campaign' in combined.columns:
        combined['campaign_fatigue'] = (combined['campaign'] > 3).astype(int)
        combined['low_campaign'] = (combined['campaign'] <= 1).astype(int)
        combined['campaign_squared'] = combined['campaign'] ** 2
    
    # Duration indicators
    if 'duration' in combined.columns:
        combined['short_call'] = (combined['duration'] < 180).astype(int)
        combined['long_call'] = (combined['duration'] > 600).astype(int)
        combined['duration_log'] = np.log1p(combined['duration'])
    
    # ============================================
    # 4. AGGREGATION FEATURES (GROUP STATISTICS)
    # ============================================
    
    if verbose:
        print("ğŸ“ˆ Creating aggregation features...")
    
    # Balance aggregations by categorical groups
    if 'job' in combined.columns and 'balance' in combined.columns:
        job_balance_mean = combined.groupby('job')['balance'].transform('mean')
        job_balance_std = combined.groupby('job')['balance'].transform('std')
        combined['job_balance_mean'] = job_balance_mean
        combined['job_balance_std'] = job_balance_std.fillna(0)
        combined['balance_vs_job_mean'] = combined['balance'] - job_balance_mean
        combined['balance_vs_job_ratio'] = combined['balance'] / (job_balance_mean + 1)
    
    if 'education' in combined.columns and 'balance' in combined.columns:
        edu_balance_mean = combined.groupby('education')['balance'].transform('mean')
        combined['education_balance_mean'] = edu_balance_mean
        combined['balance_vs_edu_mean'] = combined['balance'] - edu_balance_mean
    
    # Age aggregations by categorical groups
    if 'job' in combined.columns and 'age' in combined.columns:
        job_age_mean = combined.groupby('job')['age'].transform('mean')
        combined['job_age_mean'] = job_age_mean
        combined['age_vs_job_mean'] = combined['age'] - job_age_mean
    
    # Duration aggregations
    if 'month' in combined.columns and 'duration' in combined.columns:
        month_duration_mean = combined.groupby('month')['duration'].transform('mean')
        combined['month_duration_mean'] = month_duration_mean
        combined['duration_vs_month_mean'] = combined['duration'] - month_duration_mean
    
    # Campaign aggregations
    if 'job' in combined.columns and 'campaign' in combined.columns:
        job_campaign_mean = combined.groupby('job')['campaign'].transform('mean')
        combined['job_campaign_mean'] = job_campaign_mean
    
    # Contact frequency by month
    if 'month' in combined.columns:
        month_count = combined.groupby('month')['month'].transform('count')
        combined['month_frequency'] = month_count
        combined['month_frequency_ratio'] = month_count / len(combined)
    
    # ============================================
    # 5. LOG TRANSFORMS (SKEWED VARIABLES)
    # ============================================
    
    if verbose:
        print("ğŸ“‰ Creating log transforms...")
    
    # Log transform highly skewed variables
    if 'balance' in combined.columns:
        combined['balance_log'] = np.log1p(combined['balance'] - combined['balance'].min() + 1)
    
    if 'campaign' in combined.columns:
        combined['campaign_log'] = np.log1p(combined['campaign'])
    
    if 'previous' in combined.columns:
        combined['previous_log'] = np.log1p(combined['previous'])
    
    # ============================================
    # 6. STATISTICAL FEATURES
    # ============================================
    
    if verbose:
        print("ğŸ“Š Creating statistical features...")
    
    # Balance statistics
    if 'balance' in combined.columns:
        combined['balance_abs'] = np.abs(combined['balance'])
        combined['balance_sign'] = np.sign(combined['balance'])
    
    # Squared features (polynomial degree 2)
    if 'duration' in combined.columns:
        combined['duration_squared'] = combined['duration'] ** 2
    
    # ============================================
    # 7. MISSING VALUE INDICATORS
    # ============================================
    
    if verbose:
        print("â�“ Creating missing indicators...")
    
    for col in combined.columns:
        if combined[col].isnull().sum() > 0:
            combined[f'{col}_is_missing'] = combined[col].isnull().astype(int)
    
    # ============================================
    # 8. SPLIT BACK INTO TRAIN/TEST
    # ============================================
    
    train_enhanced = combined.iloc[:n_train].reset_index(drop=True)
    test_enhanced = combined.iloc[n_train:].reset_index(drop=True)
    
    # ============================================
    # SUMMARY
    # ============================================
    
    if verbose:
        new_features = train_enhanced.shape[1] - original_train_shape[1]
        print("\n" + "="*60)
        print("âœ… FEATURE ENGINEERING COMPLETED")
        print("="*60)
        print(f"Original features:  {original_train_shape[1]}")
        print(f"New features:       {new_features}")
        print(f"Total features:     {train_enhanced.shape[1]}")
        print(f"Train shape:        {train_enhanced.shape}")
        print(f"Test shape:         {test_enhanced.shape}")
        
        # Check for any non-numeric columns (should be none!)
        non_numeric = train_enhanced.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"âš ï¸�  Warning: Non-numeric columns found: {non_numeric}")
        else:
            print("âœ… All features are numeric!")
        print("="*60)
    
    return train_enhanced, test_enhanced
print ('completato')


def output(train):
    print("="*60)
    print("ğŸ“Š ANALISI DATASET")
    print("="*60)
    
    # Quante feature?
    print(f"\nNumero colonne: {len(train.columns)}")
    print(f"Numero righe: {len(train)}")
    
    # Quali colonne?
    print(f"\nColonne:")
    print(train.columns.tolist())
    
    # Tipi di dati?
    print(f"\nTipi:")
    print(train.dtypes.value_counts())
    
    # Prime righe?
    print(f"\nPrime 5 righe:")
    print(train.head())
    
    # Info su ogni colonna?
    print(f"\nInfo:")
    train.info()


#raccolgo i dati
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

output(train)

#devo individuare chi Ã¨ l'obbiettivo
test_ids = test['id'].copy()
test = test.drop('id', axis = 1)
y = train['y'] #la variabile target si chiama t
X = train.drop('y', axis = 1) #le features su cui allenare sono tutte quelle in train eccetto che test
X = X.drop('id', axis = 1)

#introduzo il feature engeneering
X, test = engineer_features_numeric(X, test, verbose=True)

output(X)

#a questo punto devo andare a lavorare sui dati e dividerli in validation e training set

#prima cosa devo individuare le variabiki categoriche per poi andare ad applicare OH encoder
s = (X.dtypes == 'object')
object_cols = list(s[s].index)

#oneHotEncoding
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[object_cols])) #voglio fare OH_encoding sia su tabella del train sia su tabella del test
OH_cols_test = pd.DataFrame(OH_encoder.transform(test[object_cols]))

OH_cols_train.index = X.index #quando faccio OH_encoder si perdono i nomi delle colonne
OH_cols_test.index = test.index

X = X.drop(object_cols, axis=1) #sono andato a specializzare le colonne categoriche in modo che siano comprensibili dal modello -> ora posso togliere le colonne categoriche
test = test.drop(object_cols, axis=1)

X_numeric = pd.concat([X, OH_cols_train], axis=1) #unisco il dataset senza le colonne categoriche con la specializzazione delle stesse
test_numeric = pd.concat([test, OH_cols_test], axis=1)

X_numeric.columns = X_numeric.columns.astype(str) # mi assicuro che sia tutto str
test_numeric.columns = test_numeric.columns.astype(str)

#a questo punto ho solo feature numeriche 

#normalizzazione deve essere fatta su tutto perchÃ© alleno il modello ad allenare con una particolare tipologia di dati
print("\nğŸ”§ Normalizzazione colonne numeriche...")

scaler = StandardScaler()

# Normalizza train
X_numeric = scaler.fit_transform(X_numeric)

# Normalizza test (usa lo stesso scaler!)
test_numeric = scaler.transform(test_numeric)

#a questo punto posso dividere il train in train e valid
X_train, X_val, y_train, y_val = train_test_split(
    X_numeric, y, 
    test_size=0.15,      # 15% validation
    random_state=42,
    stratify=y   # mantiente le proporzioni, serve perchÃ© Ã¨ la stima migliore che abbiamo rispetto al test set
)

#provo a implementare una LOGISTIC REGRESSION come modello

#devo creare il modello usando la libreria sklearn

model = LogisticRegression()
#alleno il modello sul training set
model.fit(X_train, y_train)
#predico i valori del validation set e li confronto con i valori reali
print("errore train")
y_predict_train_prob = model.predict_proba(X_train)[:,1]
punteggio_train = roc_auc_score(y_train, y_predict_train_prob) #punteggio usato dalla competizione
print(f"ROC AUC Score: {punteggio_train:.4f}")

print("errore validation")
y_predict_val_prob = model.predict_proba(X_val)[:,1]
y_predict_val_bin = model.predict(X_val) #errore che stavo utilizzando io prima 
auc_value = roc_auc_score(y_val, y_predict_val_prob) #punteggio usato dalla competizione
print(f"ROC AUC Score: {auc_value:.4f}")

accuracy = accuracy_score(y_predict_val_bin, y_val) #punteggio che ho usato fino ad ora
print(f'accuracy: {accuracy:4f}')

#creo la confusion matrix per plottare le predizioni e i valori reali del VALIDATION
print('confusion matrix binaria')
confusionMatrix = confusion_matrix(y_predict_val_bin, y_val)
print(confusionMatrix)

test_prediction = model.predict_proba(test_numeric)[:,1]

'''
#usiamo hyper param (voglio cercare quale Ã¨ il valore migliore per i parametri)
param_distributions = {
    'n_estimators': randint(100, 2000),        # Range: 100-2000
    'learning_rate': uniform(0.01, 0.29),      # Range: 0.01-0.3
    'max_depth': randint(4, 12),               # Range: 4-12
    'subsample': uniform(0.6, 0.4),            # Range: 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4),     # Range: 0.6-1.0
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_distributions,
    n_iter=100,      # Quante combinazioni provare? 20? 50? 100?
    cv=3,          # Quanti fold? 3? 5? <- fold sono quanti split train/val faccio
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
'''

xgb = XGBClassifier(
    n_estimators=1582,  # PiÃ¹ alberi
    learning_rate= 0.055238410897498764,  # PiÃ¹ lento = piÃ¹ robusto
    max_depth=6,
    subsample=0.9464704583099741,
    colsample_bytree= 0.6624074561769746,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
'''
xgb.fit(X_train, y_train)
#predico i valori del validation set e li confronto con i valori reali
print("errore train")
y_predict_train_prob = xgb.predict_proba(X_train)[:,1]
punteggio_train = roc_auc_score(y_train, y_predict_train_prob) #punteggio usato dalla competizione
print(f"ROC AUC Score: {punteggio_train:.4f}")

print("errore validation")
y_predict_val_prob = xgb.predict_proba(X_val)[:,1]
y_predict_val_bin = xgb.predict(X_val) #errore che stavo utilizzando io prima 
auc_value = roc_auc_score(y_val, y_predict_val_prob) #punteggio usato dalla competizione
print(f"ROC AUC Score: {auc_value:.4f}")
'''
xgb.fit(X_numeric, y)
xgb_pred = xgb.predict_proba(test_numeric)[:,1]


# 4. Crea il DataFrame finale
submission = pd.DataFrame({
    'id': test_ids,  # Prendi gli ID originali
    'target': xgb_pred       # Metti le tue predizioni
})

# 5. Salva il file CSV (senza l'indice di pandas!)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("File 'submission.csv' creato con successo!")
print(submission.head()) # Controlla le prime righe

print ("completato")


#PLOT DEI DATI
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

# Ricarichiamo i dati puliti per l'analisi
df_viz = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test_viz = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# Rimuoviamo l'ID perchÃ© non Ã¨ una feature informativa per i grafici
df_viz = df_viz.drop('id', axis=1)

# --- 1. Analisi del Target (La cosa piÃ¹ importante) ---
plt.figure(figsize=(8, 5))
sns.countplot(x='y', data=df_viz, palette='viridis')
plt.title('Distribuzione della Variabile Target (y)')
plt.xlabel('Classe')
plt.ylabel('Conteggio')
plt.show()

# Calcoliamo la percentuale di sbilanciamento
target_counts = df_viz['y'].value_counts(normalize=True)
print(f"Distribuzione Target:\n{target_counts}")

# --- 2. Separazione Feature Numeriche e Categoriche ---
numeric_cols = df_viz.select_dtypes(include=['float64', 'int64']).columns.drop('y', errors='ignore')
cat_cols = df_viz.select_dtypes(include=['object']).columns

# --- 3. Distribuzione Feature Numeriche (Train vs Test) ---
# Ãˆ cruciale vedere se il Test set ha la stessa distribuzione del Train
if len(numeric_cols) > 0:
    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.kdeplot(df_viz[col], ax=axes[i], fill=True, label='Train', color='blue', alpha=0.3)
        # Se la colonna esiste anche nel test, la plottiamo per confronto
        if col in df_test_viz.columns:
            sns.kdeplot(df_test_viz[col], ax=axes[i], fill=True, label='Test', color='orange', alpha=0.3)
        
        axes[i].set_title(f'Distribuzione: {col}')
        axes[i].legend()

    # Rimuovi assi vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# --- 4. Distribuzione Feature Categoriche ---
if len(cat_cols) > 0:
    n_rows = math.ceil(len(cat_cols) / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        # Prendiamo solo le top 10 categorie per leggibilitÃ  se ce ne sono troppe
        top_cats = df_viz[col].value_counts().nlargest(10).index
        sns.countplot(y=col, data=df_viz[df_viz[col].isin(top_cats)], ax=axes[i], palette='muted', order=top_cats)
        axes[i].set_title(f'{col} (Top 10 Categorie)')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

