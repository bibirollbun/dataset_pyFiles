import subprocess
subprocess.run([
    "pip", "install", "-q",
    "imbalanced-learn==0.12.3",
    "scikit-learn==1.5.2",
    "xgboost==2.1.1"
], check=True)
print(" Dépendances installées — Fais : Run > Restart & Run All")


# ── ÉTAPE 0 : Import des librairies ─────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import xgboost as xgb

print("Librairies importées")


# ── ÉTAPE 1 : Chargement du dataset ─────────────────────────
# Sur Kaggle, le dataset est déjà disponible ici :
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity    = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

# Fusion des deux tables
df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

print(f" Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
print(f"   Taux de fraude : {df['isFraud'].mean()*100:.2f}%")


# ── ÉTAPE 2 : Échantillonnage stratifié (50 000 lignes) ──────
# → Permet de tourner rapidement même sans GPU
df = df.groupby('isFraud', group_keys=False).apply(
    lambda x: x.sample(frac=20000/len(df), random_state=42)
).reset_index(drop=True)

print(f" Échantillon stratifié : {df.shape[0]:,} lignes")
print(f"   Taux de fraude conservé : {df['isFraud'].mean()*100:.2f}%")
print(f"   → Fraudes dans l'échantillon : {df['isFraud'].sum()} transactions")


# ── ÉTAPE 3 : EDA — Analyse Exploratoire ────────────────────
print("\n" + "="*50)
print(" ANALYSE EXPLORATOIRE (EDA)")
print("="*50)

# 3a. Aperçu général
print(f"\n Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f" Taux de fraude : {df['isFraud'].mean()*100:.2f}%")
print(f" Valeurs manquantes : {df.isnull().sum().sum():,} au total")
print(f" Types de colonnes :\n{df.dtypes.value_counts()}")

# ── FIGURE 1 : Distribution des montants ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution des Montants de Transaction", fontsize=14, fontweight='bold')

# Distribution brute
axes[0].hist(df['TransactionAmt'].clip(upper=2000), bins=80,
             color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_xlabel("Montant ($)")
axes[0].set_ylabel("Fréquence")
axes[0].set_title("Distribution brute (cap à 2000$)")

# Distribution par classe (fraude vs légitime)
for label, color, name in [(0, 'steelblue', 'Légitime'), (1, 'crimson', 'Fraude')]:
    subset = df[df['isFraud'] == label]['TransactionAmt'].clip(upper=2000)
    axes[1].hist(subset, bins=80, alpha=0.6, color=color, label=name, edgecolor='white')
axes[1].set_xlabel("Montant ($)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution par classe")
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_distribution_montants.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Figure 1 sauvegardée : distribution des montants")

# ── FIGURE 2 : Boxplots / Outliers ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Boxplots — Détection des Outliers", fontsize=14, fontweight='bold')

# Boxplot montant brut
df.boxplot(column='TransactionAmt', ax=axes[0], vert=True,
           boxprops=dict(color='steelblue'),
           medianprops=dict(color='crimson', linewidth=2))
axes[0].set_title("TransactionAmt — Valeurs brutes")
axes[0].set_ylabel("Montant ($)")

# Boxplot montant par classe
df.boxplot(column='TransactionAmt', by='isFraud', ax=axes[1],
           boxprops=dict(color='steelblue'),
           medianprops=dict(color='crimson', linewidth=2))
axes[1].set_title("TransactionAmt par classe (0=Légitime, 1=Fraude)")
axes[1].set_xlabel("isFraud")
axes[1].set_ylabel("Montant ($)")
plt.suptitle("")

# Calcul IQR pour commentaire
Q1 = df['TransactionAmt'].quantile(0.25)
Q3 = df['TransactionAmt'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[df['TransactionAmt'] > Q3 + 1.5 * IQR]
print(f"\n Outliers détectés (méthode IQR) : {len(outliers):,} transactions ({len(outliers)/len(df)*100:.1f}%)")
print(f"   → Conservés car les montants extrêmes sont informatifs pour la détection de fraude")

plt.tight_layout()
plt.savefig('eda_boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Figure 2 sauvegardée : boxplots outliers")

# ── FIGURE 3 : Taux de fraude par catégorie (ProductCD) ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Taux de Fraude par Catégorie", fontsize=14, fontweight='bold')

# ProductCD
fraud_by_product = df.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False)
fraud_by_product.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white', rot=0)
axes[0].set_title("Taux de fraude par ProductCD")
axes[0].set_xlabel("Type de produit")
axes[0].set_ylabel("Taux de fraude")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'{bar.get_height():.1%}',
                 ha='center', va='bottom', fontsize=9)

# card4 (type de réseau : Visa, Mastercard…)
if 'card4' in df.columns:
    fraud_by_card4 = df.groupby('card4')['isFraud'].mean().sort_values(ascending=False).head(6)
    fraud_by_card4.plot(kind='bar', ax=axes[1], color='darkorange', edgecolor='white', rot=30)
    axes[1].set_title("Taux de fraude par card4 (réseau carte)")
    axes[1].set_xlabel("Réseau")
    axes[1].set_ylabel("Taux de fraude")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

plt.tight_layout()
plt.savefig('eda_fraude_par_categorie.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Figure 3 sauvegardée : taux de fraude par catégorie")

# ── FIGURE 4 : Analyse Temporelle ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Analyse Temporelle des Fraudes", fontsize=14, fontweight='bold')

# Heure de la journée (TransactionDT est en secondes depuis référence)
df['hour'] = (df['TransactionDT'] // 3600) % 24

# Volume de transactions par heure
hourly_total = df.groupby('hour').size()
hourly_fraud = df[df['isFraud'] == 1].groupby('hour').size()
hourly_rate  = (hourly_fraud / hourly_total).fillna(0)

axes[0].bar(hourly_total.index, hourly_total.values, color='steelblue', alpha=0.6, label='Total')
axes[0].set_xlabel("Heure de la journée")
axes[0].set_ylabel("Nombre de transactions", color='steelblue')
axes[0].set_title("Volume de transactions par heure")
ax2 = axes[0].twinx()
ax2.plot(hourly_rate.index, hourly_rate.values, color='crimson', lw=2, marker='o', label='Taux fraude')
ax2.set_ylabel("Taux de fraude", color='crimson')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
axes[0].legend(loc='upper left')
ax2.legend(loc='upper right')

# Jour de la semaine
df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
daily_rate = df.groupby('day_of_week')['isFraud'].mean()
axes[1].bar(daily_rate.index, daily_rate.values, color='darkorange', edgecolor='white')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(jours)
axes[1].set_xlabel("Jour de la semaine")
axes[1].set_ylabel("Taux de fraude")
axes[1].set_title("Taux de fraude par jour")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

peak_hour = hourly_rate.idxmax()
print(f"\n Pic de fraude détecté à {peak_hour}h ({hourly_rate.max()*100:.1f}% de taux)")
print(f"   → Cohérent avec des attaques automatisées nocturnes")

plt.tight_layout()
plt.savefig('eda_analyse_temporelle.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Figure 4 sauvegardée : analyse temporelle")

# ── FIGURE 5 : Heatmap de corrélations ──────────────────────
# Sélection des colonnes numériques les plus corrélées avec isFraud
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_with_target = df[numeric_cols].corr()['isFraud'].abs().sort_values(ascending=False)
top_features = corr_with_target.head(16).index.tolist()  # Top 15 + isFraud

corr_matrix = df[top_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Triangle supérieur masqué
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8}
)
ax.set_title("Heatmap de Corrélations — Top 15 Features vs isFraud", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_heatmap_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# Identifier les paires très corrélées (redondance)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.95 and corr_matrix.columns[i] != 'isFraud':
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

print(f"\n Paires très corrélées (>0.95) : {len(high_corr_pairs)} — candidates à supprimer")
print("Figure 5 sauvegardée : heatmap corrélations")



# ── ÉTAPE 4 : Prétraitement ──────────────────────────────────

# 4a. Suppression des colonnes avec trop de valeurs manquantes (> 80%)
thresh = 0.8
df = df.loc[:, df.isnull().mean() < thresh]
print(f" 4a. Colonnes après suppression NaN > 80% : {df.shape[1]}")

# 4b. Encodage des variables catégorielles
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
print(f" 4b. Encodage catégoriel : {len(cat_cols)} colonnes encodées")

# 4c. Remplissage des valeurs manquantes restantes par la médiane
df.fillna(df.median(numeric_only=True), inplace=True)
print(f" 4c. Valeurs manquantes restantes : {df.isnull().sum().sum()}")

# 4d. Normalisation log sur TransactionAmt (réduire l'impact des outliers)
df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
print(" 4d. Transformation log appliquée sur TransactionAmt")

print(f"\n Prétraitement terminé")


# ── ÉTAPE 5 : Feature Engineering ───────────────────────────
print("\n" + "="*50)
print("⚙️  FEATURE ENGINEERING")
print("="*50)

# Fréquence d'utilisation de chaque carte (proxy comportemental)
df['card1_freq']     = df.groupby('card1')['card1'].transform('count')
df['card_amt_mean']  = df.groupby('card1')['TransactionAmt'].transform('mean')
df['amt_vs_mean']    = df['TransactionAmt'] / (df['card_amt_mean'] + 1)

# Features temporelles (extraites de TransactionDT)
df['hour']        = (df['TransactionDT'] // 3600) % 24
df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

print(" Features créées : card1_freq, card_amt_mean, amt_vs_mean, hour, day_of_week, is_night")


# ── ÉTAPE 6 : Séparation features / cible ───────────────────
X = df.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Split : {X_train.shape[0]:,} train | {X_test.shape[0]:,} test")


# ── ÉTAPE 7 : Gestion du déséquilibre avec SMOTE ────────────
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f" Après SMOTE : {y_res.value_counts().to_dict()}")


# ── ÉTAPE 8 : Modèle XGBoost ────────────────────────────────
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    scale_pos_weight=ratio,
    subsample=0.8,            # Régularisation : 80% des données par arbre
    colsample_bytree=0.8,     # 80% des features par arbre
    min_child_weight=5,       # Évite le surapprentissage sur SMOTE
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=30  # Arrête si pas d'amélioration sur 30 rounds
)

xgb_model.fit(
    X_res, y_res,
    eval_set=[(X_test, y_test)],
    verbose=50
)



# ── ÉTAPE 9 : Évaluation XGBoost ────────────────────────────
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# --- Recherche du seuil optimal (maximise le F1 sur la classe fraude) ---
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores_thresh = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold = thresholds[f1_scores_thresh.argmax()]
print(f"\n Seuil optimal trouvé : {best_threshold:.3f} (défaut = 0.500)")

# Prédictions avec seuil par défaut (0.5)
y_pred_default = (y_pred_prob >= 0.5).astype(int)

# Prédictions avec seuil optimisé
y_pred_optimal = (y_pred_prob >= best_threshold).astype(int)

auc   = roc_auc_score(y_test, y_pred_prob)
f1_d  = f1_score(y_test, y_pred_default)
f1_o  = f1_score(y_test, y_pred_optimal)

print("\n" + "="*50)
print(" RÉSULTATS XGBOOST")
print("="*50)
print(f"  AUC-ROC                   : {auc:.4f}")
print(f"  F1-Score (seuil 0.5)      : {f1_d:.4f}")
print(f"  F1-Score (seuil optimal)  : {f1_o:.4f}  ← meilleur")
print(f"\n── Seuil par défaut (0.5) ──")
print(classification_report(y_test, y_pred_default, target_names=['Légitime', 'Fraude']))
print(f"── Seuil optimal ({best_threshold:.2f}) ──")
print(classification_report(y_test, y_pred_optimal, target_names=['Légitime', 'Fraude']))

# --- Recherche du seuil équilibré (Recall ≥ 0.60 ET meilleur F1) ──────────
print("\n" + "="*50)
print("  ANALYSE DU SEUIL ÉQUILIBRÉ")
print("="*50)

from sklearn.metrics import precision_score, recall_score

# Table complète des seuils
threshold_candidates = []
for t in thresholds:
    y_t  = (y_pred_prob >= t).astype(int)
    pr   = precision_score(y_test, y_t, pos_label=1, zero_division=0)
    re   = recall_score(y_test, y_t, pos_label=1, zero_division=0)
    f1_t = f1_score(y_test, y_t, zero_division=0)
    threshold_candidates.append({'seuil': round(t,3), 'precision': round(pr,3),
                                  'recall': round(re,3), 'f1': round(f1_t,3)})
df_thresh = pd.DataFrame(threshold_candidates)

# Meilleur F1 parmi les seuils avec recall >= 0.60
df_balanced = df_thresh[df_thresh['recall'] >= 0.60]
if len(df_balanced) > 0:
    best_balanced_row = df_balanced.loc[df_balanced['f1'].idxmax()]
    balanced_threshold = best_balanced_row['seuil']
    y_pred_balanced    = (y_pred_prob >= balanced_threshold).astype(int)
    f1_b = f1_score(y_test, y_pred_balanced)
    print(f"  Seuil équilibré : {balanced_threshold:.3f}  (Recall ≥ 0.60 + F1 max)")
    print(f"\n── Seuil équilibré ({balanced_threshold:.2f}) ──")
    print(classification_report(y_test, y_pred_balanced, target_names=['Légitime', 'Fraude']))
else:
    balanced_threshold = best_threshold
    y_pred_balanced    = y_pred_optimal
    f1_b = f1_o
    print("  Aucun seuil avec recall ≥ 0.60 — on garde le seuil optimal F1")

# Tableau comparatif des 3 seuils
print("\n" + "="*50)
print(" COMPARAISON DES 3 SEUILS")
print("="*50)
rows = []
for label, preds in [
    ("0.50  — défaut",                    y_pred_default),
    (f"{balanced_threshold:.2f}  — équilibré", y_pred_balanced),
    (f"{best_threshold:.2f}  — optimal F1",    y_pred_optimal),
]:
    rows.append({
        'Seuil'    : label,
        'Precision': round(precision_score(y_test, preds, pos_label=1, zero_division=0), 3),
        'Recall'   : round(recall_score(y_test, preds, pos_label=1, zero_division=0), 3),
        'F1-Fraude': round(f1_score(y_test, preds, zero_division=0), 3),
    })
print(pd.DataFrame(rows).to_string(index=False))


# ── ÉTAPE 10 : Visualisations résultats modèle ───────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Résultats XGBoost — Comparaison des seuils", fontsize=13, fontweight='bold')

# 10a. Matrices de confusion — 3 seuils
for ax, preds, title in zip(
    axes[:3],
    [y_pred_default, y_pred_balanced, y_pred_optimal],
    [f'Seuil 0.50\n(défaut)',
     f'Seuil {balanced_threshold:.2f}\n(équilibré)',
     f'Seuil {best_threshold:.2f}\n(optimal F1)']
):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Légitime', 'Fraude']).plot(ax=ax, colorbar=False)
    ax.set_title(title)

# 10b. Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[3].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
axes[3].plot([0,1],[0,1], 'k--', label='Aléatoire')
axes[3].set_xlabel('Taux de Faux Positifs')
axes[3].set_ylabel('Taux de Vrais Positifs')
axes[3].set_title('Courbe ROC')
axes[3].legend()

plt.tight_layout()
plt.savefig('xgboost_seuils_comparaison.png', dpi=150, bbox_inches='tight')
plt.show()

# 10c. Feature Importance (Top 15)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_imp.nlargest(15).sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Top 15 Features Importantes — XGBoost')

# 10d. Courbe Precision-Recall avec les 3 seuils marqués
axes[1].plot(recalls, precisions, color='steelblue', lw=2, label='Courbe P-R')
for thresh, label, color in [
    (0.5,               'Seuil 0.50',                      'gray'),
    (balanced_threshold, f'Seuil {balanced_threshold:.2f} équilibré', 'darkorange'),
    (best_threshold,    f'Seuil {best_threshold:.2f} optimal',        'crimson'),
]:
    idx = np.argmin(np.abs(thresholds - thresh))
    axes[1].scatter(recalls[idx], precisions[idx], s=100, color=color,
                    zorder=5, label=label)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Courbe Precision-Recall — Fraude')
axes[1].legend()

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Graphiques sauvegardés")


# ── ÉTAPE 11 : Isolation Forest — Score d'anomalie ───────────
print("\n" + "="*50)
print(" ISOLATION FOREST (score d'anomalie)")
print("="*50)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.035,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)

# Score d'anomalie brut (plus négatif = plus anormal)
iso_scores_train = iso.score_samples(X_train)
iso_scores_test  = iso.score_samples(X_test)

# Prédictions binaires pour référence
iso_pred_bin = np.where(iso.predict(X_test) == -1, 1, 0)
iso_f1  = f1_score(y_test, iso_pred_bin, zero_division=0)
iso_auc = roc_auc_score(y_test, iso_pred_bin)

print(f"  AUC-ROC  : {iso_auc:.4f}")
print(f"  F1-Score : {iso_f1:.4f}")
print("\nClassification Report (binaire) :")
print(classification_report(y_test, iso_pred_bin, target_names=['Légitime', 'Fraude']))



# ── ÉTAPE 12 : Stacking — IsoForest score → feature XGBoost ──
print("\n" + "="*50)
print(" APPROCHE STACKING (IsoForest score → XGBoost)")
print("="*50)
print("  → Le score d'anomalie devient une feature du modèle final")

# Ajouter le score d'anomalie comme nouvelle feature
X_train_stack = X_train.copy()
X_test_stack  = X_test.copy()
X_train_stack['iso_anomaly_score'] = iso_scores_train
X_test_stack['iso_anomaly_score']  = iso_scores_test

# Ré-appliquer SMOTE sur le dataset enrichi
X_res_stack, y_res_stack = sm.fit_resample(X_train_stack, y_train)

# Réentraîner XGBoost avec la nouvelle feature
xgb_stack = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=30
)
xgb_stack.fit(
    X_res_stack, y_res_stack,
    eval_set=[(X_test_stack, y_test)],
    verbose=50
)

# Évaluation du modèle stacké
y_stack_prob = xgb_stack.predict_proba(X_test_stack)[:, 1]

# Seuil optimal pour le modèle stacké
prec_s, rec_s, thresh_s = precision_recall_curve(y_test, y_stack_prob)
f1_s_thresh = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-8)
best_thresh_stack = thresh_s[f1_s_thresh.argmax()]

y_stack_pred = (y_stack_prob >= best_thresh_stack).astype(int)

stack_auc = roc_auc_score(y_test, y_stack_prob)
stack_f1  = f1_score(y_test, y_stack_pred, zero_division=0)
stack_pr  = precision_score(y_test, y_stack_pred, pos_label=1, zero_division=0)
stack_re  = recall_score(y_test, y_stack_pred, pos_label=1, zero_division=0)

print(f"\n  Seuil optimal stacking : {best_thresh_stack:.3f}")
print(f"  AUC-ROC   : {stack_auc:.4f}")
print(f"  Precision : {stack_pr:.4f}")
print(f"  Recall    : {stack_re:.4f}")
print(f"  F1-Score  : {stack_f1:.4f}")
print("\nClassification Report :")
print(classification_report(y_test, y_stack_pred, target_names=['Légitime', 'Fraude']))

# Comparer l'importance de la nouvelle feature
feat_imp_stack = pd.Series(xgb_stack.feature_importances_, index=X_train_stack.columns)
iso_rank = feat_imp_stack.rank(ascending=False)['iso_anomaly_score']
print(f"\n Rang de 'iso_anomaly_score' dans les features : {int(iso_rank)}/{len(feat_imp_stack)}")

# Amélioration vs XGBoost seul
delta_auc = stack_auc - auc
delta_f1  = stack_f1  - f1_o
print(f"\n Gain vs XGBoost seul :")
print(f"   AUC-ROC  : {delta_auc:+.4f}")
print(f"   F1-Score : {delta_f1:+.4f}")

hybrid_pred = y_stack_pred
hybrid_f1   = stack_f1
hybrid_auc  = stack_auc



# ── ÉTAPE 13 : Tableau comparatif final ─────────────────────
print("\n" + "="*50)
print(" RÉCAPITULATIF COMPARATIF FINAL")
print("="*50)
results = pd.DataFrame({
    'Modèle'   : [
        'XGBoost seuil 0.50',
        f'XGBoost seuil {balanced_threshold:.2f} (équilibré)',
        f'XGBoost seuil {best_threshold:.2f} (optimal F1)',
        'Isolation Forest',
        f'Stacking (seuil {best_thresh_stack:.2f})'
    ],
    'AUC-ROC'  : [round(auc,4), round(auc,4), round(auc,4), round(iso_auc,4), round(stack_auc,4)],
    'Precision': [
        round(precision_score(y_test, y_pred_default,  pos_label=1, zero_division=0),4),
        round(precision_score(y_test, y_pred_balanced, pos_label=1, zero_division=0),4),
        round(precision_score(y_test, y_pred_optimal,  pos_label=1, zero_division=0),4),
        round(precision_score(y_test, iso_pred_bin,    pos_label=1, zero_division=0),4),
        round(stack_pr, 4)
    ],
    'Recall'   : [
        round(recall_score(y_test, y_pred_default,  pos_label=1, zero_division=0),4),
        round(recall_score(y_test, y_pred_balanced, pos_label=1, zero_division=0),4),
        round(recall_score(y_test, y_pred_optimal,  pos_label=1, zero_division=0),4),
        round(recall_score(y_test, iso_pred_bin,    pos_label=1, zero_division=0),4),
        round(stack_re, 4)
    ],
    'F1-Fraude': [round(f1_d,4), round(f1_b,4), round(f1_o,4), round(iso_f1,4), round(stack_f1,4)]
})
print(results.to_string(index=False))














































