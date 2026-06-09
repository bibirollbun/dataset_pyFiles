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


# -------------------------- ä¼˜åŒ–ç‰ˆï¼šå®Œæ•´æµ�ç¨‹ --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# è®¾ç½®ä¸­æ–‡å­—ä½“
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=== ä¼˜åŒ–ç‰ˆï¼šå†…å¤–å�‘äººæ ¼é¢„æµ‹ Kaggleæ¯”èµ›è§£å†³æ–¹æ¡ˆ ===")

# =============================================================================
# 1. å¢�å¼ºç‰ˆæ•°æ�®åˆ†æ��
# =============================================================================
print("\n1. å¢�å¼ºç‰ˆæ•°æ�®åˆ†æ��é˜¶æ®µ")

# æ•°æ�®åŠ è½½
input_dir = '/kaggle/input/playground-series-s5e7'
train_df = pd.read_csv(f'{input_dir}/train.csv')
test_df = pd.read_csv(f'{input_dir}/test.csv')
sample_sub = pd.read_csv(f'{input_dir}/sample_submission.csv')

print(f"è®­ç»ƒé›†å½¢çŠ¶: {train_df.shape}")
print(f"æµ‹è¯•é›†å½¢çŠ¶: {test_df.shape}")

# è¯¦ç»†çš„æ•°æ�®æ�¢ç´¢
print("\nè®­ç»ƒé›†åŸºæœ¬ä¿¡æ�¯:")
print(train_df.info())

# ç›®æ ‡å�˜é‡�åˆ†æ��
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
target_dist = train_df['Personality'].value_counts()
plt.pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('ç›®æ ‡å�˜é‡�åˆ†å¸ƒ')

plt.subplot(1, 3, 2)
sns.countplot(data=train_df, x='Personality')
plt.title('äººæ ¼ç±»å�‹æ•°é‡�åˆ†å¸ƒ')
plt.ylabel('æ•°é‡�')

plt.subplot(1, 3, 3)
# è®¡ç®—æ•°å€¼ç‰¹å¾�çš„å�‡å€¼æŒ‰äººæ ¼ç±»å�‹åˆ†ç»„
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    mean_by_personality = train_df.groupby('Personality')[numeric_cols].mean().mean(axis=1)
    plt.bar(mean_by_personality.index, mean_by_personality.values)
    plt.title('æ•°å€¼ç‰¹å¾�å�‡å€¼å¯¹æ¯”')
    plt.ylabel('å¹³å�‡ç‰¹å¾�å€¼')

plt.tight_layout()
plt.show()

# ç‰¹å¾�ç›¸å…³æ€§åˆ†æ��
if len(numeric_cols) > 2:
    plt.figure(figsize=(12, 10))
    correlation_matrix = train_df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('ç‰¹å¾�ç›¸å…³æ€§çƒ­å›¾')
    plt.tight_layout()
    plt.show()

# ç¼ºå¤±å€¼åˆ†æ��
missing_data = train_df.isnull().sum()
if missing_data.sum() > 0:
    plt.figure(figsize=(10, 6))
    missing_data[missing_data > 0].plot(kind='bar')
    plt.title('ç¼ºå¤±å€¼åˆ†å¸ƒ')
    plt.ylabel('ç¼ºå¤±æ•°é‡�')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. å¢�å¼ºç‰ˆæ•°æ�®é¢„å¤„ç�†
# =============================================================================
print("\n2. å¢�å¼ºç‰ˆæ•°æ�®é¢„å¤„ç�†é˜¶æ®µ")

# åˆ›å»ºæ•°æ�®å‰¯æœ¬
train_processed = train_df.copy()
test_processed = test_df.copy()

# åˆ†ç¦»ç‰¹å¾�å’Œç›®æ ‡
X = train_processed.drop(['id', 'Personality'], axis=1)
y = train_processed['Personality']
X_test = test_processed.drop('id', axis=1)

# è¯†åˆ«ç‰¹å¾�ç±»å�‹
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"æ•°å€¼ç‰¹å¾�: {numeric_features}")
print(f"åˆ†ç±»ç‰¹å¾�: {categorical_features}")

# é«˜çº§ç¼ºå¤±å€¼å¤„ç�†
def advanced_imputation(X_train, X_test, numeric_features, categorical_features):
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    
    # æ•°å€¼ç‰¹å¾�ï¼šä½¿ç”¨æ›´å¤�æ�‚çš„ç­–ç•¥
    if numeric_features:
        num_imputer = SimpleImputer(strategy='median')
        X_train_imp[numeric_features] = num_imputer.fit_transform(X_train[numeric_features])
        X_test_imp[numeric_features] = num_imputer.transform(X_test[numeric_features])
    
    # åˆ†ç±»ç‰¹å¾�ï¼šä½¿ç”¨ä¼—æ•°å¹¶æ·»åŠ ç¼ºå¤±æŒ‡ç¤ºå™¨
    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_imp[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
        X_test_imp[categorical_features] = cat_imputer.transform(X_test[categorical_features])
        
        # ä¸ºåˆ†ç±»ç‰¹å¾�æ·»åŠ ç¼ºå¤±æŒ‡ç¤ºå™¨
        for col in categorical_features:
            if X_train[col].isnull().sum() > 0:
                X_train_imp[f'{col}_missing'] = X_train[col].isnull().astype(int)
                X_test_imp[f'{col}_missing'] = X_test[col].isnull().astype(int)
    
    return X_train_imp, X_test_imp

X_imputed, X_test_imputed = advanced_imputation(X, X_test, numeric_features, categorical_features)

# ç‰¹å¾�ç¼–ç �ä¼˜åŒ–
def advanced_encoding(X_train, X_test, categorical_features):
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # å¯¹åˆ†ç±»ç‰¹å¾�ä½¿ç”¨ç›®æ ‡ç¼–ç �ï¼ˆæ›´é«˜çº§çš„ç¼–ç �æ–¹å¼�ï¼‰
    for col in categorical_features:
        if col in X_train.columns:
            # è®¡ç®—æ¯�ä¸ªç±»åˆ«çš„ç›®æ ‡å�‡å€¼
            target_mean = train_processed.groupby(col)['Personality'].apply(
                lambda x: (x == 'Extrovert').mean()
            ).to_dict()
            
            X_train_enc[f'{col}_target_enc'] = X_train[col].map(target_mean)
            X_test_enc[f'{col}_target_enc'] = X_test[col].map(target_mean)
            
            # å¯¹äº�æœªçŸ¥ç±»åˆ«ï¼Œä½¿ç”¨å…¨å±€å�‡å€¼
            global_mean = (train_processed['Personality'] == 'Extrovert').mean()
            X_train_enc[f'{col}_target_enc'].fillna(global_mean, inplace=True)
            X_test_enc[f'{col}_target_enc'].fillna(global_mean, inplace=True)
    
    # ç§»é™¤å�Ÿå§‹åˆ†ç±»ç‰¹å¾�
    X_train_enc = X_train_enc.drop(categorical_features, axis=1)
    X_test_enc = X_test_enc.drop(categorical_features, axis=1)
    
    return X_train_enc, X_test_enc

X_encoded, X_test_encoded = advanced_encoding(X_imputed, X_test_imputed, categorical_features)

# ç‰¹å¾�ç¼©æ”¾
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# ç›®æ ‡å�˜é‡�ç¼–ç �
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Introvert=0, Extrovert=1

print(f"å¤„ç�†å��è®­ç»ƒé›†å½¢çŠ¶: {X_scaled.shape}")
print(f"å¤„ç�†å��æµ‹è¯•é›†å½¢çŠ¶: {X_test_scaled.shape}")
print(f"ç›®æ ‡å�˜é‡�ç¼–ç �: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# =============================================================================
# 3. å¢�å¼ºç‰ˆæ¨¡å�‹è®­ç»ƒ
# =============================================================================
print("\n3. å¢�å¼ºç‰ˆæ¨¡å�‹è®­ç»ƒé˜¶æ®µ")

# æ•°æ�®æ‹†åˆ†
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"è®­ç»ƒé›†: {X_train.shape}, éªŒè¯�é›†: {X_val.shape}")

# å®šä¹‰å¤šä¸ªé«˜çº§æ¨¡å�‹
models = {
    'XGBoost': XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
}

# è®­ç»ƒå’Œè¯„ä¼°æ¨¡å�‹
results = {}
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(models.items()):
    print(f"\nè®­ç»ƒ {name}...")
    
    # è®­ç»ƒæ¨¡å�‹
    model.fit(X_train, y_train)
    
    # é¢„æµ‹
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # è®¡ç®—æŒ‡æ ‡
    accuracy = accuracy_score(y_val, y_val_pred)
    auc_score = roc_auc_score(y_val, y_val_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': y_val_pred,
        'probabilities': y_val_proba
    }
    
    print(f"{name} - å‡†ç¡®ç�‡: {accuracy:.4f}, AUC: {auc_score:.4f}")

# æ¨¡å�‹æ¯”è¾ƒå�¯è§†åŒ–
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# å‡†ç¡®ç�‡æ¯”è¾ƒ
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
auc_scores = [results[name]['auc'] for name in model_names]

axes[0, 0].bar(model_names, accuracies, color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
axes[0, 0].set_title('æ¨¡å�‹å‡†ç¡®ç�‡æ¯”è¾ƒ')
axes[0, 0].set_ylabel('å‡†ç¡®ç�‡')
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# AUCæ¯”è¾ƒ
axes[0, 1].bar(model_names, auc_scores, color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
axes[0, 1].set_title('æ¨¡å�‹AUCæ¯”è¾ƒ')
axes[0, 1].set_ylabel('AUCåˆ†æ•°')
for i, v in enumerate(auc_scores):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# ROCæ›²çº¿
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='éš�æœºåˆ†ç±»å™¨')
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_val, result['probabilities'])
    axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.4f})')
axes[1, 0].set_xlabel('å�‡æ­£ç�‡')
axes[1, 0].set_ylabel('çœŸæ­£ç�‡')
axes[1, 0].set_title('ROCæ›²çº¿æ¯”è¾ƒ')
axes[1, 0].legend()

# é€‰æ‹©æœ€ä½³æ¨¡å�‹çš„æ··æ·†çŸ©é˜µ
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_result = results[best_model_name]
cm = confusion_matrix(y_val, best_result['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
axes[1, 1].set_title(f'{best_model_name} æ··æ·†çŸ©é˜µ')
axes[1, 1].set_xlabel('é¢„æµ‹æ ‡ç­¾')
axes[1, 1].set_ylabel('çœŸå®�æ ‡ç­¾')

plt.tight_layout()
plt.show()

# äº¤å�‰éªŒè¯�
print(f"\nå¯¹æœ€ä½³æ¨¡å�‹ {best_model_name} è¿›è¡Œäº¤å�‰éªŒè¯�...")
best_model = results[best_model_name]['model']
cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5, scoring='roc_auc')
print(f"äº¤å�‰éªŒè¯�AUCåˆ†æ•°: {[f'{score:.4f}' for score in cv_scores]}")
print(f"å¹³å�‡AUC: {cv_scores.mean():.4f} (Â±{cv_scores.std()*2:.4f})")

# æ¨¡å�‹é›†æˆ�ï¼ˆè¿›ä¸€æ­¥æ��é«˜æ€§èƒ½ï¼‰
print("\nè®­ç»ƒé›†æˆ�æ¨¡å�‹...")
voting_model = VotingClassifier(
    estimators=[(name, results[name]['model']) for name in model_names],
    voting='soft'
)
voting_model.fit(X_train, y_train)
y_val_proba_ensemble = voting_model.predict_proba(X_val)[:, 1]
ensemble_auc = roc_auc_score(y_val, y_val_proba_ensemble)
print(f"é›†æˆ�æ¨¡å�‹AUC: {ensemble_auc:.4f}")

# æ›´æ–°æœ€ä½³æ¨¡å�‹ä¸ºé›†æˆ�æ¨¡å�‹
if ensemble_auc > results[best_model_name]['auc']:
    best_model = voting_model
    best_model_name = 'Ensemble'
    print("é›†æˆ�æ¨¡å�‹è¡¨ç�°æœ€ä½³ï¼�")
else:
    best_model = results[best_model_name]['model']

# =============================================================================
# 4. ç»“æ�œé¢„æµ‹å’Œæ��äº¤
# =============================================================================
print("\n4. ç»“æ�œé¢„æµ‹å’Œæ��äº¤é˜¶æ®µ")

# ä½¿ç”¨æœ€ä½³æ¨¡å�‹è¿›è¡Œé¢„æµ‹
final_predictions = best_model.predict(X_test_scaled)
final_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

# è½¬æ�¢å›�å�Ÿå§‹æ ‡ç­¾
final_predictions_labels = label_encoder.inverse_transform(final_predictions)

# åˆ›å»ºæ��äº¤æ–‡ä»¶
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_predictions_labels
})

# ä¿�å­˜æ��äº¤æ–‡ä»¶
submission.to_csv('submission_optimized.csv', index=False)

# é¢„æµ‹ç»“æ�œåˆ†æ��
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
pred_counts = pd.Series(final_predictions_labels).value_counts()
plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('æµ‹è¯•é›†é¢„æµ‹åˆ†å¸ƒ')

plt.subplot(1, 2, 2)
plt.hist(final_probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='åˆ†ç±»é˜ˆå€¼')
plt.xlabel('å¤–å�‘æ¦‚ç�‡')
plt.ylabel('é¢‘æ•°')
plt.title('å¤–å�‘æ¦‚ç�‡åˆ†å¸ƒ')
plt.legend()

plt.tight_layout()
plt.show()

# æœ€ç»ˆæ€»ç»“
print("\n" + "="*60)
print("ğŸ�‰ğŸ�‰ ä¼˜åŒ–ç‰ˆæµ�ç¨‹å®Œæˆ�æ€»ç»“")
print("="*60)
print(f"æœ€ä½³æ¨¡å�‹: {best_model_name}")
print(f"éªŒè¯�é›†AUC: {max(results[best_model_name]['auc'] if best_model_name != 'Ensemble' else ensemble_auc, ensemble_auc):.4f}")
print(f"äº¤å�‰éªŒè¯�å¹³å�‡AUC: {cv_scores.mean():.4f}")
print(f"é¢„æµ‹æ ·æœ¬åˆ†å¸ƒ:")
print(f"  - Introvert: {(final_predictions == 0).sum()} ({(final_predictions == 0).mean():.2%})")
print(f"  - Extrovert: {(final_predictions == 1).sum()} ({(final_predictions == 1).mean():.2%})")
print(f"æ��äº¤æ–‡ä»¶å·²ä¿�å­˜: submission_optimized.csv")
print("="*60)

# ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��ï¼ˆå¦‚æ�œæ¨¡å�‹æ”¯æŒ�ï¼‰
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��')
    plt.xlabel('é‡�è¦�æ€§')
    plt.tight_layout()
    plt.show()

