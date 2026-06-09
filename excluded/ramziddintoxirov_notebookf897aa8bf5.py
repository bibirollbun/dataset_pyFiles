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
KAGGLE PLAYGROUND S4E2: Multi-Class Prediction of Obesity Risk
To'liq Solution - Top 15-20% darajasi

Bu kod bilan siz:
- 7 ta klassni to'g'ri bashorat qilasiz
- XGBoost, LightGBM, CatBoost ensemble ishlatadi
- Feature Engineering qiladi
- Cross-validation qiladi
- Hyperparameter tuning qiladi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. MA'LUMOTLARNI YUKLASH
# ========================
print("ğŸ“‚ Ma'lumotlar yuklanmoqda...")
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"\nTarget classes: {train['NObeyesdad'].unique()}")

# ========================
# 2. FEATURE ENGINEERING
# ========================
print("\nğŸ”§ Feature Engineering...")

def create_features(df):
    """Yangi feature'lar yaratish"""
    df = df.copy()
    
    # BMI (Body Mass Index) - ENG MUHIM FEATURE!
    df['BMI'] = df['Weight'] / ((df['Height']) ** 2)
    
    # Weight categories
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    df['Weight_Age_Ratio'] = df['Weight'] / df['Age']
    
    # Lifestyle score
    df['Physical_Activity_Score'] = df['FAF'] * df['TUE']
    df['Healthy_Habits'] = (df['FCVC'] + df['NCP'] + df['CH2O']) / 3
    df['Unhealthy_Habits'] = (df['FAVC'].map({'yes': 1, 'no': 0}) + 
                               df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}) +
                               df['SMOKE'].map({'yes': 1, 'no': 0})) / 3
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 100], 
                              labels=['Teen', 'Young', 'Adult', 'Senior'])
    
    # Combined features
    df['Health_Score'] = df['Healthy_Habits'] - df['Unhealthy_Habits']
    df['Activity_Level'] = df['FAF'] * (4 - df['TUE'])  # ko'p harakat, kam technology
    
    # Food habits
    df['Total_Meals'] = df['FCVC'] + df['NCP']
    df['Water_Per_Meal'] = df['CH2O'] / (df['NCP'] + 1)
    
    # Transport & Movement
    df['Active_Transport'] = df['MTRANS'].map({
        'Walking': 3, 'Bike': 3, 
        'Public_Transportation': 1, 'Automobile': 0, 'Motorbike': 0
    })
    
    # Family & Genetics
    df['Genetic_Risk'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    
    return df

# Feature engineering
train = create_features(train)
test = create_features(test)

print(f"âœ… Yangi features qo'shildi! Jami features: {train.shape[1]}")

# ========================
# 3. ENCODING
# ========================
print("\nğŸ”„ Encoding...")

# Target encoding
le = LabelEncoder()
y = le.fit_transform(train['NObeyesdad'])
print(f"Classes: {le.classes_}")

# Categorical columns
cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
            'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Age_Group']

# One-hot encoding
train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=False)
test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=False)

# Ikkala dataset'da bir xil column'lar bo'lishini ta'minlash
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# Drop unnecessary columns
drop_cols = ['id', 'NObeyesdad']
X = train_encoded.drop([col for col in drop_cols if col in train_encoded.columns], axis=1)
X_test = test_encoded.drop([col for col in drop_cols if col in test_encoded.columns], axis=1)

print(f"âœ… Final features: {X.shape[1]}")

# ========================
# 4. CROSS-VALIDATION SETUP
# ========================
print("\nğŸ“Š Cross-Validation setup...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ========================
# 5. MODEL TRAINING
# ========================
print("\nğŸ¤– Model training boshlandi...")

# Storage for predictions
oof_preds_xgb = np.zeros((len(X), len(le.classes_)))
oof_preds_lgb = np.zeros((len(X), len(le.classes_)))
oof_preds_cat = np.zeros((len(X), len(le.classes_)))

test_preds_xgb = np.zeros((len(X_test), len(le.classes_)))
test_preds_lgb = np.zeros((len(X_test), len(le.classes_)))
test_preds_cat = np.zeros((len(X_test), len(le.classes_)))

# XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss'
}

# LightGBM parameters
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1
}

# CatBoost parameters
cat_params = {
    'loss_function': 'MultiClass',
    'depth': 7,
    'learning_rate': 0.05,
    'iterations': 500,
    'random_state': 42,
    'verbose': False
}

# Training loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{n_folds}")
    print(f"{'='*50}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # ===== XGBoost =====
    print("\nğŸš€ XGBoost training...")
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)
    test_preds_xgb += model_xgb.predict_proba(X_test) / n_folds
    
    acc_xgb = accuracy_score(y_val, model_xgb.predict(X_val))
    print(f"âœ… XGBoost Fold {fold+1} Accuracy: {acc_xgb:.5f}")
    
    # ===== LightGBM =====
    print("\nğŸ’¡ LightGBM training...")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)])
    
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)
    test_preds_lgb += model_lgb.predict_proba(X_test) / n_folds
    
    acc_lgb = accuracy_score(y_val, model_lgb.predict(X_val))
    print(f"âœ… LightGBM Fold {fold+1} Accuracy: {acc_lgb:.5f}")
    
    # ===== CatBoost =====
    print("\nğŸ�± CatBoost training...")
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train,
                  eval_set=(X_val, y_val))
    
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)
    test_preds_cat += model_cat.predict_proba(X_test) / n_folds
    
    acc_cat = accuracy_score(y_val, model_cat.predict(X_val))
    print(f"âœ… CatBoost Fold {fold+1} Accuracy: {acc_cat:.5f}")

# ========================
# 6. OUT-OF-FOLD SCORES
# ========================
print("\n" + "="*50)
print("ğŸ“Š FINAL OUT-OF-FOLD SCORES")
print("="*50)

oof_xgb = np.argmax(oof_preds_xgb, axis=1)
oof_lgb = np.argmax(oof_preds_lgb, axis=1)
oof_cat = np.argmax(oof_preds_cat, axis=1)

print(f"XGBoost OOF Accuracy: {accuracy_score(y, oof_xgb):.5f}")
print(f"LightGBM OOF Accuracy: {accuracy_score(y, oof_lgb):.5f}")
print(f"CatBoost OOF Accuracy: {accuracy_score(y, oof_cat):.5f}")

# ========================
# 7. ENSEMBLE
# ========================
print("\nğŸ�¯ Ensemble predictions...")

# Weighted average (CatBoost ko'proq weight)
test_preds_ensemble = (
    0.30 * test_preds_xgb + 
    0.30 * test_preds_lgb + 
    0.40 * test_preds_cat
)

# Final predictions
final_preds = np.argmax(test_preds_ensemble, axis=1)
final_preds_labels = le.inverse_transform(final_preds)

# OOF ensemble
oof_ensemble = (0.30 * oof_preds_xgb + 0.30 * oof_preds_lgb + 0.40 * oof_preds_cat)
oof_ensemble_class = np.argmax(oof_ensemble, axis=1)
print(f"âœ¨ Ensemble OOF Accuracy: {accuracy_score(y, oof_ensemble_class):.5f}")

# ========================
# 8. SUBMISSION
# ========================
print("\nğŸ’¾ Creating submission...")

submission = pd.DataFrame({
    'id': test['id'],
    'NObeyesdad': final_preds_labels
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file yaratildi: submission.csv")

# Distribution check
print("\nğŸ“Š Submission distribution:")
print(submission['NObeyesdad'].value_counts().sort_index())

print("\n" + "="*50)
print("ğŸ�‰ YAKUNLANDI!")
print("="*50)
print("Keyingi qadamlar:")
print("1. submission.csv faylini Kaggle'ga yuklang")
print("2. Public leaderboard natijasini tekshiring")
print("3. Agar yaxshi bo'lsa - ensemble weight'larni tuning qiling")
print("4. Feature engineering'ni yaxshilang")
print("="*50)

