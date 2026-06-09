import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# LightGBM, XGBoost ve CatBoost import
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nðŸ“‹ Train sÃ¼tunlarÄ±:")
print(train.columns.tolist())
print(f"\nðŸ“‹ Test sÃ¼tunlarÄ±:")
print(test.columns.tolist())
print(f"\nÄ°lk 5 satÄ±r:")
print(train.head())
print(f"\nVeri tipleri:")
print(train.dtypes)


print("\nðŸ“Š Veri analizi yapÄ±lÄ±yor...")
print("\nEksik deÄŸerler:")
print(train.isnull().sum())

print("\nHedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±:") 



# Hedef deÄŸiÅŸkeni bul
target_col = None
if 'Exited' in train.columns:
    target_col = 'Exited'
elif 'target' in train.columns:
    target_col = 'target'
else:
    # Son sÃ¼tunu hedef olarak kabul et
    target_col = train.columns[-1]


print(f" Hedef deÄŸiÅŸken: {target_col}")
print(f"Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±:")
print(train[target_col].value_counts())
print(f"{target_col} oranÄ±: {train[target_col].mean():.2%}")



# SayÄ±sal ve kategorik deÄŸiÅŸkenleri ayÄ±r
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train.select_dtypes(include=['object']).columns.tolist()



# id ve target'Ä± Ã§Ä±kar
exclude_cols = ['id', target_col, 'CustomerId']
numeric_features = [col for col in numeric_features if col not in exclude_cols]
categorical_features = [col for col in categorical_features if col not in exclude_cols]



print(f" SayÄ±sal Ã¶zellikler ({len(numeric_features)}): {numeric_features}")
print(f" Kategorik Ã¶zellikler ({len(categorical_features)}): {categorical_features}")




def feature_engineering(df, train_mode=True):
    """Yeni Ã¶zellikler oluÅŸtur - dinamik sÃ¼tun kontrolÃ¼ ile"""
    df = df.copy()
    
    # YaÅŸ ile ilgili Ã¶zellikler
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                 labels=['Young', 'Middle', 'Senior', 'Elder'])
        df['Age_Squared'] = df['Age'] ** 2
    
    # Kredi skoru ile ilgili Ã¶zellikler
    if 'CreditScore' in df.columns:
        df['CreditScoreGroup'] = pd.cut(df['CreditScore'], 
                                         bins=[0, 580, 670, 740, 900],
                                         labels=['Poor', 'Fair', 'Good', 'Excellent'])
        df['CreditScore_Normalized'] = (df['CreditScore'] - df['CreditScore'].mean()) / df['CreditScore'].std()
    
    # Bakiye ile ilgili Ã¶zellikler
    if 'Balance' in df.columns:
        df['HasBalance'] = (df['Balance'] > 0).astype(int)
        df['Balance_Log'] = np.log1p(df['Balance'])
        
        if 'NumOfProducts' in df.columns:
            df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)
        
        if 'EstimatedSalary' in df.columns:
            df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
            df['Salary_Log'] = np.log1p(df['EstimatedSalary'])
    
    # Aktivite Ã¶zellikleri
    if 'IsActiveMember' in df.columns and 'HasCrCard' in df.columns:
        df['IsActiveMember_HasCard'] = df['IsActiveMember'] * df['HasCrCard']
    
    # Tenure ile ilgili Ã¶zellikler
    if 'Tenure' in df.columns:
        if 'Age' in df.columns:
            df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)
            df['Age_Tenure'] = df['Age'] * df['Tenure']
        if 'NumOfProducts' in df.columns:
            df['Products_Tenure'] = df['NumOfProducts'] * df['Tenure']
    
    # Demografik kombinasyonlar
    if 'Geography' in df.columns and 'Gender' in df.columns:
        df['Geography_Gender'] = df['Geography'].astype(str) + '_' + df['Gender'].astype(str)
    
    # ÃœrÃ¼n sayÄ±sÄ± Ã¶zellikleri
    if 'NumOfProducts' in df.columns:
        df['HasMultipleProducts'] = (df['NumOfProducts'] > 1).astype(int)
    
    return df



# Ã–zellik mÃ¼hendisliÄŸi uygula
train_fe = feature_engineering(train, train_mode=True)
test_fe = feature_engineering(test, train_mode=False)

print(f"Yeni Ã¶zellikler eklendi")
print(f"Train shape: {train_fe.shape}")
print(f"Test shape: {test_fe.shape}")



# Kategorik deÄŸiÅŸkenleri encode et
le_dict = {}
categorical_cols_to_encode = []

# TÃ¼m kategorik sÃ¼tunlarÄ± bul
for col in train_fe.columns:
    if train_fe[col].dtype == 'object' or col.endswith('Group'):
        if col not in ['id', 'CustomerId']:
            categorical_cols_to_encode.append(col)

print(f"Encode edilecek kategorik sÃ¼tunlar: {categorical_cols_to_encode}")




for col in categorical_cols_to_encode:
    if col in train_fe.columns and col in test_fe.columns:
        le = LabelEncoder()
        train_fe[col] = le.fit_transform(train_fe[col].astype(str))
        test_fe[col] = le.transform(test_fe[col].astype(str))
        le_dict[col] = le

# Train ve test verilerini hazÄ±rla
exclude_cols = ['id', target_col, 'CustomerId']
feature_cols = [col for col in train_fe.columns if col not in exclude_cols]

X = train_fe[feature_cols]
y = train_fe[target_col]
X_test = test_fe[feature_cols]

print(f"Ã–zellik sayÄ±sÄ±: {len(feature_cols)}")
print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")



# Stratified K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Model parametreleri
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False
}

# Ensemble iÃ§in tahminler
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))

lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
cat_test = np.zeros(len(X_test))


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"  Fold {fold + 1}/{n_splits}")
    print('='*60)
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_test += lgb_model.predict(X_test) / n_splits
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=1000,
        evals=[(xgb_train, 'train'), (xgb_val, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    xgb_oof[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
    xgb_test += xgb_model.predict(xgb.DMatrix(X_test)) / n_splits
    
    # CatBoost
    print("\nTraining CatBoost...")
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    
    cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    cat_test += cat_model.predict_proba(X_test)[:, 1] / n_splits
    
    # Fold sonuÃ§larÄ±
    lgb_auc = roc_auc_score(y_val, lgb_oof[val_idx])
    xgb_auc = roc_auc_score(y_val, xgb_oof[val_idx])
    cat_auc = roc_auc_score(y_val, cat_oof[val_idx])
    
    print(f"  Fold {fold + 1} SonuÃ§larÄ±:")
    print(f"  LightGBM AUC: {lgb_auc:.5f}")
    print(f"  XGBoost AUC:  {xgb_auc:.5f}")
    print(f"  CatBoost AUC: {cat_auc:.5f}")



print(" Ensemble tahminleri oluÅŸturuluyor...")
# Weighted average ensemble
oof_preds = (lgb_oof * 0.4 + xgb_oof * 0.3 + cat_oof * 0.3)
test_preds = (lgb_test * 0.4 + xgb_test * 0.3 + cat_test * 0.3)



from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix



# OOF skorlarÄ±
lgb_oof_auc = roc_auc_score(y, lgb_oof)
xgb_oof_auc = roc_auc_score(y, xgb_oof)
cat_oof_auc = roc_auc_score(y, cat_oof)
ensemble_auc = roc_auc_score(y, oof_preds)

print(f"  Out-of-Fold AUC SkorlarÄ±:")
print(f"  LightGBM: {lgb_oof_auc:.5f}")
print(f"  XGBoost:  {xgb_oof_auc:.5f}")
print(f"  CatBoost: {cat_oof_auc:.5f}")
print(f"  Ensemble: {ensemble_auc:.5f}")




# Binary predictions iÃ§in threshold
threshold = 0.5
binary_preds = (oof_preds > threshold).astype(int)
accuracy = accuracy_score(y, binary_preds)

print(f"  Accuracy (threshold={threshold}): {accuracy:.5f}")
print("  Classification Report:")
print(classification_report(y, binary_preds))




# Confusion Matrix
cm = confusion_matrix(y, binary_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix - Ensemble Model\nAUC: {ensemble_auc:.4f}, Accuracy: {accuracy:.4f}')
plt.ylabel('GerÃ§ek DeÄŸer')
plt.xlabel('Tahmin')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close() 




# ID sÃ¼tununu bul
id_col = 'id' if 'id' in test_fe.columns else test_fe.columns[0]

# Ana submission
submission = pd.DataFrame({
    id_col: test_fe[id_col],
    target_col: test_preds
})
submission.to_csv('submission.csv', index=False)



# Her model iÃ§in ayrÄ± submission
lgb_submission = pd.DataFrame({
    id_col: test_fe[id_col],
    target_col: lgb_test
})
lgb_submission.to_csv('submission_lightgbm.csv', index=False)



xgb_submission = pd.DataFrame({
    id_col: test_fe[id_col],
    target_col: xgb_test
})
xgb_submission.to_csv('submission_xgboost.csv', index=False)
print(" 'submission_xgboost.csv' oluÅŸturuldu")




cat_submission = pd.DataFrame({
    id_col: test_fe[id_col],
    target_col: cat_test
})
cat_submission.to_csv('submission_catboost.csv', index=False)



print(f" Submission Ã¶nizleme:")
print(submission.head(10))
print(f" Submission istatistikleri:")
print(submission[target_col].describe())


print(f" FÄ°NAL SKORLAR:")
print(f" Ensemble AUC: {ensemble_auc:.5f}")
print(f" Accuracy: {accuracy:.5f}")
print(f" LightGBM AUC: {lgb_oof_auc:.5f}")
print(f" XGBoost AUC: {xgb_oof_auc:.5f}")
print(f" CatBoost AUC: {cat_oof_auc:.5f}")




