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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Veriyi yÃ¼kle
train_df = pd.read_csv('/kaggle/input/playground-series-s3e17/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e17/test.csv')

# Temel bilgileri gÃ¶ster
print("EÄ�Ä°TÄ°M VERÄ°SÄ° BOYUTU:", train_df.shape)
print("TEST VERÄ°SÄ° BOYUTU:", test_df.shape)


# EÄŸitim verisinin ilk 5 satÄ±rÄ±
train_df.head()


# Veri tipleri
train_df.dtypes


# BoÅŸ deÄŸer kontrolÃ¼
print("EÄ�Ä°TÄ°M VERÄ°SÄ° - BOÅ� DEÄ�ERLER:")
print(train_df.isnull().sum())


# Temel istatistikler
train_df.describe()


# Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±
print("Machine Failure DaÄŸÄ±lÄ±mÄ±:")
print(train_df['Machine failure'].value_counts())
print("\nYÃ¼zde DaÄŸÄ±lÄ±m:")
print(train_df['Machine failure'].value_counts(normalize=True))


# SÃ¼tun isimlerini temizle
def clean_column_names(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[\[\]]', '', regex=True)
    return df

train_df = clean_column_names(train_df)
test_df = clean_column_names(test_df)

print("Yeni sÃ¼tun isimleri:")
print(train_df.columns.tolist())


# TemizlenmiÅŸ verinin ilk 5 satÄ±rÄ±
train_df.head()


# Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ± grafiÄŸi
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Machine_failure')
plt.title('Machine Failure DaÄŸÄ±lÄ±mÄ±')
plt.xlabel('Machine Failure (0: Normal, 1: ArÄ±za)')
plt.ylabel('SayÄ±')
plt.show()


# Makine tiplerinin daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
train_df['Type'].value_counts().plot(kind='bar')
plt.title('Makine Tiplerinin DaÄŸÄ±lÄ±mÄ±')
plt.xlabel('Type')
plt.ylabel('SayÄ±')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
type_failure = train_df.groupby('Type')['Machine_failure'].mean()
type_failure.plot(kind='bar')
plt.title('Makine Tipine GÃ¶re ArÄ±za OranÄ±')
plt.xlabel('Type')
plt.ylabel('ArÄ±za OranÄ±')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# SayÄ±sal deÄŸiÅŸkenlerin listesi
numerical_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 
                  'Torque_Nm', 'Tool_wear_min']

# SayÄ±sal deÄŸiÅŸkenlerin daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    plt.hist(train_df[col], bins=50, alpha=0.7, color='skyblue')
    plt.title(f'{col} DaÄŸÄ±lÄ±mÄ±')
    plt.xlabel(col)
    plt.ylabel('Frekans')

plt.tight_layout()
plt.show()


# Boxplot ile aykÄ±rÄ± deÄŸerler
plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(train_df[col])
    plt.title(f'{col} Boxplot')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# Hedef deÄŸiÅŸkene gÃ¶re sayÄ±sal deÄŸiÅŸkenlerin daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=train_df, x='Machine_failure', y=col)
    plt.title(f'{col} - Machine Failure')
    plt.xlabel('Machine Failure')

plt.tight_layout()
plt.show()


# Korelasyon matrisi
plt.figure(figsize=(12, 10))
correlation_matrix = train_df[numerical_cols + ['Machine_failure']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korelasyon Matrisi')
plt.show()


# Yeni Ã¶zellikler oluÅŸturma
def create_features(df):
    # SÄ±caklÄ±k farkÄ±
    df['Temperature_difference_K'] = df['Process_temperature_K'] - df['Air_temperature_K']
    
    # GÃ¼Ã§ hesaplama (Tork * HÄ±z)
    df['Power_W'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
    
    # TakÄ±m aÅŸÄ±nma oranÄ±
    df['Tool_wear_rate'] = df['Tool_wear_min'] / (df['Rotational_speed_rpm'] + 1)  # 0'a bÃ¶lme hatasÄ± iÃ§in +1
    
    # SÄ±caklÄ±k oranÄ±
    df['Temperature_ratio'] = df['Process_temperature_K'] / df['Air_temperature_K']
    
    # HÄ±z bÃ¶lÃ¼ tork
    df['Speed_to_torque_ratio'] = df['Rotational_speed_rpm'] / (df['Torque_Nm'] + 1)
    
    return df

# Ã–zellikleri oluÅŸturma
train_df = create_features(train_df)
test_df = create_features(test_df)

print("Yeni oluÅŸturulan Ã¶zellikler:")
new_features = ['Temperature_difference_K', 'Power_W', 'Tool_wear_rate', 'Temperature_ratio', 'Speed_to_torque_ratio']
print(new_features)


# Yeni Ã¶zelliklerin daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(15, 12))

for i, col in enumerate(new_features, 1):
    plt.subplot(3, 2, i)
    plt.hist(train_df[col], bins=50, alpha=0.7, color='lightgreen')
    plt.title(f'{col} DaÄŸÄ±lÄ±mÄ±')
    plt.xlabel(col)
    plt.ylabel('Frekans')

plt.tight_layout()
plt.show()


# Yeni Ã¶zelliklerin hedef deÄŸiÅŸkenle iliÅŸkisi
plt.figure(figsize=(15, 12))

for i, col in enumerate(new_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=train_df, x='Machine_failure', y=col)
    plt.title(f'{col} - Machine Failure')
    plt.xlabel('Machine Failure')

plt.tight_layout()
plt.show()


# Type deÄŸiÅŸkeninin one-hot encoding
train_df_encoded = pd.get_dummies(train_df, columns=['Type'], prefix='Type')
test_df_encoded = pd.get_dummies(test_df, columns=['Type'], prefix='Type')

print("Encoding sonrasÄ± sÃ¼tunlar:")
print([col for col in train_df_encoded.columns if 'Type_' in col])


# Gerekli kÃ¼tÃ¼phaneleri iÃ§e aktar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# Ã–zellikleri ve hedef deÄŸiÅŸkeni belirle
feature_columns = [
    'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
    'Torque_Nm', 'Tool_wear_min', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
    'Temperature_difference_K', 'Power_W', 'Tool_wear_rate', 
    'Temperature_ratio', 'Speed_to_torque_ratio',
    'Type_H', 'Type_L', 'Type_M'
]

X = train_df_encoded[feature_columns]
y = train_df_encoded['Machine_failure']

print("KullanÄ±lan Ã¶zellikler:")
print(feature_columns)
print(f"\nÃ–zellik matrisi boyutu: {X.shape}")
print(f"Hedef deÄŸiÅŸken boyutu: {y.shape}")


# Veriyi bÃ¶l
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"EÄŸitim seti boyutu: {X_train.shape}")
print(f"Validasyon seti boyutu: {X_val.shape}")
print(f"EÄŸitim seti hedef daÄŸÄ±lÄ±mÄ±: {y_train.value_counts(normalize=True)}")
print(f"Validasyon seti hedef daÄŸÄ±lÄ±mÄ±: {y_val.value_counts(normalize=True)}")


# Ã–zellik Ã¶lÃ§eklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Test seti iÃ§in de Ã¶lÃ§eklendirme
X_test = test_df_encoded[feature_columns]
X_test_scaled = scaler.transform(X_test)

print("Ã–lÃ§eklendirme tamamlandÄ±")
print(f"EÄŸitim seti Ã¶lÃ§eklendirilmiÅŸ boyut: {X_train_scaled.shape}")


# Random Forest modeli
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Dengesiz sÄ±nÄ±f iÃ§in
)

# Modeli eÄŸit
rf_model.fit(X_train, y_train)

# Tahminler
y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

# AUC skoru
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"Random Forest AUC Score: {auc_score:.4f}")


# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Feature Importance grafiÄŸi
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("En Ã–nemli 10 Ã–zellik:")
print(feature_importance.head(10))


# XGBoost modeli
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=10  # Dengesiz sÄ±nÄ±f iÃ§in
)

# Modeli eÄŸit
xgb_model.fit(X_train, y_train)

# Tahminler
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

# AUC skoru
auc_score_xgb = roc_auc_score(y_val, y_pred_proba_xgb)
print(f"XGBoost AUC Score: {auc_score_xgb:.4f}")
print(f"Random Forest AUC Score: {auc_score:.4f}")


# Ä°ki modeli karÅŸÄ±laÅŸtÄ±rma
models_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'AUC_Score': [auc_score, auc_score_xgb]
})

plt.figure(figsize=(8, 6))
sns.barplot(data=models_comparison, x='Model', y='AUC_Score')
plt.title('Model KarÅŸÄ±laÅŸtÄ±rmasÄ± - AUC Score')
plt.ylabel('AUC Score')
plt.ylim(0.8, 1.0)
for i, v in enumerate(models_comparison['AUC_Score']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.show()

print(models_comparison)


# En iyi modeli seÃ§ (burada XGBoost varsayalÄ±m)
best_model = xgb_model if auc_score_xgb > auc_score else rf_model

# Test seti iÃ§in tahminler
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Submission dosyasÄ± oluÅŸturma
submission = pd.DataFrame({
    'id': test_df['id'],
    'Machine failure': test_predictions
})

print("Submission dosyasÄ± ilk 5 satÄ±rÄ±:")
print(submission.head())


# Submission dosyasÄ±nÄ± kaydet
submission.to_csv('submission.csv', index=False)
print("Submission dosyasÄ± kaydedildi!")
print(f"Submission dosyasÄ± boyutu: {submission.shape}")


# Test seti tahminlerinin daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(10, 6))
plt.hist(test_predictions, bins=50, alpha=0.7, color='purple')
plt.title('Test Seti Tahminlerinin DaÄŸÄ±lÄ±mÄ±')
plt.xlabel('Tahmin Edilen ArÄ±za OlasÄ±lÄ±ÄŸÄ±')
plt.ylabel('Frekans')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='0.5 EÅŸiÄŸi')
plt.legend()
plt.show()

print(f"Tahminlerin istatistikleri:")
print(f"Min: {test_predictions.min():.4f}")
print(f"Max: {test_predictions.max():.4f}")
print(f"Ortalama: {test_predictions.mean():.4f}")
print(f"Medyan: {np.median(test_predictions):.4f}")


# Hiperparametre optimizasyonu iÃ§in Ã¶rnek (Grid Search)
from sklearn.model_selection import GridSearchCV

# KÃ¼Ã§Ã¼k bir grid search Ã¶rneÄŸi (hÄ±zlÄ± olmasÄ± iÃ§in sÄ±nÄ±rlÄ± parametre)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'min_samples_split': [2, 5]
}

# KÃ¼Ã§Ã¼k bir Ã¶rnek Ã¼zerinde Ã§alÄ±ÅŸtÄ±r (hÄ±zlÄ± olmasÄ± iÃ§in)
rf_small = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_small.fit(X_train.head(1000), y_train.head(1000))  # KÃ¼Ã§Ã¼k Ã¶rnek

print("Hiperparametre optimizasyonu iÃ§in hazÄ±r!")
print("Tam optimizasyon iÃ§in daha fazla zaman gerekir.")


print("ğŸ�† PROJE Ã–ZETÄ° ğŸ�†")
print("="*50)
print(f"ğŸ“Š Veri Seti Boyutu: {train_df.shape[0]} satÄ±r, {train_df.shape[1]} sÃ¼tun")
print(f"ğŸ�¯ Hedef DeÄŸiÅŸken Dengesi: {y.mean():.3f} arÄ±za oranÄ±")
print(f"âš™ï¸�  OluÅŸturulan Ã–zellik SayÄ±sÄ±: {len(feature_columns)}")
print(f"ğŸ“ˆ En Ä°yi Model AUC Skoru: {max(auc_score, auc_score_xgb):.4f}")
print(f"ğŸ“� Submission dosyasÄ± hazÄ±r: submission.csv")
print("="*50)
print("âœ… Proje tamamlandÄ±!")


# Modeli kaydet
import joblib
joblib.dump(rf_model, 'model.pkl')
print("Model baÅŸarÄ±yla kaydedildi!")




