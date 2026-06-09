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


df = pd.read_csv('/kaggle/input/liver-guard-multi-class-prediction-for-cirrhosis/train.csv')
df.head(5)


import matplotlib.pyplot as plt
numeric_cols = df.select_dtypes(include='number').columns.drop('id')

for col in numeric_cols:
    # Histogram
    plt.figure(figsize=(6, 3))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f'{col} Histogram')
    plt.xlabel(col)
    plt.ylabel('Sayı')
    plt.tight_layout()
    plt.show()
    
    # Boxplot
    plt.figure(figsize=(6, 2))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f'{col} Boxplot')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
plt.matshow(corr, fignum=1)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.colorbar()
plt.title('Sayısal Değişkenler Korelasyon Matrisi', pad=20)
plt.tight_layout()
plt.show()


missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100
missing_df = pd.concat([missing_count, missing_percent], axis=1)
missing_df.columns = ['Eksik Adet', 'Eksik %']
print("=== Eksik Değer Tablosu ===\n", missing_df.sort_values('Eksik %', ascending=False))

plt.figure(figsize=(6, 3))
missing_percent.sort_values(ascending=False).plot.bar()
plt.title('Sütunlara Göre Eksik Değer %')
plt.ylabel('Yüzde')
plt.tight_layout()
plt.show()



cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop('Status')

df_encoded = df.copy()
df_encoded[cat_cols] = df_encoded[cat_cols].fillna('Missing')
df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

print("=== Örnek Kodlanmış Veri ===")
print(df_encoded.head())


y = df['Status']
X = df.drop('Status', axis=1)

print("X sütunları:", X.columns.tolist())
print("y sınıf dağılımı:\n", y.value_counts())


# —— BLOK 2: Eksik Değer İmputasyonu ——
from sklearn.impute import SimpleImputer

# 1) Numerik ve kategorik sütunları seçin
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

# 2) Medyan & en sık değer imputasyonu
num_imp = SimpleImputer(strategy='median')
X[num_cols] = num_imp.fit_transform(X[num_cols])

cat_imp = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

# 3) Kontrol
print("Eksik sayıları (örn.):\n", X.isnull().sum().loc[cat_cols+num_cols].head())



# —— BLOK 3: Kategorik Kodlama & Ölçekleme ——
from sklearn.preprocessing import StandardScaler

# 1) Kategorikleri “Missing” dolgu ile hazırla
X[cat_cols] = X[cat_cols].fillna('Missing')

# 2) One-Hot Encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 3) Sayısalları StandardScaler ile ölçeklendir
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 4) İlk 5 satırı inceleyin
print(X.head())



# —— BLOK 4: Eğitim/Doğrulama Setlerine Bölme ——
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Eğitim şekli:", X_train.shape, "– Doğrulama şekli:", X_val.shape)
print("y_train dağılımı:\n", y_train.value_counts(normalize=True))



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Varsayım: Önceki hücrede oluşturulmuş X_train, X_val, y_train, y_val kullanılacaktır.

# ——— BLOK 1: Lojistik Regresyon (Baseline) ———
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

print("=== Lojistik Regresyon Sonuçları ===")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))


# ——— BLOK 2: Random Forest ———
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("\n=== Random Forest Sonuçları ===")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))


# —— BLOK 3: XGBoost Eğitimi (Label Encoding ile) ——
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# 1) Hedef etiketleri sayısala çevir
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)   # ['C','CL','D'] → [0,1,2]
y_val_enc   = le.transform(y_val)

# 2) Modeli tanımla ve eğit
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train_enc)

# 3) Tahmin & Raporla
y_pred_xgb = xgb_clf.predict(X_val)
print("Label ↔ Kod eşlemesi:", dict(zip(le.classes_, le.transform(le.classes_))))
print("\n=== XGBoost Sonuçları ===")
print("Accuracy:", accuracy_score(y_val_enc, y_pred_xgb))
print(classification_report(
    y_val_enc,
    y_pred_xgb,
    target_names=le.classes_
))



# —— BLOK 4: XGBoost Hiperparametre Optimizasyonu ——
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth':    [3, 5],
    'learning_rate':[0.01, 0.1]
}

grid = GridSearchCV(
    xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Burada da y_train_enc kullanılmalı
grid.fit(X_train, y_train_enc)
best = grid.best_estimator_

# En iyi modelin performansı
y_pred_best = best.predict(X_val)
print("\n=== Hiperparametreli XGBoost Sonuçları ===")
print("En iyi parametreler:", grid.best_params_)
print("Accuracy:", accuracy_score(y_val_enc, y_pred_best))
print(classification_report(
    y_val_enc,
    y_pred_best,
    target_names=le.classes_
))


