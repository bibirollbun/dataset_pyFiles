import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv') 
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
 
print("Train shape:", train.shape)
print("Test shape:", test.shape) 
print(train.head())


train.info()


train.describe()


train.isnull().sum()


print("Yol Tipi:")
print(train['road_type'].value_counts())
print("\nYüzdelik Dağılım:")
print(train['road_type'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='road_type')
plt.title('Yol tipi', fontsize=14, fontweight='bold')
plt.xlabel('road_type')
plt.ylabel('Count')
plt.show()


print("Hava Durumu:")
print(train['weather'].value_counts())
print("\nYüzdelik Dağılım:")
print(train['weather'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='weather')
plt.title('Hava Durumu', fontsize=14, fontweight='bold')
plt.xlabel('weather')
plt.ylabel('Count')
plt.show()


print("Aydınlatma:")
print(train['lighting'].value_counts())
print("\nYüzdelik Dağılım:")
print(train['lighting'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='lighting')
plt.title('Aydınlatma', fontsize=14, fontweight='bold')
plt.xlabel('lighting')
plt.ylabel('Count')
plt.show()


print("Zaman Dilimi:")
print(train['time_of_day'].value_counts())
print("\nYüzdelik Dağılım:")
print(train['time_of_day'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='time_of_day')
plt.title('Zaman Dilimi', fontsize=14, fontweight='bold')
plt.xlabel('time_of_day')
plt.ylabel('Count')
plt.show()


print("Şerit Sayısı:")
print(train['num_lanes'].value_counts())
print("\nYüzdelik Dağılım:")
print(train['num_lanes'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='num_lanes')
plt.title('Şerit Sayısı:', fontsize=14, fontweight='bold')
plt.xlabel('num_lanes')
plt.ylabel('Count')
plt.show()


print("\n--- Hedef Değişken Dağılımı ---")
if 'accident_risk' in train.columns:
    print(train['accident_risk'].value_counts())
    
    plt.figure(figsize=(8, 5))
    train['accident_risk'].value_counts().plot(kind='bar')
    plt.title('Accident Risk Dağılımı')
    plt.xlabel('Risk Seviyesi')
    plt.ylabel('Frekans')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



# Sayısal ve kategorik değişkenleri ayır
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

if 'id' in numeric_cols:
    numeric_cols.remove('id')
if 'accident_risk' in numeric_cols:
    numeric_cols.remove('accident_risk')


print("\n--- Sayısal Değişkenler ---")
print(numeric_cols)

print("\n--- Kategorik Değişkenler ---")
print(categorical_cols)


# Sayısal değişkenlerin histogramları
if len(numeric_cols) > 0:
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        train[col].hist(bins=30, edgecolor='black')
        plt.title(f'{col} Dağılımı')
        plt.xlabel(col)
        plt.ylabel('Frekans')
    plt.tight_layout()
    plt.show()


# ID'yi kaydet
train_ids = train['id'] if 'id' in train.columns else None
test_ids = test['id'] if 'id' in test.columns else None


# ID sütununu kaldır
if 'id' in train.columns:
    train = train.drop('id', axis=1)
if 'id' in test.columns:
    test = test.drop('id', axis=1)



# Hedef değişkeni ayır
if 'accident_risk' in train.columns:
    y = train['accident_risk']
    X = train.drop('accident_risk', axis=1)
else:
    print("UYARI: Hedef değişken bulunamadı!")
    y = None
    X = train.copy()


# Kategorik değişkenleri encode et
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))
    
    if col in test.columns:
        test[col] = test[col].astype(str)
        # Test setinde yeni kategoriler varsa, en sık görüleni ata
        test[col] = test[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
        test[col] = label_encoders[col].transform(test[col])



for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].mean(), inplace=True)

for col in test.columns:
    if test[col].isnull().sum() > 0:
        test[col].fillna(test[col].mean(), inplace=True)

print("Eksik değerler dolduruldu.")


# Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
print(f"\nEğitim seti: {X_train.shape}")
print(f"Validasyon seti: {X_val.shape}")


# Random Forest modeli
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)



model.fit(X_train, y_train)


# Tahminler
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
    


print("\n--- Eğitim Seti Performansı ---")
print(f"MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")


print("\n--- Validasyon Seti Performansı ---")
print(f"MSE: {mean_squared_error(y_val, y_pred_val):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_val)):.4f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_val):.4f}")
print(f"R² Score: {r2_score(y_val, y_pred_val):.4f}")



# Gerçek vs Tahmin grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin (Validasyon)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_val - y_pred_val
plt.scatter(y_pred_val, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Residuals (Hata)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Özellik önemleri
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- En Önemli 10 Özellik ---")
print(feature_importance.head(10))


plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Önem')
plt.title('Top 15 Özellik Önemleri')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


test_predictions = model.predict(test_scaled)

print(f"\nTest seti tahmin sayısı: {len(test_predictions)}")
print("\nTahmin istatistikleri:")
print(f"Min: {test_predictions.min():.4f}")
print(f"Max: {test_predictions.max():.4f}")
print(f"Mean: {test_predictions.mean():.4f}")
print(f"Median: {np.median(test_predictions):.4f}")
print(f"Std: {test_predictions.std():.4f}")


sonuc=pd.DataFrame()
if len(test_predictions.shape) > 1:
    test_predictions = test_predictions.flatten()
sonuc['accident_risk']=test_predictions


test_cp = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_cp


sonuc['id']=test_cp['id']


sonuc['accident_risk']=sonuc['accident_risk'].astype('int32')


sonuc.to_csv('sonuc.csv',index=False)





