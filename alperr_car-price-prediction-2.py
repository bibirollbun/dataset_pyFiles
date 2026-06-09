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


train_data = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


train_data.info()


train_data.isnull().sum()


missing_train_data = train_data.isna().mean()*100
missing_test_data = test_data.isna().mean()*100


print(missing_train_data)
print("\n")
print(missing_test_data)


df_train = train_data.copy()
df_test = test_data.copy()


df_train['accident'].value_counts(dropna=False)


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.barplot(x='accident', y='price', data=df_train, errorbar=None)
plt.title('Average Price by Accident History')
plt.xlabel('Accident')
plt.ylabel('Price')
plt.show()


df_train['clean_title'].value_counts(dropna=False)


# Binary kategorik
df_train['accident'].fillna('None reported', inplace=True)
df_train['clean_title'].fillna('Yes', inplace=True)

df_test['accident'].fillna('None reported', inplace=True) 
df_test['clean_title'].fillna('Yes', inplace=True)


print(df_train.isnull().sum())
print(df_test.isnull().sum())


print(train_data['fuel_type'].value_counts(dropna=False))
print("\n")
print(test_data['fuel_type'].value_counts(dropna=False))



df_train['fuel_type'] = df_train['fuel_type'].replace(['–', 'not supported'], np.nan)
df_test['fuel_type'] = df_test['fuel_type'].replace(['–', 'not supported'], np.nan)



print(df_train['fuel_type'].value_counts(dropna=False))
print("\n")
print(df_test['fuel_type'].value_counts(dropna=False))



count_dash = (df_train['engine'] == '–').sum()
print(f"Engine sütununda '–' olan satır sayısı: {count_dash}")


df_fuel_missing = df_train[df_train['fuel_type'].isna()|df_train['fuel_type'].isnull()].copy()


df_fuel_missing.info()


# 'engine' sütununda 'electric' ifadesi geçenlerin sayısı (büyük-küçük harf duyarlılığını kaldırmak için .str.lower() kullanalım)
electric_count = df_fuel_missing['engine'].str.lower().str.contains('electric fuel ').sum()

print(f"'fuel_type' null olan ve 'engine' sütununda 'electric' geçen satır sayısı: {electric_count}")


# 1. fuel_type eksik ve engine'de electric geçen satırlar
df_electric_missing = df_train[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('electric fuel'))
]

# 2. Marka dağılımı
brand_counts = df_electric_missing['brand'].value_counts()
    
print("Elektrikli motorlu araçların marka dağılımı (fuel_type eksik olanlar içinde):")
print(brand_counts)



df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('electric fuel')), 
    'fuel_type'
] = 'Electric'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('electric fuel')), 
    'fuel_type'
] = 'Electric'



print(df_train['fuel_type'].value_counts(dropna=False))
print("\n")
print(df_test['fuel_type'].value_counts(dropna=False))



df_non_electric_missing = df_train[
    (df_train['fuel_type'].isnull()) & 
    (~df_train['engine'].str.lower().str.contains('electric fuel'))
]

print(f"Electric olmayan satır sayısı: {len(df_non_electric_missing)}")




# fuel_type null olan satırlar

df_fuel_missing = df_train[df_train['fuel_type'].isna()].copy()
t_df_fuel_missing = df_test[df_test['fuel_type'].isna()].copy()
# Yakıt türleri listesi
fuel_keywords = ['gasoline', 'diesel', 'hybrid', 'electric fuel', 'flex', 'plug-in']

for fuel in fuel_keywords:
    count = df_fuel_missing['engine'].str.lower().str.contains(fuel).sum()
    print(f"'fuel_type' null olan ve 'engine' sütununda '{fuel}' geçen satır sayısı: {count}")
print("\n")

for fuel in fuel_keywords:
    count = t_df_fuel_missing['engine'].str.lower().str.contains(fuel).sum()
    print(f"'fuel_type' null olan ve 'engine' sütununda '{fuel}' geçen satır sayısı: {count}")



df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('gasoline')), 
    'fuel_type'
] = 'Gasoline'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('gasoline')), 
    'fuel_type'
] = 'Gasoline'


df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('diesel')), 
    'fuel_type'
] = 'Diesel'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('diesel')), 
    'fuel_type'
] = 'Diesel'


df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('hybrid')), 
    'fuel_type'
] = 'Hybrid'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('hybrid')), 
    'fuel_type'
] = 'Hybrid'


df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('flex')), 
    'fuel_type'
] = 'E85 Flex Fuel'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('flex')), 
    'fuel_type'
] = 'E85 Flex Fuel'


df_train.loc[
    (df_train['fuel_type'].isnull()) & 
    (df_train['engine'].str.lower().str.contains('plug-in')), 
    'fuel_type'
] = 'Plug-In Hybrid'

df_test.loc[
    (df_test['fuel_type'].isnull()) & 
    (df_test['engine'].str.lower().str.contains('plug-in')), 
    'fuel_type'
] = 'Plug-In Hybrid'



print(df_train['fuel_type'].value_counts(dropna=False))
print("\n")
print(df_test['fuel_type'].value_counts(dropna=False))



df_train[df_train['fuel_type']=='Electric']


count_dash = (df_train['engine'] == '–').sum()
print(f"Engine sütununda '–' olan satır sayısı: {count_dash}")

count_dash2 = ((df_train['engine'] == '–') & (df_train['fuel_type'].isna())).sum()

print(f"Engine sütununda '–' olan ve fuel_type null olan satır sayısı: {count_dash2}")


df_train.nunique()


import numpy as np

group_features_list = [
    ['brand', 'model', 'model_year', 'transmission'],
    ['brand', 'model', 'model_year'],
    ['brand', 'model'],
    ['brand'],
]

# engine içindeki dash ve benzeri ifadeleri NaN yap (tür değiştirme olmadan)
for df in [df_train, df_test]:
    df['engine'] = df['engine'].replace(['', '-', '–', '—', 'None', 'none', 'nan', 'NaN'], np.nan)

def fill_by_modes(df, col):
    for group_features in group_features_list:
        mode_map = (
            df[df[col].notna()]
            .groupby(group_features)[col]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)
            .reset_index()
        )
        mode_dict = {tuple(row[group_features]): row[col] for _, row in mode_map.iterrows()}
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask].apply(lambda r: mode_dict.get(tuple(r[group_features]), np.nan), axis=1)
    df[col].fillna(df[col].mode().iat[0], inplace=True)

fill_by_modes(df_train, 'engine')
fill_by_modes(df_test, 'engine')

fill_by_modes(df_train, 'fuel_type')
fill_by_modes(df_test, 'fuel_type')

print("df_train eksik engine:", df_train['engine'].isna().sum())
print("df_train eksik fuel_type:", df_train['fuel_type'].isna().sum())
print("df_test eksik engine:", df_test['engine'].isna().sum())
print("df_test eksik fuel_type:", df_test['fuel_type'].isna().sum())



print(df_train['fuel_type'].isnull().sum())  # df_train üzerinde kontrol et
print(df_test['fuel_type'].isnull().sum())   # df_test üzerinde kontrol et



print(df_train.isnull().sum())
print("\n")
print(df_test.isnull().sum())


df_train.info()


df_test.info()


import pandas as pd
import numpy as np

def encode_all_categoricals(df_train, df_test, onehot_thresh=10, tiny_offset=1e-6):
    """
    Train ve test üzerinde tutarlı şekilde encoding uygular.
    
    - onehot_thresh: bir kategorik sütuna one-hot uygulanacak maksimum benzersiz kategori sayısı.
    - tiny_offset: aynı frekanslı kategorilerin ayırt edilmesi için eklenecek küçük offset.
    
    Returns:
      df_train_enc, df_test_enc, encoders
      - encoders: her sütun için kullanılan method ve mapping bilgisi
    """
    # Kopyalar
    df_train_enc = df_train.copy()
    df_test_enc = df_test.copy()
    
    # Hangi sütunlar kategorik (id/target hariç)
    exclude = set(['id', 'price'])  # gerekirse ekleyebilirsin
    cat_cols = [c for c in df_train_enc.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    
    encoders = {}  # metadata: {col: {'method': 'onehot'/'freq', ...}}
    
    for col in cat_cols:
        # unique değer sayısı (train bazlı)
        nunique = df_train_enc[col].nunique(dropna=True)
        
        if nunique == 0:
            # tümü NaN ise, test'te de NaN kalacak; replace ile 'missing' koyabilirsin
            encoders[col] = {'method': 'all_nan'}
            continue
        
        if nunique <= onehot_thresh:
            # ----- One-Hot Encoding -----
            encoders[col] = {'method': 'onehot'}
            
            # get_dummies ile train one-hot
            dummies_train = pd.get_dummies(df_train_enc[col], prefix=col, dummy_na=False)
            df_train_enc = pd.concat([df_train_enc.drop(columns=[col]), dummies_train], axis=1)
            
            # test için aynı sütunları oluştur ve hizala
            dummies_test = pd.get_dummies(df_test_enc[col], prefix=col, dummy_na=False)
            # Align columns: eksikleri 0 ile doldur
            for c in dummies_train.columns:
                if c not in dummies_test.columns:
                    dummies_test[c] = 0
            # test'te ekstra sütunlar varsa kaldır
            dummies_test = dummies_test[dummies_train.columns]
            df_test_enc = pd.concat([df_test_enc.drop(columns=[col]), dummies_test], axis=1)
        
        else:
            # ----- Safe Frequency Encoding -----
            # Train frekansları (NaN hariç)
            freq = df_train_enc[col].value_counts(dropna=True)
            unique_cats = sorted(freq.index.astype(str))  # string yap ve sırala tutarlı offset için
            
            # tiny offset üret (alfabetik sıraya dayalı)
            offset_map = {cat: i * tiny_offset for i, cat in enumerate(unique_cats)}
            
            # freq + offset
            freq_with_offset = {cat: freq.loc[cat] + offset_map[cat] for cat in unique_cats}
            
            # Map'leri sakla (train mapping)
            encoders[col] = {
                'method': 'freq_with_offset',
                'freq_map': freq_with_offset,
                'min_freq': min(freq_with_offset.values())
            }
            
            # Train: map et (NaN bırak)
            df_train_enc[col + '_freq_enc'] = df_train_enc[col].astype(str).map(lambda x: freq_with_offset.get(x, np.nan))
            # Test: map et, görülmeyenlere fallback (min_freq * 0.1)
            fallback = encoders[col]['min_freq'] * 0.1 if encoders[col]['min_freq'] > 0 else 0.0
            df_test_enc[col + '_freq_enc'] = df_test_enc[col].astype(str).map(lambda x: freq_with_offset.get(x, fallback))
            
            # Orijinali bırakma veya silme tercihi: burada orijinalini siliyoruz
            df_train_enc.drop(columns=[col], inplace=True)
            df_test_enc.drop(columns=[col], inplace=True)
    
    return df_train_enc, df_test_enc, encoders

# Kullanım örneği:
df_train_enc, df_test_enc, encoders = encode_all_categoricals(df_train, df_test, onehot_thresh=10)



df_train_enc.info()


df_train_enc.head()


# df_train_enc içindeki bool sütunları sayıya çevir
bool_cols = df_train_enc.select_dtypes(include='bool').columns

for col in bool_cols:
    df_train_enc[col] = df_train_enc[col].astype(int)
    df_test_enc[col] = df_test_enc[col].astype(int)


df_train_enc.info()


from sklearn.preprocessing import StandardScaler

# Sayısal sütunlar: price hariç
numeric_cols = [col for col in df_train_enc.select_dtypes(include=['int64', 'float64']).columns if col != 'price']

# Scaler oluştur
scaler = StandardScaler()

# Train ve test üzerinde sadece seçilen sayısal sütunları ölçekle
df_train_enc[numeric_cols] = scaler.fit_transform(df_train_enc[numeric_cols])
df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

print("✅ Normalizasyon tamamlandı.")



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW


X_train = df_train_enc.drop(columns=['price', 'id'])
y_train = df_train_enc['price'].astype('float32')   # burada tipi float32 yapıyoruz

X_test = df_test_enc.drop(columns=['id'])



from tensorflow.keras import backend as K
# RMSE metriği tanımı
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU

model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.1),

    Dense(256),  # Yeni eklenen katman
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.1),

    Dense(128),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.15),

    Dense(128),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.15),

    Dense(64),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.15),

    Dense(32),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.1),

    Dense(1, activation='linear')
])

# Modeli derle
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])


# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)


# Eğitme
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=300,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# Eğitim sonrası performans
train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
print(f"Train RMSE: {train_rmse:.2f}")

# Test için tahmin
predictions = model.predict(X_test)


t = pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
t.shape


predictions = predictions.flatten()


import pandas as pd

# Örnek: predictions numpy array veya pandas Series olabilir
# predictions = model.predict(X_test).flatten()

submission = pd.DataFrame({
    'id': df_test['id'],
    'price': predictions
})

submission.to_csv('submission.csv', index=False)


