import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc


!unzip /kaggle/input/santander-product-recommendation/train_ver2.csv.zip
!unzip /kaggle/input/santander-product-recommendation/test_ver2.csv.zip


def bellek_azalt(df):
    """
    Manuel tip tanımlamak yerine, bu fonksiyon her kolona bakar 
    ve veriyi bozmadan en küçük veri tipine (int8, float32 vs.) çevirir.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Başlangıç Bellek Kullanımı: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Tamsayıları küçült (int64 -> int8/16/32)
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Ondalıkları küçült (float64 -> float32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 bazen sorun çıkarır, 32 güvenli
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            # Object (String) olanları 'category' yap (Çok büyük hız kazandırır)
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Bitiş Bellek Kullanımı: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% tasarruf)')
    return df


train_data = pd.read_csv('/kaggle/working/train_ver2.csv', nrows=500000)
test_data = pd.read_csv('/kaggle/working/test_ver2.csv') 


drop_cols = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov', 'conyuemp', 'fecha_dato']
train_data.drop(columns=drop_cols, errors='ignore', inplace=True)
test_data.drop(columns=drop_cols, errors='ignore', inplace=True)


train_data.head()


train_data.info()


test_data.head()


test_data.info()


numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(-1)


cat_cols = train_data.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    # Train ve Test'i birlikte kodluyoruz ki değerler tutsun
    combined = pd.concat([train_data[col], test_data[col]], axis=0).astype('category')
    train_data[col] = combined[:len(train_data)].cat.codes
    test_data[col] = combined[len(train_data):].cat.codes


train_data = bellek_azalt(train_data)


# Hedef değişkeni ayır (Son kolon hedef kabul edilir)
y = train_data.iloc[:, -1]
X = train_data.drop(train_data.columns[-1], axis=1)


# Train'de olup Test'te olmayan tüm kolonlar bizim hedef ürünlerimizdir.
target_cols = [col for col in train_data.columns if col not in test_data.columns]

print(f"Hedef ürün sayısı: {len(target_cols)}")


y = train_data.iloc[:, -1]
X = train_data.drop(columns=target_cols, axis=1)
test_ids = test_data['ncodpers']
X_test = test_data[X.columns]


gc.collect()


print(f"Eğitim (X) kolonları: {X.columns.tolist()}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier


# Hız için ağaç sayısını (n_estimators) düşük, derinliği (max_depth) sınırlı tuttum.
model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score


y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nModel Doğruluğu: {acc:.4f}")


beklenen_kolonlar = X.columns


for col in beklenen_kolonlar:
    # Eğer test setinde bu kolon varsa ve tipi 'object' (yazı) ise
    if col in X_test.columns and X_test[col].dtype == 'object':
        print(f"Onarılıyor: {col}")
        # ' NA' veya hatalı metinleri NaN yapar, sonra -1 ile doldurur
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)


print("Eksik değerler -1 ile dolduruluyor...")
X_test = X_test.fillna(-1)


if X_test.isnull().values.any():
    print("UYARI: Hala boş değerler var!")
else:
    print("Veri temiz, boş değer yok.")


final_preds = model.predict(X_test)


target_product_name = "ind_recibo_ult1" 
 
product_preds = []
for p in final_preds:
    if p == 1:
        product_preds.append(target_product_name)
    else: 
        product_preds.append("ind_cco_fin_ult1") 
 
submission = pd.DataFrame({
    'ncodpers': test_ids,
    'added_products': product_preds
})
 
submission.to_csv('submission.csv', index=False)
print("Düzeltilmiş submission.csv oluşturuldu.")


submission

