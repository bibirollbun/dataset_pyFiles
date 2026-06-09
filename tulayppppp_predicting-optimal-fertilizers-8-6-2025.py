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


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_df


train_df.info()


train_df.describe()


train_df.describe().T


train_df.head(5)


train_df.tail(5)


features = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]

target = "target_column"  

for feature in features:
    plt.figure(figsize=(20, 12))
    
   
    plt.subplot(3, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram + KDE of {feature}")
    
  
    plt.subplot(3, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Boxplot of {feature}")
    
 
    plt.subplot(3, 2, 3)
    sns.violinplot(x=train_df[feature])
    plt.title(f"Violin Plot of {feature}")
    
  
    if target in train_df.columns:
        plt.subplot(3, 2, 4)
        sns.scatterplot(x=train_df[feature], y=train_df[target])
        plt.title(f"Scatter Plot of {feature} vs {target}")
    
    
    if target in train_df.columns and train_df[target].dtype == 'object':
        plt.subplot(3, 2, 5)
        sns.swarmplot(x=train_df[target], y=train_df[feature])
        plt.title(f"Swarm Plot of {feature} by {target}")
    
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

numerical_features = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

for feature in ["Soil Type", "Crop Type"]:
    counts = train_df[feature].value_counts()
    
  
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=feature, order=counts.index)
    plt.title(f"{feature} - Countplot")
    plt.xticks(rotation=45)
    plt.show()
    
  
    fig = px.treemap(
        names=counts.index,
        parents=[""] * len(counts),
        values=counts.values,
        title=f"{feature} - Treemap",
        color=counts.values,
        color_continuous_scale='Viridis'
    )
    fig.show()
    
  
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        textprops={'fontsize': 10}
    )
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(f"{feature} - Donut Chart")
    plt.axis('equal')
    plt.show()
    
   
    pivot_table = train_df.pivot_table(
        index=feature,
        values=numerical_features,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"{feature} bazında sayısal özelliklerin ortalaması (Heatmap)")
    plt.ylabel(feature)
    plt.show()
    
  
    print(f"{feature} için benzersiz değer sayısı: {train_df[feature].nunique()}")
    print(f"{feature} sütunundaki eksik değer sayısı: {train_df[feature].isnull().sum()}")
    print("-" * 50)



train_df.isna().sum()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_df


test_df.isna().sum()


test_df.shape


train_df.columns = train_df.columns.map(lambda x: x.strip())
test_df.columns = test_df.columns.map(lambda x: x.strip())


missing_in_test = [col for col in train_df.columns 
                   if col not in test_df.columns and col != 'Fertilizer Name']


missing_in_train = [col for col in test_df.columns 
                    if col not in train_df.columns and col != 'id']

if missing_in_test or missing_in_train:
    print(f"Test veri setinde eksik sütunlar: {missing_in_test}")
    print(f"Eğitim veri setinde eksik sütunlar: {missing_in_train}")
    
   
    for col in missing_in_test:
        test_df.loc[:, col] = 0
    
   
    for col in missing_in_train:
        if col != 'id':
            train_df.loc[:, col] = 0



print(train_df.columns)
print(test_df.columns)


print("Eğitim sütunları:", train_df.columns.tolist())
print("Test sütunları:", test_df.columns.tolist())

missing_in_test = [col for col in train_df.columns if col not in test_df.columns and col != 'Fertilizer Name']
missing_in_train = [col for col in test_df.columns if col not in train_df.columns and col != 'id']

print("Eksik sütunlar testte:", missing_in_test)
print("Eksik sütunlar eğitimde:", missing_in_train)


integer = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
string = ['Soil Type', 'Crop Type']
target_name = 'Fertilizer Name'


base_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

label_encode_columns = []
for feature in base_features:
    label_encode_columns.append(f"{feature}_quantile")
    label_encode_columns.append(f"{feature}_equal")


from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(train_df, test_df, columns):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}

    for col in columns:
        if col in train_df.columns:
            le = LabelEncoder()
            train_encoded[col] = le.fit_transform(train_df[col].astype(str))
            encoders[col] = le
            if col in test_df.columns:
                
                test_encoded[col] = test_df[col].map(lambda x: le.transform([str(x)])[0] if x in le.classes_ else -1)

    return train_encoded, test_encoded, encoders


train_data_encoded, test_data_encoded, label_encoders = encode_categorical_columns(
    train_df, test_df, string
)


print("Kodlama öncesi eğitim verisi (ilk 5 satır):")
print(train_df[string].head())

print("\nKodlama öncesi test verisi (ilk 5 satır):")
print(test_df[string].head())


train_data_encoded, test_data_encoded, label_encoders = encode_categorical_columns(
    train_df, test_df, string
)


print("\nKodlama sonrası eğitim verisi (ilk 5 satır):")
print(train_data_encoded[string].head())


print("\nKodlama sonrası test verisi (ilk 5 satır):")
print(test_data_encoded[string].head())


label_encode_cols = [
    'Temparature_quantile', 'Temparature_equal',
    'Humidity_quantile', 'Humidity_equal',
    'Moisture_quantile', 'Moisture_equal',
    'Nitrogen_quantile', 'Nitrogen_equal',
    'Potassium_quantile', 'Potassium_equal',
    'Phosphorous_quantile', 'Phosphorous_equal'
]



print("Veri setinde mevcut sütunlar:")
print(train_data_encoded.columns.tolist())


from sklearn.preprocessing import LabelEncoder

def encode_remaining_objects(train_df, test_df, exclude_cols=None):
    """
    Eğitim veri setinde halen 'object' tipinde olan sütunları label encode eder.
    Hedef sütunlar exclude_cols listesinde belirtilir ve kodlanmaz.

    Args:
        train_df (pd.DataFrame): Eğitim veri seti
        test_df (pd.DataFrame): Test veri seti
        exclude_cols (list): Kodlanmayacak sütun isimleri

    Returns:
        train_encoded, test_encoded
    """
    if exclude_cols is None:
        exclude_cols = []

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    object_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
    cols_to_encode = [col for col in object_cols if col not in exclude_cols]

    if cols_to_encode:
        print(f"Uyarı: Bu sütunlar hala kodlama gerektiriyor: {cols_to_encode}")
        for col in cols_to_encode:
            le = LabelEncoder()
            train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))
            if col in test_encoded.columns:
                test_encoded[col] = test_encoded[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return train_encoded, test_encoded


train_data_encoded, test_data_encoded = encode_remaining_objects(
    train_data_encoded, test_data_encoded, exclude_cols=['Fertilizer Name']
)


print("Kodlanan sütunlar:", train_data_encoded.select_dtypes(include=['int64', 'int32']).columns.tolist())


from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler nesnesi oluştur
scaler = MinMaxScaler()

# Ölçeklendirilmek istenen sayısal sütunlar
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Eğitim veri setinde sayısal sütunları ölçeklendir
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])

# Test veri setinde aynı scaler ile dönüşüm uygula
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Örnek veri
train_data = pd.DataFrame({
    'Temparature': [20, 25, 30],
    'Humidity': [30, 45, 50],
})

test_data = pd.DataFrame({
    'Temparature': [22, 28],
    'Humidity': [35, 48],
})

numeric_cols = ['Temparature', 'Humidity']
scaler = MinMaxScaler()

train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

print(train_data)
print(test_data)



print(train_data.columns.tolist())


numeric_cols = ['Temparature', 'Humidity']


numeric_cols = [col for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'] if col in train_data.columns]

X_train = train_df[numeric_cols]
y_train = train_df['Fertilizer Name']
X_train
y_train


train_data.head()


print(train_data.shape)


print(dir())  # Tanımlı değişkenleri listeler


from sklearn.preprocessing import MinMaxScaler

numeric_cols = [col for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'] if col in train_data.columns]

scaler = MinMaxScaler()

train_data_scaled = train_data.copy()
train_data_scaled[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])

test_data_scaled = test_data.copy()
test_data_scaled[numeric_cols] = scaler.transform(test_data[numeric_cols])



missing_cols = [col for col in train_data_scaled.columns if col not in test_data_scaled.columns]


print(train_data_scaled.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Eğitim ve doğrulama seti oluştur (örnek %20 doğrulama)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2. Model oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)

# 3. Doğrulama seti üzerinde tahmin yap
y_val_pred = model.predict(X_val)

# 4. Performans ölçümü
print("Doğrulama Seti Doğruluk:", accuracy_score(y_val, y_val_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_val, y_val_pred))





print("train_df satır sayısı:", train_df.shape[0])
print("X_train satır sayısı:", X_train.shape[0])
print("y_train satır sayısı:", y_train.shape[0])


target_column = 'Fertilizer Name'
numeric_cols = [col for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'] if col in train_df.columns]

# Özellikler ve hedef
X_train = train_df[numeric_cols]
y_train = train_df[target_column]

print(X_train.shape, y_train.shape)  # Satır sayıları eşit olmalı



train_df.head()


test_df.columns.tolist()


train_df.columns.tolist()


numeric_cols = [col for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'] if col in test_data_scaled.columns]


import pandas as pd

# Test verisini yükle
test_original = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_original
# Gerekli feature'lar (özellikler) ile x_test'i oluştur
x_test = test_original[['Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
x_test


print(test_data_scaled.columns.tolist())


expected_columns = X_train.columns.tolist()


missing_in_test = [col for col in expected_columns if col not in test_data.columns]


for col in missing_in_test:
    test_data[col] = 0 # Sayısal veriler için, X


extra_in_test = [col for col in test_data.columns if col not in expected_columns]
if extra_in_test:
    test_data = test_data.drop(columns=extra_in_test)


test_data = test_data[expected_columns]


test_predictions = model.predict(test_data_scaled)
print("Tahminler:", test_predictions)


test_predictions = model.predict(test_data_scaled)
print("Tahminler:", test_predictions)


print(X_train.columns)


print(test_data_scaled.columns)


test_data_scaled = test_data_scaled[train_data_scaled.columns]


test_predictions = model.predict(test_data_scaled)
print("Tahminler:", test_predictions)


# Eğer hedef LabelEncoder ile kodlandıysa
if 'Fertilizer Name' in label_encoders:
    y_encoder = label_encoders['Fertilizer Name']
    test_predictions_labels = y_encoder.inverse_transform(test_predictions)
else:
    test_predictions_labels = test_predictions


if 'Humidity' in test_data_scaled.columns:
    submission_df = test_data_scaled[['Humidity']].copy()
else:
    print("HATA: Test verisinde 'id' sütunu bulunamadı!")
    print("Mevcut sütunlar:", test_data_scaled.columns.tolist())
submission_df['Fertilizer'] = test_predictions_labels


submission_df.to_csv('submission.csv', index=False)
print("Tahminler 'submission.csv' dosyasına kaydedildi.")


import joblib
joblib.dump(model, 'trained_model.pkl')
print("Model 'trained_model.pkl' olarak kaydedildi.")


model_loaded = joblib.load('trained_model.pkl')
model_loaded


# 1. Sample submission formatını kontrol edin
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
print("Sample submission formatı:")
print(sample_submission.head())
print("Sütun adları:", sample_submission.columns.tolist())


# 2. ID sütununu kontrol edin
print("\nTest verisindeki sütunlar:")
print(test_data_scaled.columns.tolist())




# 3. Submission dosyasını doğru formatta oluşturun
if 'id' in test_data_scaled.columns:
    submission_df = pd.DataFrame({
        'id': test_data_scaled['id'],
        'Fertilizer Name': test_predictions_labels  # Boşluk yok, sample_submission'daki gibi
    })
else:
    # Eğer ID yoksa, test verisinden alın
    test_original = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    submission_df = pd.DataFrame({
        'id': test_original['id'],
        'Fertilizer Name': test_predictions_labels
    })

# 4. Submission dosyasını kontrol edin
print("\nSubmission dosyası:")
print("Boyut:", submission_df.shape)
print("Sütunlar:", submission_df.columns.tolist())
print("İlk 5 satır:")
print(submission_df.head(10))

# 5. Sample submission ile karşılaştırın
print("\n=== FORMAT KONTROLÜ ===")
print("Sample submission boyutu:", sample_submission.shape)
print("Sizin submission boyutu:", submission_df.shape)
print("Sütun adları uyumlu mu?", list(sample_submission.columns) == list(submission_df.columns))

# 6. Fertilizer değerlerini kontrol edin
print("\nFertilizer değerleri:")
print("Unique tahminler:", submission_df['Fertilizer'].unique())
print("Tahmin sayısı:", submission_df['Fertilizer'].value_counts())

# 7. Dosyayı kaydedin
submission_df.to_csv('submission.csv', index=False)
print("\n✅ Submission dosyası başarıyla kaydedildi!")

# 8. Son kontrol
final_check = pd.read_csv('submission.csv')
print("\nKaydedilen dosyanın son kontrolü:")
print(final_check.head())



# --- Veri Yükleme ve Ön İşleme Adımlarınız (Tüm test verisinin işlendiğinden emin olun) ---
# Orijinal test verisini yükleyin
test_original = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Eğitim aşamasından kaydettiğiniz X_train.columns'ı kullandığınızdan emin olun.
# Bu, özelliklerin eşleştirilmesi için çok önemlidir.
# Eğer X_train.columns'a doğrudan erişemiyorsanız, 6 özelliğin adını bildiğinizden emin olun.
# Gösterim için, X_train_cols = ['FeatureA', 'FeatureB', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium'] varsayalım.
# Bunları, modelinizi eğittikten sonraki gerçek X_train.columns'tan almalısınız.
X_train_cols = model.feature_names_in_ # Bu, sklearn'ın yeni sürümlerinde model.fit() sonrası kullanılabilir.


# Tahmin için test verisini hazırlayın
# Modelin beklediği sadece özellikler seçilir (X_train.columns'tan).
# Ve önceki hatalarda bahsedildiği gibi eksik sütunlar ele alınır.
test_data = test_original[X_train_cols] # Yalnızca X_train'deki özellikleri seçer.

# Eğer test_data'da X_train_cols'a göre eksik sütunlar varsa
# (örneğin, onları attıysanız veya başlangıçta raw_test_data'da yoktularsa)
# Sayı doğru olsa bile bu kısım hala çok önemlidir.
# Ölçeklemeden önce modelin beklediği tüm sütunların `test_data` içinde bulunduğundan emin olun.
missing_cols_in_test_data = [col for col in X_train_cols if col not in test_data.columns]
for col in missing_cols_in_test_data:
    # 0 ile doldurun veya daha uygun olarak, o sütun için X_train'deki ortalama/medyan ile doldurun.
    # Sağlam bir çözüm için, X_train ortalamalarını/medyanlarını kaydetmeniz gerekir.
    test_data[col] = 0 # Veya X_train[col].mean() (eğer kaydettiyseniz)

# Ölçeklemeden önce sütunların doğru sırada olduğundan emin olun
test_data = test_data[X_train_cols]

# Test verisini, eğitim aşamasında *fit edilmiş* ölçekleyiciyi kullanarak ölçeklendirin.
# 'scaler' nesnenizin daha önce X_train_scaled üzerinde fit edildiğinden emin olun.
test_data_scaled = scaler.transform(test_data)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=X_train_cols)


# --- Tahmin Adımınız ---
# 'test_predictions' buradan geliyor.
# `predict` metoduna TÜM test_data_scaled'i ilettiğinizden emin olun.
test_predictions = model.predict(test_data_scaled) # Bu, şimdi 250.000 tahmin döndürmelidir.

# --- Sayısal tahminleri etiketlere eşleme (varsa) ---
# Eğer modeliniz sayısal etiketler (örneğin 0, 1, 2...) çıktı veriyorsa ve yarışma metin (string)
# (örneğin "Üre", "NPK" gibi) bekliyorsa, bunları geri eşleştirmeniz gerekir.
# Eğitim aşamanızda bir eşleme tanımlamış olmalısınız, örneğin:
# label_mapping = {0: 'Üre', 1: 'NPK', ...}
# Eğer modeliniz zaten metin çıktı veriyorsa, o zaman test_predictions_labels = test_predictions.

# Örnek eşleme (kendi gerçek eşlemenizle değiştirin)
# Modelinizin 0, 1, 2, ... gibi sayılar ürettiğini varsayalım.
# Ve eğitim için kullanılan gerçek gübre isimlerinin bir listesi olsun.
# Örnek: fertilizer_names = ['Üre', 'NPK', 'DAP', 'Potasyum', 'Jips', 'MOP']
# Ardından tahmin edilen sayıları isimlere geri eşleştirmek için bir sözlük oluşturun.
# Bu sözlük, hedef değişkeninizi (y_train) ilk yüklediğinizde/ön işlediğinizde oluşturulmalıdır.
# Örneğin: unique_fertilizer_names = y_train.unique()
#              label_to_name_map = {i: name for i, name in enumerate(unique_fertilizer_names)}
# test_predictions_labels = [label_to_name_map[pred] for pred in test_predictions]

# test_predictions'ın zaten doğru metin etiketlerini içerdiğini varsayarsak,
# veya burada çalışan bir eşlemeniz varsa:
test_predictions_labels = test_predictions # Eğer modeliniz doğrudan etiketleri çıktı veriyorsa, bunu kullanın.
# VEYA
# test_predictions_labels = [etiket_çözücü_fonksiyonunuz(tahmin) for tahmin in test_predictions]


# --- Gönderim Dosyası Oluşturma Adımınız ---
# Hatanın nedeni burasıydı, ancak düzeltme önceki adımlarda.
# `test_original['id']` 250.000 girişe sahip.
# `test_predictions_labels` bu noktada KESİNLİKLE 250.000 girişe sahip OLMALI.
submission_df = pd.DataFrame({
    'id': test_original['id'],
    'Fertilizer Name': test_predictions_labels
})

submission_df.to_csv('submission.csv', index=False)
print("Gönderim dosyası 'submission.csv' başarıyla oluşturuldu!")


import pandas as pd


submission_df.to_csv('submission.csv', index=False)


submission_df.to_csv('submission.csv', index=False)

print("Gönderim dosyası 'submission.csv' başarıyla oluşturuldu!")

