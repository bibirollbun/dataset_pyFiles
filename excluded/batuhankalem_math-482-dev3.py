import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Veri setlerini pandas DataFrame olarak okuyalım
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")
train_df.head()


# 'id' sütunlarını daha sonra gönderim dosyası için ayıralım
train_ids = train_df['id']
test_ids = test_df['id']

# Modelde kullanmayacağımız 'id' sütununu kaldıralım
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Kategorik ve boolean sütunları seçelim
categorical_cols = train_df.select_dtypes(include=['object', 'bool']).columns

# pd.get_dummies() ile bu sütunları otomatik olarak 0'lara ve 1'lere çevirelim
train_processed = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_processed = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Sütunları hizalamak için train ve test setlerini hizalayalım
# Bu, her iki veri setinin de tam olarak aynı sütunlara sahip olmasını sağlar
train_labels = train_processed['accident_risk']
X = train_processed.drop('accident_risk', axis=1)
X_test = test_processed.reindex(columns=X.columns, fill_value=0)

X.head()


# Modelimizi tanımlayalım
model = LinearRegression()

# Tüm eğitim verisini kullanarak modeli eğitelim
print("Lineer Regresyon modeli eğitiliyor...")
model.fit(X, train_labels)
print("Model eğitimi tamamlandı.")

# Not: Bu basit baseline'da bir validation seti ayırmadık.
# Kendi çalışmanızda, model performansını ölçmek için train_test_split kullanmanız tavsiye edilir.


print("Test seti üzerinde tahminler yapılıyor...")
predictions = model.predict(X_test)

# Yarışma kurallarına göre tahminlerin 0 ile 1 arasında olması gerekiyor.
# np.clip() fonksiyonu ile değerleri bu aralığa sıkıştıralım.
predictions = np.clip(predictions, 0, 1)

# Gönderim dosyasını sample_submission formatına uygun olarak oluşturalım
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': predictions})
submission_df.to_csv('KalemsRegression.csv', index=False)

print("KalemsRegression.csv dosyası başarıyla oluşturuldu!")
submission_df.head()

