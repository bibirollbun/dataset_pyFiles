# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dosyaları yükleyelim
sales_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
sales_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")

# İlk 5 satırı görüntüleyerek veri setlerini keşfedelim
print("Sales Train Set:")
print(sales_train.head())

print("\nSales Test Set:")
print(sales_test.head())

print("\nInventory Set:")
print(inventory.head())

print("\nCalendar Set:")
print(calendar.head())

print("\nTest Weights:")
print(test_weights.head())

# Veri boyutlarını kontrol edelim
print(f"\nSales Train Shape: {sales_train.shape}")
print(f"Sales Test Shape: {sales_test.shape}")
print(f"Inventory Shape: {inventory.shape}")
print(f"Calendar Shape: {calendar.shape}")
print(f"Test Weights Shape: {test_weights.shape}")



# Tarih sütunlarını datetime formatına çevirin
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])
calendar['date'] = pd.to_datetime(calendar['date'])

# Eksik değer kontrolü
print("Eksik değerler (Sales Train):")
print(sales_train.isnull().sum())

# Satış verilerini ve tatil bilgilerini birleştirme
train_merged = sales_train.merge(calendar, on=['date', 'warehouse'], how='left')
test_merged = sales_test.merge(calendar, on=['date', 'warehouse'], how='left')

# Envanter bilgilerini satış verilerine ekleme
train_merged = train_merged.merge(inventory, on=['unique_id', 'warehouse'], how='left')
test_merged = test_merged.merge(inventory, on=['unique_id', 'warehouse'], how='left')

# Özellik mühendisliği: Birim başına gelir ve indirim yüzdesi
train_merged['revenue_per_unit'] = train_merged['sales'] / train_merged['total_orders']
train_merged['discount_total'] = train_merged[[col for col in train_merged.columns if 'discount' in col]].sum(axis=1)
test_merged['discount_total'] = test_merged[[col for col in test_merged.columns if 'discount' in col]].sum(axis=1)

# Tatil günlerini dummy değişkenlere çevirme
train_merged = pd.get_dummies(train_merged, columns=['holiday'])
test_merged = pd.get_dummies(test_merged, columns=['holiday'])

# Verinin ilk 5 satırını kontrol etme
print("Birleştirilmiş Eğitim Verisi:")
print(train_merged.head())

# Kaydedilmesi için CSV çıktısı
train_merged.to_csv("train_prepared.csv", index=False)
test_merged.to_csv("test_prepared.csv", index=False)


# Eksik değerlerin doldurulması (ortalama ile doldurma)
train_merged['total_orders'] = train_merged['total_orders'].fillna(train_merged['total_orders'].mean())
train_merged['sales'] = train_merged['sales'].fillna(train_merged['sales'].mean())

# Yeni özellikler türetme: Mevsim, haftanın günü ve hafta sonu bilgisi
train_merged['season'] = train_merged['date'].dt.month % 12 // 3 + 1
train_merged['day_of_week'] = train_merged['date'].dt.dayofweek
train_merged['is_weekend'] = train_merged['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Satış ve indirim yüzdesi arasındaki ilişki
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_merged, x='discount_total', y='sales')
plt.title("Satış ve İndirim Yüzdesi Arasındaki İlişki")
plt.xlabel("İndirim Toplamı")
plt.ylabel("Satış")
plt.show()

# Veriyi kaydetme
train_merged.to_csv("train_cleaned.csv", index=False)


import matplotlib.pyplot as plt

# Negatif indirim değerlerini temizleme
# Negatif indirim toplamlarını sıfır olarak değiştirme (eğer anlamlı değilse)
train_merged['discount_total'] = train_merged['discount_total'].apply(lambda x: x if x >= 0 else 0)

# İndirim yüzdesi özelliği oluşturma
train_merged['discount_percentage'] = train_merged['discount_total'] / train_merged['sell_price_main']

# Yeni grafiği oluşturma: İndirim Yüzdesi ve Satış İlişkisi
plt.figure(figsize=(10, 6))
plt.scatter(train_merged['discount_percentage'], train_merged['sales'], alpha=0.5)
plt.title("Satış ve İndirim Yüzdesi Arasındaki İlişki")
plt.xlabel("İndirim Yüzdesi")
plt.ylabel("Satış")
plt.show()


# Mantıksız indirim yüzdesi değerlerini temizleme
train_merged = train_merged[(train_merged['discount_percentage'] >= 0) & (train_merged['discount_percentage'] <= 1)]


correlation = train_merged[['discount_percentage', 'sales']].corr()
print(correlation)


category_sales = train_merged.groupby('L1_category_name_en')['sales'].mean()
print(category_sales)


import seaborn as sns
sns.boxplot(x=train_merged['discount_percentage'])


train_merged = train_merged[train_merged['discount_percentage'] <= 0.5]


category_corr = train_merged.groupby('L1_category_name_en')[['discount_percentage', 'sales']].corr().iloc[0::2, -1]
print(category_corr)


train_merged['discount_range'] = pd.cut(train_merged['discount_percentage'], 
                                        bins=[0, 0.05, 0.10, 1.0], 
                                        labels=['Low (0-5%)', 'Medium (5-10%)', 'High (10%+)'])

discount_sales = train_merged.groupby('discount_range')['sales'].mean()
print(discount_sales)


holiday_sales = train_merged.groupby('holiday_0')['sales'].mean()
print(holiday_sales)


warehouse_sales = train_merged.groupby('warehouse')['sales'].mean().sort_values(ascending=False)
print(warehouse_sales.head(5))


category_warehouse_sales = train_merged.groupby(['L1_category_name_en', 'warehouse'])['sales'].mean()
print(category_warehouse_sales)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Özellik ve hedef seçimi
X = train_merged[['discount_percentage', 'availability', 'holiday_0']]
y = train_merged['sales']

# Eğitim ve test seti ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Tahminler
predictions = model.predict(X_test)


# Gerçek değerler ve tahmin edilen değerleri bir DataFrame'de birleştirme
results = pd.DataFrame({
    'Gerçek Değerler': y_test.values,  # y_test bir pandas Series ise, values kullanarak NumPy dizisine dönüştürüyoruz
    'Tahmin Edilen Değerler': predictions
})

# Fark sütununu ekleme
results['Fark'] = results['Gerçek Değerler'] - results['Tahmin Edilen Değerler']

# Sonuçları listeleme
print(results)


results.to_csv('tahmin_sonuclari.csv', index=False, encoding='utf-8-sig')
print("Sonuçlar tahmin_sonuclari.csv dosyasına düzgün şekilde kaydedildi.")


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae}")


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
print(f"MSE: {mse}, RMSE: {rmse}")


import matplotlib.pyplot as plt
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Tahmin ve Gerçek Değerlerin Dağılımı")
plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Hiperparametre ayarları
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# GridSearch ile XGBoost
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# En iyi model ve tahmin
best_xgb = grid_search.best_estimator_
predictions = best_xgb.predict(X_test)

# Yeni metrikler
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"MAE: {mae}, RMSE: {rmse}")



submission = pd.DataFrame({
    'id': X_test.index,  # Test verisetindeki id sütunu veya index
    'sales': predictions  # Tahmin edilen sonuçlar
})
submission.to_csv("tahmin_sonuclari.csv", index=False)

