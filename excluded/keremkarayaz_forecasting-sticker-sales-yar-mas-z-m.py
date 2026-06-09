from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.feature_selection import RFE


df_train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_train.head()


country_sales = df_train.groupby('country')['num_sold'].sum().reset_index()

# Grafik: Country bazında satışlar
plt.figure(figsize=(8, 6))
sns.barplot(data=country_sales, x='country', y='num_sold', palette='viridis')
plt.title('Country Bazında Toplam Satışlar', fontsize=14)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Num Sold', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


df_grouped = df_train[['country', 'num_sold']].groupby('country').sum().reset_index()

# Veriyi satışa göre sıralama
df_grouped = df_grouped.sort_values('num_sold', ascending=False)

# Grafik boyutunu ayarlama
plt.figure(figsize=(10,6))

# Bar grafiğini çizme
plt.bar(df_grouped['country'], df_grouped['num_sold'], color='skyblue')

# Eksenleri ve başlığı ekleme
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.title('Total Sales by Country')

# X eksenindeki etiketleri döndürme
plt.xticks(rotation=45)

# Grafiği gösterme
plt.tight_layout()
plt.show()


df_train.isnull().sum()


df_test.isnull().sum()


df_no_kenya = df_train[df_train['country'] != 'Kenya']
sales_avg_per_country = df_no_kenya.groupby('country')['num_sold'].mean()
for country in sales_avg_per_country.index:
    df_train.loc[(df_train['country'] == country) & (df_train['num_sold'].isna()), 'num_sold'] = sales_avg_per_country[country]



df_train['date'] = pd.to_datetime(df_train['date'])

df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month

# Ayları mevsimlere gruplayan bir fonksiyon
def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'  # İlkbahar
    elif month in [6, 7, 8]:
        return 'summer'  # Yaz
    elif month in [9, 10, 11]:
        return 'autumn'  # Sonbahar
    elif month in [12, 1, 2]:
        return 'winter'  # Kış

# Yeni 'season' sütununu ekleyelim
#df_train['season'] = df_train['month'].apply(get_season)



df_train['sin_month'] = np.sin(2 * np.pi * df_train['month'] / 12)
#df_train['cos_month'] = np.cos(2 * np.pi * df_train['month'] / 12)


df_train['day'] = df_train['date'].dt.day

df_train['week_of_year'] = df_train['day'] // 7
df_train['dayofweek'] = df_train['date'].dt.dayofweek  # Haftanın günü (0 = Pazartesi)
df_train['is_weekend'] = df_train['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)



df_train.head()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in ['product', 'country', 'store']:
    if column in df_train.columns:
        df_train[f'{column}_encoded'] = label_encoder.fit_transform(df_train[column])

df_train=df_train.drop(["store","country","product","date"],axis=1)

df_train.head()


df_test.head()


df_train.info()


df_train = df_train[df_train['num_sold'] < 5000]


#x=df_train.drop(["num_sold"],axis=1)
#y=df_train["num_sold"]

y = df_train["num_sold"]
x = df_train.drop(columns=["num_sold"])


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)


#model = RandomForestRegressor()
#rfe = RFE(model, n_features_to_select=1)  # En iyi 10 özelliği seçmek için
#rfe.fit(x_train, y_train)
#X_rfe = rfe.fit_transform(x_train, y_train)


#feature_ranking = pd.DataFrame({
#    'Feature': x_train.columns,
#    'RFE_Rank': rfe.ranking_
#})


#feature_ranking = feature_ranking.sort_values(by='RFE_Rank', ascending=True)


#plt.figure(figsize=(12, 8))
#sns.barplot(x='RFE_Rank', y='Feature', data=feature_ranking, palette='viridis')
#plt.title('Özelliklerin RFE Sıralaması')
#plt.xlabel('RFE Katkı Sırası (1 = En Önemli)')
#plt.ylabel('Özellikler')
#plt.show()


#selected_features = x.columns[rfe.support_]
#print("Seçilen özellikler:", selected_features)


#X_train_rfe = x_train[selected_features]
#X_test_rfe = x_test[selected_features]


#selected_features_rfe = [f for f, support in zip(X_train_rfe.columns, rfe.support_) if support]
#print("Seçilen Özellikler:", selected_features_rfe)


#model1=rfe.fit(X_train_rfe, y_train)


#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
#y_pred = model1.predict(X_test_rfe)
#print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))


#selected_features_rfe = [f for f, support in zip(X_train_rfe.columns, rfe.support_) if support]
#print("Seçilen Özellikler:", selected_features_rfe)


import matplotlib.pyplot as plt
import seaborn as sns

# RFE'nin verdiği sıralamaya göre verilerin görselleştirilmesi
#importance = rfe.ranking_  # Özelliklerin sıralanması
#features = X_train_rfe.columns

# Özellikleri ve sıralamaları bir DataFrame'e dönüştürüp sıralıyoruz
#feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
#feature_importance = feature_importance.sort_values(by='Importance')

# Özelliklerin görselleştirilmesi
#plt.figure(figsize=(12, 8))
#sns.barplot(x='Importance', y='Feature', data=feature_importance)
#plt.title('RFE Özellik Önem Dereceleri')
#plt.show()


#y_test_pred = model.predict(X_test_rfe)


#X_train_rfe.head()


model.score(x_test,y_test)


df_train.info()


#X_train_rfe = x_train[selected_features]
#X_test_rfe = x_test[selected_features]


df_test['date'] = pd.to_datetime(df_test['date'])

df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month

# Ayları mevsimlere gruplayan bir fonksiyon
def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'  # İlkbahar
    elif month in [6, 7, 8]:
        return 'summer'  # Yaz
    elif month in [9, 10, 11]:
        return 'autumn'  # Sonbahar
    elif month in [12, 1, 2]:
        return 'winter'  # Kış

# Yeni 'season' sütununu ekleyelim
#df_test['season'] = df_test['month'].apply(get_season)

df_test['sin_month'] = np.sin(2 * np.pi * df_test['month'] / 12)
#df_test['cos_month'] = np.cos(2 * np.pi * df_test['month'] / 12)

df_test['day'] = df_test['date'].dt.day

df_test['week_of_year'] = df_test['day'] // 7 ####

df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['is_weekend'] = df_test['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

df_test=df_test.drop(["date"],axis=1)


for column in ['product', 'country', 'store']:
    if column in df_test.columns:
        df_test[f'{column}_encoded'] = label_encoder.fit_transform(df_test[column])

df_test=df_test.drop(["store","country","product"],axis=1)

df_test.head()


predictions=model.predict(df_test)


y_hata=pd.DataFrame()
y_hata["Tahmin"]=predictions
y_hata["y"]=y
y_hata = y_hata[y_hata['y'].notna()]
y_hata.head()


y_hata["error"]=y_hata["y"]-y_hata["Tahmin"]
y_hata.head()


plt.scatter(y_hata["Tahmin"], y_hata["error"])
plt.xlabel("Tahmin")
plt.ylabel("Hata")
plt.title("Tahmin vs. Hata")
plt.axhline(0, color='red', linestyle='--')
plt.show()


predictions_log=np.log(y_hata["Tahmin"])
y_test_log=np.log(y_hata["y"])


y_hata[y_hata["error"] > 1500]


#mean_absolute_percentage_error(y_test_log,predictions_log)


mean_absolute_percentage_error(y_hata["y"],y_hata["Tahmin"])


from scipy.stats import zscore
z_scores = zscore(df_train['num_sold'])  # 'num_sold' yerine uç değerleri kontrol etmek istediğiniz kolonu yazın
abs_z_scores = np.abs(z_scores)
outliers = np.where(abs_z_scores > 3)  # Z-skoru 3'ten büyük olan verileri alır

Q1 = df_train['num_sold'].quantile(0.25)
Q3 = df_train['num_sold'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Uç değerler
outliers = df_train[(df_train['num_sold'] < lower_bound) | (df_train['num_sold'] > upper_bound)]

sns.boxplot(x=df_train['num_sold'])


submission = pd.DataFrame({
    'id': df_test['id'],  # Test setindeki ID'ler
    'num_sold': predictions  # Tahmin edilen fiyatlar
})


submission.to_csv('submission_for_kaggle3.csv', index=False)


df_test.head()


df_train.head()

