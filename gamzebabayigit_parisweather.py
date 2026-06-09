import numpy as np  
import pandas as pd  
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


train=pd.read_csv('/kaggle/input/x-hec-ts-2025-predicting-air-quality-in-paris/train.csv')
test=pd.read_csv('/kaggle/input/x-hec-ts-2025-predicting-air-quality-in-paris/test.csv')
sample_submission=pd.read_csv('/kaggle/input/x-hec-ts-2025-predicting-air-quality-in-paris/sample_submission.csv')


train.head()


test.head()


sample_submission.head()


train.info()


train.isnull().sum()


train['valeur_NO2']=train['valeur_NO2'].fillna(train['valeur_NO2'].mean())
train['valeur_CO']=train['valeur_CO'].fillna(train['valeur_CO'].mean())
train['valeur_O3']=train['valeur_O3'].fillna(train['valeur_O3'].mean())
train['valeur_PM10']=train['valeur_PM10'].fillna(train['valeur_PM10'].mean())
train['valeur_PM25']=train['valeur_PM25'].fillna(train['valeur_PM25'].mean())


train['id'] = pd.to_datetime(train['id'], format='%Y-%m-%d %H')
test['id'] = pd.to_datetime(test['id'], format='%Y-%m-%d %H')


train['year'] = train['id'].dt.year
train['month'] = train['id'].dt.month
train['day'] = train['id'].dt.day
train['hour'] = train['id'].dt.hour
train['dayofweek'] = train['id'].dt.dayofweek

test['year'] = test['id'].dt.year
test['month'] = test['id'].dt.month
test['day'] = test['id'].dt.day
test['hour'] = test['id'].dt.hour
test['dayofweek'] = test['id'].dt.dayofweek


figure=px.line(train,x='id',y='valeur_CO')
figure.show()


figure=px.line(train,x='id',y='valeur_O3')
figure.show()


figure=px.line(train,x='id',y='valeur_PM10')
figure.show()


figure=px.line(train,x='id',y='valeur_PM25')
figure.show()


fig = px.line(train, x='id', y='valeur_NO2', title='NO2 Levels Over Time')
fig.show()


from prophet import Prophet
from prophet.plot import plot_plotly



def train_prophet_model(train_data, test_data, target_column):
    """
    Prophet modelini eğitir ve tahmin üretir.
    """
    # 1. Veriyi Prophet formatına hazırla ('id' -> 'ds', target -> 'y')
    df_prophet = train_data[['id', target_column]].copy()
    df_prophet.columns = ['ds', 'y']
    
    # Eksik verileri çıkar (Prophet NaN sevmez)
    df_prophet = df_prophet.dropna()
    
    # NOT: 'date' sütunu oluşturmaya gerek yok, 'ds' zaten tarih formatında.
    # Prophet modeli otomatik olarak tarih özelliklerini kullanacaktır.

    # 2. Modeli Başlat ve Eğit
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    # 3. Tahmin Yapılacak Tarihleri Oluştur
    future = pd.DataFrame({'ds': test_data['id']})
    
    # 4. Tahmin Yap
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], model


from sklearn.metrics import mean_squared_error, mean_absolute_error

split_date = train['id'].max() - pd.Timedelta(weeks=4)

train_subset = train[train['id'] < split_date].copy()
val_subset = train[train['id'] >= split_date].copy()

print(f"Eğitim Seti Bitiş: {train_subset['id'].max()}")
print(f"Doğrulama Seti Başlangıç: {val_subset['id'].min()}")

# 2. Modeli Eğit
target_col = 'valeur_NO2'  # Hedef değişken
print(f"\n'{target_col}' için model eğitiliyor...")

# Düzeltilmiş fonksiyonu çağırıyoruz
forecast, model = train_prophet_model(train_subset, val_subset, target_col)

# 3. Başarıyı Ölç
# Gerçek değerler ile tahminleri birleştir
val_with_preds = pd.merge(val_subset[['id', target_col]], forecast[['ds', 'yhat']], left_on='id', right_on='ds')
val_with_preds = val_with_preds.dropna()

mse = mean_squared_error(val_with_preds[target_col], val_with_preds['yhat'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(val_with_preds[target_col], val_with_preds['yhat'])

print(f"\n--- Model Başarısı ({target_col}) ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

# 4. Grafik
plt.figure(figsize=(15, 6))
plt.plot(val_with_preds['id'], val_with_preds[target_col], label='Gerçek Değerler', alpha=0.6)
plt.plot(val_with_preds['id'], val_with_preds['yhat'], label='Model Tahmini', color='red', alpha=0.6)
plt.title(f'{target_col} Tahmin Performansı')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\nTraining Prophet model for NO2...")
no2_predictions, no2_model = train_prophet_model(train, test, 'valeur_NO2')


fig = plot_plotly(no2_model, no2_predictions)
fig.show()


pollutants = ['valeur_NO2', 'valeur_CO', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
predictions_dict = {}

for pollutant in pollutants:
    print(f"\nTraining model for {pollutant}...")
    pred, model = train_prophet_model(train, test, pollutant)
    predictions_dict[pollutant] = pred['yhat'].values


submission = test[['id']].copy()



submission['id'] = submission['id'].dt.strftime('%Y-%m-%d %H')


for pollutant in pollutants:
    submission[pollutant] = predictions_dict[pollutant]


submission.to_csv('submission.csv', index=False)



submission


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




