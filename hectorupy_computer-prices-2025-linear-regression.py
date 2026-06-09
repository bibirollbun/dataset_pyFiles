import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_all.csv')


train_data_X = train_data.drop(columns = ['price'])
train_data_Y = train_data[['price']]


train_data.head()


train_data.shape


train_data.info()


train_data[["battery_wh","charger_watts","psu_watts","wifi","bluetooth","weight_kg","warranty_months"]]


train_data[["storage_type", "storage_gb", "storage_drive_count", "display_type", "display_size_in", "resolution", "refresh_hz"]]


train_data[["gpu_brand","gpu_model","gpu_tier","vram_gb","ram_gb"]]


train_data[["ID","device_type","brand","model","release_year","os","form_factor"]]


train_data[["cpu_brand","cpu_model","cpu_tier","cpu_cores","cpu_threads","cpu_base_ghz","cpu_boost_ghz"]]


train_data['form_factor'].unique()


train_data['os'].unique()


train_data['brand'].unique()


train_data['device_type'].unique()


train_data['gpu_brand'].unique()


train_data['display_type'].unique()


train_data['storage_type'].unique()


train_data['cpu_brand'].unique()


train_data['release_year'].unique()


train_data['resolution'].unique()


train_data['wifi'].unique()


train_data['bluetooth'].unique()


train_data['cpu_model'].unique()


train_data['model'].unique()


train_data['gpu_model'].unique()


train_data_X = train_data_X.astype({'release_year':object, 'bluetooth':object})


train_data_X = train_data_X.drop(columns = ['ID','cpu_model','model','gpu_model', 'warranty_months','storage_drive_count'])


years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
bluet = [4.2, 5., 5.1, 5.2, 5.3]
resolution = ['1920x1080','2560x1440','3440x1440','2560x1600','2880x1800','3840x2160']
wifi = ['Wi-Fi 5','Wi-Fi 6', 'Wi-Fi 6E', 'Wi-Fi 7']


enc = OrdinalEncoder(categories = [years,bluet,resolution,wifi])


train_data_X[['release_year','bluetooth','resolution','wifi']] = enc.fit_transform(train_data_X[['release_year','bluetooth','resolution','wifi']])


ohe = OneHotEncoder(sparse_output = False).set_output(transform = 'pandas')


transform = ohe.fit_transform(train_data_X[['form_factor','os','brand','device_type','gpu_brand','display_type','storage_type','cpu_brand']])


train_data_X = pd.concat([train_data_X, transform], axis = 1).drop(columns = ['form_factor','os','brand','device_type','gpu_brand','display_type','storage_type','cpu_brand'])


train_data_X.head()


X_train_valid, X_test, y_train_valid, y_test = train_test_split(train_data_X, train_data_Y, test_size=0.2, random_state = 42)
kf = KFold(5)

for train_indices, validation_indices in kf.split(X_train_valid):
    X_train = X_train_valid.iloc[train_indices]
    X_valid = X_train_valid.iloc[validation_indices]
    y_train = y_train_valid.iloc[train_indices]
    y_valid = y_train_valid.iloc[validation_indices]


model = LinearRegression()
model.fit(X_train, y_train)


y_valid_predict = model.predict(X_valid)
y_train_predict = model.predict(X_train)


def resumen(nombre, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{nombre}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²  : {r2:.3f}")


resumen("TRAIN", y_train, y_train_predict)
resumen("TEST ", y_valid, y_valid_predict)

# 5. Gráfica Real vs Predicho (en test)
plt.scatter(y_valid, y_valid_predict, alpha=0.6)
min_v = min(y_valid.min()['price'], y_valid_predict.min())
max_v = max(y_valid.max()['price'], y_valid_predict.max())
plt.plot([min_v, max_v], [min_v, max_v], "r--")  # línea ideal y=x
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Real vs Predicho")
plt.show()

# 6. Gráfica de residuos (en test)
residuos = y_valid - y_valid_predict
plt.scatter(y_valid_predict, residuos, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicción")
plt.ylabel("Residuo (y_real - y_pred)")
plt.title("Residuos del modelo")
plt.show()


scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("R2 por validación cruzada:", scores)
print("Promedio R2:", scores.mean())


test_data = pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_test.csv')


test_data = test_data.astype({'release_year':object, 'bluetooth':object})
test_data = test_data.drop(columns = ['ID','cpu_model','model','gpu_model', 'warranty_months','storage_drive_count'])
test_data[['release_year','bluetooth','resolution','wifi']] = enc.fit_transform(test_data[['release_year','bluetooth','resolution','wifi']])
transform_test = ohe.fit_transform(test_data[['form_factor','os','brand','device_type','gpu_brand','display_type','storage_type','cpu_brand']])
test_data = pd.concat([test_data, transform_test], axis = 1).drop(columns = ['form_factor','os','brand','device_type','gpu_brand','display_type','storage_type','cpu_brand'])


sub_predict = model.predict(test_data)


sub = pd.read_csv('/kaggle/input/computer-prices-2025/sample_submission.csv')
sub['price'] = sub_predict
sub.to_csv('submission.csv',index=False)




