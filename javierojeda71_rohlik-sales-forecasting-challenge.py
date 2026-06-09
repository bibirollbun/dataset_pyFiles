import pandas as pd
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
holiday = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
train = train.dropna(subset=['sales'])




train = train.merge(weights, on="unique_id", how="left")
test = test.merge(weights, on="unique_id", how="left")





# Convertir la columna de fecha a datetime (si aún no lo está)
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
# Ordenar el DataFrame por fecha en orden ascendente
train = train.sort_values(by="date", ascending=True)

train["year"] = train["date"].dt.year
test["year"] = test["date"].dt.year
train["weekday"] = train["date"].dt.weekday  # Día de la semana (0 = Lunes, 6 = Domingo)
test["weekday"] = test["date"].dt.weekday  # Día de la semana (0 = Lunes, 6 = Domingo)
train["is_weekend"] = (train["weekday"] >= 5).astype(int)
test["is_weekend"] = (test["weekday"] >= 5).astype(int)
train["weekday"] = train["weekday"].astype(str)
test["weekday"] = test["weekday"].astype(str)
train["month_day"] = train["date"].dt.strftime("%m-%d")
test["month_day"] = test["date"].dt.strftime("%m-%d")
train['week_of_year'] = train['date'].dt.isocalendar().week
test['week_of_year'] = train['date'].dt.isocalendar().week
# Crear el DataFrame con los datos del PIB per cápita
ceic_data = {
    "year": [2020, 2021, 2022, 2023, 2024], 
    "GDP per capita": [30431, 33354, 35989, 38091, 39569]
}
data_GDP = pd.DataFrame(ceic_data)

# Suponiendo que 'train' ya es un DataFrame de Pandas
train = train.merge(data_GDP, on="year", how="left")
test = test.merge(data_GDP, on="year", how="left")

print(train.head())  # Ver las primeras filas



import pandas as pd

# Diccionario de almacén a país
warehouse_to_country = {
    "Budapest_1": "Hungary",
    "Prague_2": "Czech Republic",
    "Brno_1": "Czech Republic",
    "Prague_3": "Czech Republic",
    "Frankfurt_1": "Germany",
    "Munich_1": "Germany",
    "Prague_1": "Czech Republic"
}

# Convertir el diccionario en un DataFrame
df_warehouse = pd.DataFrame(list(warehouse_to_country.items()), columns=["warehouse", "country"])
df_warehouse

df_result = df_warehouse.merge(holiday, on="warehouse", how="left")
df_result.columns
df_result["date"] = pd.to_datetime(df_result["date"])

# Filtrar las filas donde 'holiday_name' no es NaN
df_result_valid = df_result[df_result['holiday_name'].notna()]
# Agrupar por país y obtener las vacaciones de cada país en una lista
holidays_per_country = df_result_valid.groupby('country')['holiday_name'].agg(list).reset_index()

# Asegurarnos de que haya días festivos, es decir, holiday_name no es NaN
df_valid = df_result[df_result['holiday_name'].notna()]

# Extraer el día y mes de cada fecha
df_valid['month_day'] = df_valid['date'].dt.strftime('%d-%m')

# Verificar si los días festivos son los mismos en el mismo día y mes cada año para cada país
holidays_same_day_month = df_valid.groupby(['country', 'holiday_name'])['month_day'].nunique()

# Filtrar los días festivos que son consistentes (caen en el mismo día y mes)
consistent_holidays = holidays_same_day_month[holidays_same_day_month == 1]

# Filtrar los registros con los días festivos consistentes
result = df_valid[df_valid['holiday_name'].isin(consistent_holidays.index.get_level_values(1))]

# Seleccionar las columnas necesarias
result = result[['month_day', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays',"warehouse"]]

# Eliminar duplicados
result = result.drop_duplicates()
train = train.merge(result, on=['warehouse', 'month_day'], how='left')
test = test.merge(result, on=['warehouse', 'month_day'], how='left')
train.fillna({"school_holidays": 0, "holiday": 0, "shops_closed": 0, 'winter_school_holidays': 0}, inplace=True)
test.fillna({"school_holidays": 0, "holiday": 0, "shops_closed": 0, 'winter_school_holidays': 0}, inplace=True)



# Agrupar para calcular la media de ventas por warehouse y día de la semana
avg_sales_by_warehouse = train.groupby(["warehouse", "weekday"])["sales"].mean().reset_index()

# Ordenar por día de la semana (0 = Lunes, 6 = Domingo) (Opcional, no es necesario para merge)
avg_sales_by_warehouse = avg_sales_by_warehouse.sort_values(by=["warehouse", "weekday"])

# Renombrar la columna de ventas promedio para evitar sobreescrituras
avg_sales_by_warehouse.rename(columns={"sales": "avg_sales_by_warehouse_by_day"}, inplace=True)

# Hacer merge con train para añadir la media de ventas por warehouse y día de la semana
train = train.merge(avg_sales_by_warehouse, on=["warehouse", "weekday"], how="left")
test = test.merge(avg_sales_by_warehouse, on=["warehouse", "weekday"], how="left")
test["avg_sales_by_warehouse_by_day"].fillna(train["sales"].mean(), inplace=True)



# Calcular la media de 'sell_price_main' por cada 'warehouse'
mean_price_by_warehouse = train.groupby('warehouse')['sell_price_main'].mean()

# Mapear la media calculada a los valores del dataset original
train['mean_sell_price_per_warehouse'] = train['warehouse'].map(mean_price_by_warehouse)
test['mean_sell_price_per_warehouse'] = test['warehouse'].map(mean_price_by_warehouse)



test.isnull().sum()


# Hacer merge entre `train` e `inventory` por `unique_id`
train2 = pd.merge(train, inventory[['unique_id', 'L1_category_name_en',"L2_category_name_en","name"]], on='unique_id', how='left')
test2 = pd.merge(test, inventory[['unique_id', 'L1_category_name_en',"L2_category_name_en","name"]], on='unique_id', how='left')



# Agrupar para calcular la media de ventas por categoría y semana
avg_sales_L1_category_name_en = train2.groupby(["L1_category_name_en", "week_of_year"])["sales"].mean().reset_index()

# Renombrar la columna de ventas promedio para evitar sobreescrituras
avg_sales_L1_category_name_en.rename(columns={"sales": "avg_sales_by_category_by_w"}, inplace=True)

# Hacer merge con train para añadir la media de ventas por categoría y semana
train2 = train2.merge(avg_sales_L1_category_name_en, on=["L1_category_name_en", "week_of_year"], how="left")
test2 = test2.merge(avg_sales_L1_category_name_en, on=["L1_category_name_en", "week_of_year"], how="left")
test2["avg_sales_by_category_by_w"].fillna(train2["sales"].mean(), inplace=True)



# Crear una nueva columna que sea la suma de las interacciones entre el precio y todos los descuentos
discount_interaction = (
    train2['sell_price_main'] * train2['type_0_discount'] +
    train2['sell_price_main'] * train2['type_1_discount'] +
    train2['sell_price_main'] * train2['type_2_discount'] +
    # Añadir más descuentos si los tienes
    train2['sell_price_main'] * train2['type_3_discount'] +
    train2['sell_price_main'] * train2['type_4_discount'] +
    train2['sell_price_main'] * train2['type_5_discount'] +
    train2['sell_price_main'] * train2['type_6_discount']
)

# Asignar la nueva columna a train2
train2['discount_interaction_sum'] = discount_interaction

train2['any_discount'] = (
    (train2['type_0_discount'] > 0) |
    (train2['type_1_discount'] > 0) |
    (train2['type_2_discount'] > 0) |
    (train2['type_3_discount'] > 0) |
    (train2['type_4_discount'] > 0) |
    (train2['type_5_discount'] > 0) |
    (train2['type_6_discount'] > 0)
).astype(int)

# Calcular la interacción de descuentos en test2
discount_interaction_test2 = (
    test2['sell_price_main'] * test2['type_0_discount'] +
    test2['sell_price_main'] * test2['type_1_discount'] +
    test2['sell_price_main'] * test2['type_2_discount'] +
    test2['sell_price_main'] * test2['type_3_discount'] +
    test2['sell_price_main'] * test2['type_4_discount'] +
    test2['sell_price_main'] * test2['type_5_discount'] +
    test2['sell_price_main'] * test2['type_6_discount']
)

# Asignar la nueva columna a test2
test2['discount_interaction_sum'] = discount_interaction_test2

# Crear la variable any_discount en test2
test2['any_discount'] = (
    (test2['type_0_discount'] > 0) |
    (test2['type_1_discount'] > 0) |
    (test2['type_2_discount'] > 0) |
    (test2['type_3_discount'] > 0) |
    (test2['type_4_discount'] > 0) |
    (test2['type_5_discount'] > 0) |
    (test2['type_6_discount'] > 0)
).astype(int)



train2["month"] = train2["date"].astype(str).str[5:7].astype(int)
test2["month"] = test2["date"].astype(str).str[5:7].astype(int)

train2["warehouse"] = train2["warehouse"].astype(str)
test2["warehouse"] = test2["warehouse"].astype(str)

train2["orders_price"] = train2["total_orders"] * train2["sell_price_main"]
test2["orders_price"] = test2["total_orders"] * test2["sell_price_main"]


# Calcular los cuartiles
Q1 = train2['sell_price_main'].quantile(0.25)
Q3 = train2['sell_price_main'].quantile(0.75)

# Calcular el rango intercuartílico
IQR = Q3 - Q1

# Definir los intervalos basados en el rango intercuartílico
price_bins = [train2['sell_price_main'].min(), Q1 - 1.5 * IQR, Q1, Q3, Q3 + 1.5 * IQR, train2['sell_price_main'].max()]

# Asegurarse de que los bins estén en orden creciente
price_bins = sorted(price_bins)

# Crear las etiquetas para los intervalos de precio
price_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Crear una nueva columna con la categoría de precio
train2['price_category'] = pd.cut(train2['sell_price_main'], bins=price_bins, labels=price_labels)
train2['price_category'] = train2['price_category'].astype(str)
test2['price_category'] = pd.cut(test2['sell_price_main'], bins=price_bins, labels=price_labels).astype(str)

# Agrupar por L1_category_name_en y price_category, luego calcular la media de ventas
avg_sales_by_category_and_price = train2.groupby(['L1_category_name_en', 'price_category'])['sales'].mean().reset_index()

# Renombrar la columna de ventas promedio para evitar confusión
avg_sales_by_category_and_price.rename(columns={'sales': 'avg_sales_by_category_price'}, inplace=True)

# Hacer merge con el dataframe original para agregar la media de ventas por categoría de producto y precio
train2 = train2.merge(avg_sales_by_category_and_price, on=['L1_category_name_en', 'price_category'], how='left')
test2 = test2.merge(avg_sales_by_category_and_price, on=['L1_category_name_en', 'price_category'], how='left')
test2["avg_sales_by_category_price"].fillna(train2["sales"].mean(), inplace=True)

# Mostrar el dataframe actualizado con la media de ventas por categoría de producto y precio
print(train2[['L1_category_name_en', 'price_category', 'sales', 'avg_sales_by_category_price']].head(20))



avg_sales_by_category_and_price_mo = train2.groupby(['month', 'price_category'])['sales'].mean().reset_index()

# Renombrar la columna de ventas promedio para evitar confusión
avg_sales_by_category_and_price_mo.rename(columns={'sales': 'avg_sales_by_month_price'}, inplace=True)

# Hacer merge con el dataframe original para agregar la media de ventas por categoría de producto y precio
train2 = train2.merge(avg_sales_by_category_and_price_mo, on=['month', 'price_category'], how='left')
test2 = test2.merge(avg_sales_by_category_and_price_mo, on=['month', 'price_category'], how='left')
test2["avg_sales_by_month_price"].fillna(train2["sales"].mean(), inplace=True)

# Mostrar el dataframe actualizado con la media de ventas por categoría de producto y precio
print(train2[['month', 'price_category', 'sales', 'avg_sales_by_month_price']].head(50))




test2["price_category"].unique()


avg_sales_by_productweek = train2.groupby(['month', 'L2_category_name_en'])['sales'].mean().reset_index()

# Renombrar la columna de ventas promedio para evitar confusión
avg_sales_by_productweek.rename(columns={'sales': 'avg_sales_by_productweek'}, inplace=True)

# Hacer merge con el dataframe original para agregar la media de ventas por categoría de producto y precio
train2 = train2.merge(avg_sales_by_productweek, on=['month', 'L2_category_name_en'], how='left')
test2 = test2.merge(avg_sales_by_productweek, on=['month', 'L2_category_name_en'], how='left')
test2["avg_sales_by_productweek"].fillna(train2["sales"].mean(), inplace=True)

# Mostrar el dataframe actualizado con la media de ventas por categoría de producto y precio
print(train2[['month', 'L2_category_name_en', 'sales', 'avg_sales_by_productweek']].head(50))


# Agrupar para calcular la media de ventas por warehouse y día de la semana
avg_sales_by_name = train2.groupby("name")["sales"].mean().reset_index()


# Renombrar la columna de ventas promedio para evitar sobreescrituras
avg_sales_by_name.rename(columns={"sales": "avg_sales_by_name"}, inplace=True)

# Hacer merge con train para añadir la media de ventas por warehouse y día de la semana
train2 = train2.merge(avg_sales_by_name, on=["name"], how="left")
test2 = test2.merge(avg_sales_by_name, on=["name"], how="left")

test2["avg_sales_by_name"].fillna(train2["sales"].mean(), inplace=True)

test2["name_weight"] = test2["avg_sales_by_name"]* test2["weight"]

train2["name_weight"] = train2["avg_sales_by_name"]* train2["weight"]



total_sales_per_category = train2.groupby('L2_category_name_en')['sales'].sum()
low_threshold, high_threshold = total_sales_per_category.quantile([0.33, 0.66])

train2['sales_category'] = train2['L2_category_name_en'].map(lambda x: 'Poco Vendido' if total_sales_per_category[x] <= low_threshold
                                                            else 'Vendido Normal' if total_sales_per_category[x] <= high_threshold
                                                            else 'Muy Vendido')
test2['sales_category'] = test2['L2_category_name_en'].map(lambda x: 'Poco Vendido' if total_sales_per_category[x] <= low_threshold
                                                           else 'Vendido Normal' if total_sales_per_category[x] <= high_threshold
                                                           else 'Muy Vendido')




train2["media todo"] = (train2["avg_sales_by_category_price"]+train2["avg_sales_by_month_price"]+train2["avg_sales_by_category_by_w"]+train2["avg_sales_by_warehouse_by_day"]+train2["avg_sales_by_productweek"])/5
test2["media todo"] = (test2["avg_sales_by_category_price"]+test2["avg_sales_by_month_price"]+test2["avg_sales_by_category_by_w"]+test2["avg_sales_by_warehouse_by_day"]+test2["avg_sales_by_productweek"])/5




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Total de filas en el conjunto de datos
total_filas = len(train2)
total_filas_test = len(test2)

# Umbral del 20% de valores nulos
umbral = total_filas * 0.2
umbral_test = total_filas_test * 0.2
# Filtramos las columnas que tengan más de un 20% de valores nulos
columnas_a_conservar = train2.columns[train2.notnull().sum() > umbral]

columnas_a_conservar_test = test2.columns[test2.notnull().sum() > umbral_test]
# Mostrar las columnas a conservar
print("Columnas a conservar:", columnas_a_conservar)

train_filtrado = train2[columnas_a_conservar]
test_filtrado = test2[columnas_a_conservar_test]

# Separar variables independientes y dependientes
x_train = train_filtrado.drop(['unique_id', "date", "month_day", "availability", "L2_category_name_en","name", "weight"], axis=1)
x_test = test_filtrado.drop(['unique_id',"date", "month_day","L2_category_name_en","name", "weight"], axis=1)

# Identificar columnas con valores nulos
coltofix = x_train.columns[x_train.isnull().any()]
coltofixtest = x_test.columns[x_test.isnull().any()]

# Reemplazar valores nulos (moda para categóricas, media para numéricas)
for col in coltofix:
    if x_train[col].dtype == "object":
        x_train[col].fillna(x_train[col].mode()[0], inplace=True)
    else:
        x_train[col].fillna(x_train[col].mean(), inplace=True)

# Aplicar el mismo tratamiento al dataset de test
for col in coltofixtest:
    if x_test[col].dtype == "object":
        x_test[col].fillna(x_train[col].mode()[0], inplace=True)
    else:
        x_test[col].fillna(x_train[col].mean(), inplace=True)

# Normalizar columnas numéricas excluyendo 'year_hct'
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
numerical_cols_test = x_test.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
cols_to_scale = [col for col in numerical_cols if col != "sales"]
x_train[cols_to_scale] = scaler.fit_transform(x_train[cols_to_scale])
x_test[numerical_cols_test] = scaler.fit_transform(x_test[numerical_cols_test])

# Codificación one-hot para variables categóricas
categorical_cols = x_train.select_dtypes(include=['object']).columns
categorical_cols_test = x_test.select_dtypes(include=['object']).columns
x_train_encoded = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=categorical_cols_test, drop_first=True)


# Convertir booleanos a enteros
bool_cols = x_train_encoded.select_dtypes(include=['bool']).columns
bool_cols_test = x_test_encoded.select_dtypes(include=['bool']).columns
x_train_encoded[bool_cols] = x_train_encoded[bool_cols].astype(int)
x_test_encoded[bool_cols_test] = x_test_encoded[bool_cols_test].astype(int)

# DataFrames finales
df_train_final = x_train_encoded
df_test_final = x_test_encoded


faltantes_en_test = [var for var in df_train_final.columns if var not in df_test_final.columns]
print(f"Columnas en train pero no en test: {faltantes_en_test}")



# Lista de columnas esperadas después del OneHotEncoder
expected_categories = ['price_category_Low', 'price_category_Medium', 'price_category_Very High', 'price_category_Very Low']

# Verificar qué columnas faltan en test2
missing_categories = [col for col in expected_categories if col not in df_test_final.columns]

# Agregar las categorías faltantes con valor 0
for col in missing_categories:
    df_test_final[col] = 0



faltantes_en_test = [var for var in df_train_final.columns if var not in df_test_final.columns]
print(f"Columnas en train pero no en test: {faltantes_en_test}")


df_train_final

XX = df_train_final.drop(['sales'], axis=1)
y_train1 = df_train_final[['sales']]

df_train_final.columns


# Asegurar que y_train1 tiene los mismos índices que df_train_final

# Unir X (df_train_final) con y (y_train1) en un solo DataFrame
df_combined = df_train_final.copy()

# Tomar la muestra aleatoria de 20,000 filas
sample_df = df_combined.sample(n=min(40000, len(df_combined)), random_state=42)

# Separar nuevamente en X e y
X_sample = sample_df.drop(columns=["sales"])
y_sample = sample_df["sales"]

# Verificar si hay nulls
print(y_sample.isnull().sum())  # Debería ser 0



import lightgbm as lgb
import numpy as np

# Crear el modelo de Random Forest Regressor
model = lgb.LGBMRegressor(
    random_state=42,
    num_leaves=63,               # Número de hojas de los árboles. Aumentamos para un mejor ajuste
    max_depth=20,                # Profundidad máxima del árbol. Limitar para evitar sobreajuste
    learning_rate=0.01,          # Tasa de aprendizaje más baja para evitar sobreajuste y converger lentamente
    n_estimators=1500,            # Número de árboles. Incrementado para asegurar que el modelo aprenda bien
    subsample=0.8,               # Submuestra el dataset para evitar sobreajuste
    colsample_bytree=0.8,        # Submuestra las columnas para evitar sobreajuste
    min_child_samples=50,        # Número mínimo de muestras necesarias en una hoja de un árbol
    reg_alpha=0.1,               # Regularización L1 para evitar sobreajuste
    reg_lambda=0.1,              # Regularización L2 para evitar sobreajuste
    n_jobs=-1                    # Usar todos los núcleos del procesador para acelerar el entrenamiento
)
# Entrenar el modelo con el conjunto de entrenamiento
model.fit(XX, y_train1)

# Suponiendo que 'df_test_final' contiene las mismas columnas que 'df_train_final'
predictions_test = model.predict(df_test_final)

# Convertir 'predictions_test' en un array unidimensional si es necesario
predictions_test = predictions_test.ravel()

test["unique_id"] = test["unique_id"].astype(str)
test["date"] = test["date"].astype(str)

test["id"] = test["unique_id"] + "_" + test["date"]

submission = pd.DataFrame({
    'id': test['id'],  # Asegúrate de que 'test' contiene la columna 'id'
    'sales_hat': predictions_test  # Asegúrate de que las predicciones son unidimensionales
})

# Guardar el DataFrame como un archivo CSV para la submission
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Archivo 'submission.csv' creado correctamente.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Obtener la importancia de las variables
importances = rf.feature_importances_
features = X_sample.columns

# Crear un DataFrame con las importancias
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Visualizar la importancia de las variables
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df[:20])  # Top 20 features
plt.title("Importancia de las Variables en RandomForest")
plt.show()

# Evaluar el modelo
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")


