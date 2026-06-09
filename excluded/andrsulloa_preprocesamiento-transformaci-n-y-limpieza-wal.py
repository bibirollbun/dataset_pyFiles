import pandas as pd
import gc
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Cargar los conjuntos de datos
sales_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
calendar_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
prices_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


#Visualizar la información del conjunto sales
sales_data.info()


#Visualizar la información del conjunto calendar
calendar_data.info()


#Visualizar la información del conjunto prices
prices_data.info()


# Eliminar ciertas columnas innecesarias
calendar_data = calendar_data.drop(columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_TX', 'snap_WI'])


# Transformar las columnas 'd_1', 'd_2', ..., 'd_1913' de la tabla de ventas en una sola columna 'day'
sales_data_melted = pd.melt(sales_data, 
                            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                            var_name='d', 
                            value_name='sales')


# Unión izquierda de los archivos "sales" y "calendar" a través de la veriable en común "d" 
merged_data = pd.merge(sales_data_melted, calendar_data, on='d', how='left')


# Eliminar variables que ya no se necesitan
del sales_data
del sales_data_melted
del calendar_data

# Liberar memoria
gc.collect()


# Unión izquierda del archivo "prices" considerando las varaibles comunes "store_id" ; "item_id" y "wm_yr_wk".
final_data = pd.merge(merged_data, prices_data, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])


# Eliminar variables que ya no se necesitan
del merged_data
del prices_data

# Liberar memoria
gc.collect()


#Filtrar el conjunto de datos para quedarnos únicamente con la tienda "CA_1"
df_filtered = final_data[final_data['store_id'] == 'CA_1']


# Eliminar variables que ya no se necesitan
del final_data

# Liberar memoria
gc.collect()


df_filtered["date"] = pd.to_datetime(df_filtered["date"])


# Definir el rango de tiempo para los últimos 3 años
end_date = df_filtered['date'].max()  # Fecha máxima en el dataset
start_date = end_date - pd.DateOffset(years=3)  # Fecha 3 años antes de la fecha máxima

# Filtrar el dataset para incluir solo los registros de los últimos 3 años
df_last_3_years = df_filtered[df_filtered['date'] >= start_date]


# Eliminar variables que ya no se necesitan
del df_filtered

# Liberar memoria
gc.collect()


# Eliminar las columnas innecesarias
columns_to_drop = ['id', 'd', 'wm_yr_wk', 'weekday', 'store_id', 'state_id', 'wday', 'month', 'year']
df_last_3_years = df_last_3_years.drop(columns=columns_to_drop)


# Eliminar variables que ya no se necesitan
del columns_to_drop
del end_date
del start_date

# Liberar memoria
gc.collect()


df_last_3_years.info()


# Exportar el dataset final
df_last_3_years.to_csv('df_last_3_years.csv', index=False)


#Cargar el dataset
df = pd.read_csv('/kaggle/input/df-last-3-years-final/df_last_3_years.csv')


# Eliminar variables que ya no se necesitan
del df_last_3_years

# Liberar memoria
gc.collect()


# Explorar los valores faltantes
missing_data = df.isnull().sum().sort_values(ascending=False)

print("Valores faltantes en cada columna:")
print(missing_data)


# Imputar los valores faltantes de sell_price con la mediana por item_id
df['sell_price'] = df.groupby('item_id')['sell_price'].transform(lambda x: x.fillna(x.median()))


# Verificar si quedan valores faltantes en sell_price
df['sell_price'].isnull().sum()


df.info()


# Convertir la columna 'date' a tipo datetime.
df['date'] = pd.to_datetime(df['date'])


# Verificar si hay filas duplicadas en todo el dataset
duplicated_rows = df.duplicated()

# Contar el número de filas duplicadas
num_duplicated_rows = df.duplicated().sum()
print(f"Número de filas duplicadas: {num_duplicated_rows}")

# Mostrar las filas duplicadas si existen
df_duplicated = df[df.duplicated()]
print(df_duplicated)


# Filtrar datos por categoría "FOODS" para análisis a nivel de categoría.
df_foods = df[df['cat_id'] == 'FOODS'].copy()

# Filtrar datos por item_id "FOODS_2_347" para análisis a nivel de ítem o producto.
df_foods_2_347 = df[df['item_id'] == 'FOODS_2_347'].copy()


# Realizar las agregaciones relevantes para la categoría "FOODS"
df_foods_agg = df_foods.groupby('date').agg({
    'sales': 'sum',  # Sumar ventas totales para cada fecha
    'sell_price': 'mean',  # Promedio de precios de venta
    'snap_CA': 'max',  # Máximo valor de snap_CA por semana (si hubo descuento)
}).reset_index()

# Renombrar la columna de ventas a 'sales_total'
df_foods_agg.rename(columns={'sales': 'sales_total'}, inplace=True)


df_foods_agg.info()


df_foods_agg.head(3)


# Realizar las agregaciones relevantes para este conjunto de datos
df_foods_2_347_agg = df_foods_2_347.groupby('date').agg({
    'sales': 'sum',  # Sumar ventas totales para cada fecha
    'sell_price': 'mean',  # Promedio de precios de venta
    'snap_CA': 'max',  # Máximo valor de snap_CA por semana (si hubo descuento)
}).reset_index()

# Renombrar la columna de ventas a 'sales_total'
df_foods_2_347_agg.rename(columns={'sales': 'sales_total'}, inplace=True)

# Resetear el índice
df_foods_2_347_agg.reset_index(drop=True, inplace=True)


df_foods_2_347_agg.info()


df_foods_2_347_agg.head(3)

