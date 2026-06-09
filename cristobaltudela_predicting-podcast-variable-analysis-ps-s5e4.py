import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Cargar datos
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# Analizar Genre
print('ANÁLISIS DE GENRE')
print(f'Valores únicos: {train_data["Genre"].unique().tolist()}')
print('\nDistribución de valores:')
genre_counts = train_data['Genre'].value_counts()
print(genre_counts)
print(f'\nPorcentaje de cada categoría:')
print((genre_counts / len(train_data) * 100).round(2))

# Relación con variable objetivo (promedio de Listening_Time_minutes por categoría)
print('\nPromedio de tiempo de escucha por Genre:')
genre_listen = train_data.groupby('Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(genre_listen)

# Analizar Publication_Day
print('\n\nANÁLISIS DE PUBLICATION_DAY')
print(f'Valores únicos: {train_data["Publication_Day"].unique().tolist()}')
print('\nDistribución de valores:')
day_counts = train_data['Publication_Day'].value_counts()
print(day_counts)
print(f'\nPorcentaje de cada categoría:')
print((day_counts / len(train_data) * 100).round(2))

# Relación con variable objetivo
print('\nPromedio de tiempo de escucha por Publication_Day:')
day_listen = train_data.groupby('Publication_Day')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(day_listen)

# Analizar Publication_Time
print('\n\nANÁLISIS DE PUBLICATION_TIME')
print(f'Valores únicos: {train_data["Publication_Time"].unique().tolist()}')
print('\nDistribución de valores:')
time_counts = train_data['Publication_Time'].value_counts()
print(time_counts)
print(f'\nPorcentaje de cada categoría:')
print((time_counts / len(train_data) * 100).round(2))

# Relación con variable objetivo
print('\nPromedio de tiempo de escucha por Publication_Time:')
time_listen = train_data.groupby('Publication_Time')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(time_listen)

