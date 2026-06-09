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


import matplotlib.pyplot as plt
import seaborn as sns

# ConfiguraciÃ³n para visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar datos
df = pd.read_csv('/kaggle/input/autism-prediction/Autism-prediction/train.csv')  # Cambia el nombre del archivo segÃºn tu caso

# Mostrar las primeras filas de datos
df.head()


# InformaciÃ³n general
print("InformaciÃ³n del dataset:")
df.info()

print("\nDescripciÃ³n estadÃ­stica:")
df.describe()

print("\nValores nulos por columna:")
print(df.isnull().sum())

#float64 = nÃºmeros decimales (heart rate, glucose,temperature,etc.) o valores faltantes (NaN)
#objetc = columnas con texto (glascow coma scale eye opening)
#int64 = nÃºmeros enteros (target)
#Non-null count = nÃºmero de filas en columna que tienen un valor real


# Detectar columnas de texto (object o category)
text_cols = df.select_dtypes(include=['object', 'category']).columns

for col in text_cols:
    # Contar frecuencias (incluyendo NaN si existen)
    counts = df[col].value_counts(dropna=False)
    
    # Limitar a top 10 si hay muchas categorÃ­as
    if len(counts) > 15:
        counts = counts.head(10)
        title = f'{col} (Top 10)'
    else:
        title = col

    # GrÃ¡fico de barras
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title(f'Frecuencia: {title}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

    # GrÃ¡fico de pastel (solo si â‰¤ 10 categorÃ­as)
    if len(counts) <= 10:
        plt.figure(figsize=(8, 8))
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'DistribuciÃ³n: {col}')
        plt.tight_layout()
        plt.show()


#VisualizaciÃ³n del nombre de las columnas (variables)
print(df.columns.tolist())


#df = df.drop(columns=['nombre_de_la_columna', o 'columnas'])
df = df.drop(columns=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'])


#VisualizaciÃ³n del nombre de las columnas (variables)
print(df.columns.tolist())


def moda(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan

numeric_cols = df.select_dtypes(include=[np.number]).columns

df_stats = pd.DataFrame({
    'Media': df[numeric_cols].mean(),
    'Mediana': df[numeric_cols].median(),
    'Moda': df[numeric_cols].apply(moda),
    'MÃ­nimo': df[numeric_cols].min(),
    'MÃ¡ximo': df[numeric_cols].max()
})

print(df_stats.round(2))


# Definir lÃ­mites clÃ­nicos razonables (ajusta segÃºn tu dominio)
clinical_limits = {
    'ID': (1, 800),
    'age': (9, 72),
    'result': (-3.00, 13.00),
    'Class/ASD': (0.00, 1.00),
}

# Detectar outliers por columna (si la columna estÃ¡ en clinical_limits)
outliers_info = {}

for col, (min_val, max_val) in clinical_limits.items():
    if col in df.columns:
        # Contar valores fuera de rango
        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()
        total_outliers = below + above
        
        if total_outliers > 0:
            outliers_info[col] = {
                'outliers': total_outliers,
                'below_min': below,
                'above_max': above,
                'min_observed': df[col].min(),
                'max_observed': df[col].max()
            }

# Mostrar resultados
for col, info in outliers_info.items():
    print(f"\nğŸ”� {col}:")
    print(f"  - Valores < {clinical_limits[col][0]}: {info['below_min']}")
    print(f"  - Valores > {clinical_limits[col][1]}: {info['above_max']}")
    print(f"  - Total outliers: {info['outliers']}")
    print(f"  - Rango observado: [{info['min_observed']:.2f}, {info['max_observed']:.2f}]") #Rango observado entre el valor mÃ­nimo y mÃ¡ximo


# Reemplazar valores faltantes (NaN) por la media de cada columna numÃ©rica
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("Valores nulos despuÃ©s de imputaciÃ³n:")
print(df.isnull().sum())


# Supongamos que tu columna se llama 'Glascow coma scale motor response'
col_name = 'jaundice'

# Definir mapeo de categorÃ­as originales a categorÃ­as agrupadas
mapping = {
    'yes': '1',
    'no': '2', 
}

# Crear nueva columna con categorÃ­as agrupadas
df['GCS_Motor_Grouped'] = df[col_name].map(mapping)

# Mostrar frecuencias despuÃ©s de agrupar
print("\nFrecuencia de categorÃ­as agrupadas:")
print(df['GCS_Motor_Grouped'].value_counts().sort_index())


# Supongamos que tu columna se llama 'Glascow coma scale motor response'
col_name = 'austim'

# Definir mapeo de categorÃ­as originales a categorÃ­as agrupadas
mapping = {
    'yes': '1',
    'no': '2',
}

# Crear nueva columna con categorÃ­as agrupadas
df['GCS_Motor_Grouped'] = df[col_name].map(mapping)

# Mostrar frecuencias despuÃ©s de agrupar
print("\nFrecuencia de categorÃ­as agrupadas:")
print(df['GCS_Motor_Grouped'].value_counts().sort_index())


# Supongamos que tu columna se llama 'Glascow coma scale motor response'
col_name = 'used_app_before'

# Definir mapeo de categorÃ­as originales a categorÃ­as agrupadas
mapping = {
    'yes': '1',
    'no': '2', 
}

# Crear nueva columna con categorÃ­as agrupadas
df['GCS_Motor_Grouped'] = df[col_name].map(mapping)

# Mostrar frecuencias despuÃ©s de agrupar
print("\nFrecuencia de categorÃ­as agrupadas:")
print(df['GCS_Motor_Grouped'].value_counts().sort_index())


print("=== ESTADÃ�STICAS DESCRIPTIVAS ===\n")

for col in numeric_cols:
    print(f"\n--- {col} ---")
    print(f"MÃ¡ximo: {df[col].max()}")
    print(f"MÃ­nimo: {df[col].min()}")
    print(f"Promedio (Media): {df[col].mean():.2f}")
    print(f"Mediana: {df[col].median():.2f}")
    print(f"Moda: {df[col].mode()[0] if not df[col].mode().empty else 'No hay moda Ãºnica'}")

