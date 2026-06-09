import pandas as pd
import numpy as np
import random
from path import Path
import matplotlib.pyplot as plt
import seaborn as sns


crab = pd.read_csv("/kaggle/input/crab-age-prediction/CrabAgePrediction.csv")
crab["Length"] = crab["Length"] * 30.48 
crab["Diameter"] = crab["Diameter"] * 30.48 
crab["Height"] = crab["Height"] * 30.48 
crab["Weight"] = crab["Weight"] * 30.48 
crab["Shucked Weight"] = crab["Shucked Weight"] * 28.3495
crab["Viscera Weight"] = crab["Viscera Weight"] * 28.3495
crab["Shell Weight"] = crab["Shell Weight"]*28.3495
crab.info()
crab.describe()
crab.to_csv("CrabAgePrediction_SI.csv", index=False)


crab.info()
crab.describe()


import pandas as pd
import matplotlib.pyplot as plt

# Asumiendo que 'crab' es tu DataFrame
# crab = pd.read_csv('/kaggle/input/crab-age-prediction/CrabAgePrediction.csv')
# Asegúrate de que tu DataFrame 'crab' esté cargado antes de ejecutar esto.

# --- Mapeo de colores y nombres ---
# Asumo 'F' (Female), 'M' (Male), 'I' (Indeterminate/Juvenile)
color_map = {
    'F': '#1f78b4',  # Azul claro
    'M': '#a6cee3',  # Azul oscuro/medio
    'I': '#b2df8a'   # Verde/amarillo claro
}

# Diccionario para traducir las abreviaturas a nombres en español para la leyenda
leyenda_espanol = {
    'F': 'H',
    'M': 'M',
    'I': 'I'
}

# Si tus valores en 'Sex' fueran 'H', 'M', 'I' (para 'H'embra), usa:
# leyenda_espanol = {
#     'H': 'Hembra',
#     'M': 'Macho',
#     'I': 'Indeterminado/Joven'
# }
# Y actualiza el 'color_map' consecuentemente.


fig, ax = plt.subplots(figsize=(10, 6))

for sex_type, group_data in crab.groupby('Sex'):
    # Usar 'sex_type' para el color y 'leyenda_espanol[sex_type]' para la etiqueta
    if sex_type in color_map:
        group_data['Age'].hist(
            ax=ax,
            bins=15,
            alpha=0.6,
            color=color_map[sex_type],
            # *** CAMBIO CLAVE AQUÍ: Usar el diccionario de traducción ***
            label=leyenda_espanol.get(sex_type, sex_type) # Usa la traducción o la abreviatura si no encuentra la clave
        )

# === Personalizar el gráfico ===
ax.set_title('Distribución de Edad por Sexo de los Cangrejos')
ax.set_xlabel('Edad (Años)')
ax.set_ylabel('Frecuencia (Número de Individuos)')
ax.grid(axis='y', alpha=0.75)
ax.legend(title='Sexo') # El título de la leyenda ya está en español

plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que el DataFrame ya está cargado como 'crab'
# crab = pd.read_csv('/Kaggle/input/crab-age-prediction/CrabAgePrediction.csv')

# Lista de las variables que queremos incluir en el gráfico
variables_seleccionadas = ['Length', 'Diameter', 'Height', 'Weight', 'Age']
plt.subplots_adjust(top=0.95)
# Generar el gráfico de dispersión matricial, usando solo las columnas seleccionadas
ax=sns.pairplot(crab, vars=variables_seleccionadas)


# Mostrar el gráfico
plt.show()



plt.figure(figsize = (25,10))
ax=sns.heatmap(crab.corr(numeric_only=True), annot = True, cmap="coolwarm")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')


plt.figure(figsize=(10,6))
sns.lineplot(
    data=crab.groupby("Age", as_index=False)["Shucked Weight"].mean(),
    x="Age", y="Shucked Weight", marker="o"
)
plt.title("Promedio de peso de la carne del cangrejo según la edad")
plt.xlabel("Edad (años)")
plt.ylabel("Peso promedio de la carne")
plt.show()


# Set the style for the plot to make it visually appealing
sns.set_style("whitegrid")

# Create a figure and axes for the plot
plt.figure(figsize=(10, 6))

# Use a histogram to show the frequency of each age
sns.histplot(data=crab, x='Age', bins=20, kde=True, color='skyblue')

# Add a title and labels for clarity
plt.title('Distribution of Crab Ages', fontsize=16)
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show the plot
plt.show()

# --- Alternative Plot: KDE Plot ---
# This plot shows the probability density of ages, which can be smoother
plt.figure(figsize=(10, 6))
sns.kdeplot(data=crab, x='Age', fill=True, color='purple')
plt.title('Kernel Density Estimate of Crab Ages', fontsize=16)
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

crab_filtered = crab[crab['Age'] <= 22].copy()
crab_filtered['Combined Weight'] = crab_filtered['Shucked Weight'] + crab_filtered['Viscera Weight']

crab_filtered['Benefit per Month'] = crab_filtered['Combined Weight'] / crab_filtered['Age']

plt.figure(figsize=(12, 8))

sns.boxplot(x='Age', y='Benefit per Month', data=crab_filtered)

plt.title("Distribución del beneficio por mes invertido (peso combinado / edad)", fontsize=16)
plt.xlabel("Edad (meses)", fontsize=12)
plt.ylabel("Peso de carne + vísceras / Edad", fontsize=12)

plt.xticks(rotation=45)

plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que 'crab' ya está cargado y preprocesado
# crab = pd.read_csv('/kaggle/input/crab-age-prediction/CrabAgePrediction.csv')

# ---
# Código para recrear el DataFrame 'crab_filtered' con una columna 'Sex'
# para fines de demostración, ya que el gráfico original no la usa
crab_filtered = crab[crab['Age'] <= 22].copy()
crab_filtered['Combined Weight'] = crab_filtered['Shucked Weight'] + crab_filtered['Viscera Weight']
crab_filtered['Benefit per Month'] = crab_filtered['Combined Weight'] / crab_filtered['Age']
# Nota: La columna 'Sex' ya existe en el dataset original.
# Si el dataset tuviera valores 'I' (indefinido), 'M' (macho) y 'F' (hembra),
# este código funcionará.
# ---

plt.figure(figsize=(14, 8))

# Crea el boxplot, usando 'hue' para diferenciar por sexo
sns.boxplot(x='Age', y='Benefit per Month', data=crab_filtered, hue='Sex', palette='Paired')

# Añade títulos y etiquetas
plt.title('Distribución del beneficio por mes invertido (peso combinado / edad) por sexo', fontsize=16)
plt.xlabel('Edad (meses)', fontsize=12)
plt.ylabel('Peso de carne + vísceras / Edad', fontsize=12)
plt.xticks(rotation=45)

# Ajusta la leyenda para mayor claridad
plt.legend(title='Sexo')

# Muestra el gráfico
plt.show()


crabs = crab.copy()
ax=crabs.plot(kind = "scatter", x = 'Length', y = 'Height', grid = True,
          s = crabs["Weight"]*0.1, label = 'Peso', c = "Age", cmap = 'jet',
           colorbar = True, legend = True, figsize = (10, 7))
# Cambiar etiquetas de los ejes
ax.set_xlabel("Longitud")
ax.set_ylabel("Altura")


crab.hist(bins = 50, figsize = (10, 7))


import pandas as pd

# Suponiendo que tu DataFrame se llama 'df' y ya lo cargaste
# Por ejemplo: df = pd.read_csv('CrabAgePrediction.csv')
df=crab.copy()
# Crear una nueva columna para los 'grupos de peso' basada en los rangos de tu tabla
bins = [0, 4, 9, 14, 19, 24, 29, float('inf')]
labels = ['<4', '4-9', '9-14', '14-19', '19-24', '24-29', '>29']
df['Peso_Group'] = pd.cut(df['Shell Weight'] + df['Viscera Weight'] + df['Shucked Weight'], bins=bins, labels=labels, right=False)

# Crear una nueva columna para los 'grupos de edad'
age_bins = [0, 6, 8, 10, 12, 14, 16, float('inf')]
age_labels = ['<6', '6-7', '8-9', '10-11', '12-13', '14-15', '>16']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Crear la tabla de contingencia con frecuencias relativas (%)
relative_freq_table = pd.crosstab(
    index=df['Age_Group'], 
    columns=df['Peso_Group'], 
    normalize='index'  # Normaliza por fila para obtener los porcentajes
) * 100

# Añadir la columna de total general
relative_freq_table['Total general'] = relative_freq_table.sum(axis=1)

# Añadir la fila de total general
relative_freq_table.loc['Total general'] = relative_freq_table.sum()
relative_freq_table.loc['Total general', 'Total general'] = 100  # Asegurar que el total sea 100%

# Opcional: Redondear los valores a dos decimales para una mejor visualización
relative_freq_table = relative_freq_table.round(2)

print(relative_freq_table)

