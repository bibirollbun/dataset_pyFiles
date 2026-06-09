# Instalamos versiones compatibles en Kaggle
!pip install -q scikit-learn==1.3.2 imbalanced-learn==0.11.0



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv('/kaggle/input/titanic-machine-learning-u-lima/test.csv')
df2 = pd.read_csv('/kaggle/input/titanic-machine-learning-u-lima/train.csv')


# .shape nos da la cantidad de filas y columnas
print("Conjunto de entrenamiento completo: ", "Filas:", df2.shape[0], ", Columnas: ", df2.shape[1])
print("Conjunto de prueba sin la columna 'Survived': ", "Filas:", df1.shape[0], ", Columnas: ", df1.shape[1])
print("Atributos: ", df1.columns.tolist())

# Cada fila representa a una persona. Hay 12 atributos originales por persona:
# 'PassengerId', 'Survived', 'Pclass',
# 'Name', 'Sex', 'Age', 'SibSp',
# 'Parch', 'Ticket', 'Fare',
# 'Cabin', 'Embarked'



# Eliminamos la columna 'PassengerId' si existe, si no, lo ignora
df1 = df1.drop("PassengerId", axis=1, errors='ignore')
df2 = df2.drop("PassengerId", axis=1, errors='ignore')




df = pd.concat([df1, df2], axis=0, ignore_index=True)
df.dropna(subset=['Survived'], inplace=True)
df = df.reset_index()
df = df.drop('index', axis=1)


df.head()


df.tail()


info = df.info()
print(info)

# El método info() es útil para obtener una descripción rápida de los datos.
# Nos muestra que tenemos 891 registros, el tipo de cada atributo
# y la cantidad de valores que no son nulos.



# Podemos ver un resumen de los ATRIBUTOS NUMÉRICOS
describe = df.describe()
print(describe)

# NOTA:
# => EDAD:
# - La edad promedio de los pasajeros es de 29 años.

# => TARIFA (Fare):
# - El precio promedio del boleto es de $32. Hay una desviación estándar muy alta.
# - Tenemos boletos con precio $0, lo cual no tiene sentido.
# - Exploraremos esto más adelante en el cuaderno.

# => CLASE (Pclass):
# - La clase promedio de los pasajeros es 2ª clase.



# Podemos ver la cantidad de valores únicos para cada columna, 
# esto es especialmente útil para los ATRIBUTOS CATEGÓRICOS en el conjunto de datos. 
# En otras palabras, vemos qué categorías existen y cuántos registros pertenecen a cada una:

for i in df.columns:
    print("//////////////////////////////////////////////////////\n")
    print(i, " columna:\n", df[i].value_counts())
    print("\n//////////////////////////////////////////////////////\n")



# Visualizando los datos faltantes
sns.heatmap(df.isnull(), cbar=False)

# Revisamos los valores faltantes en todos los atributos
for i in df.columns:
    missing_values = df[i].isna()
    print(f'{i} tiene {missing_values.sum()} valores faltantes')

# NOTA:
# Cabin tiene 687 valores faltantes.
# Age tiene 177 valores faltantes.
# Embarked tiene 2 valores faltantes.



# Podemos ver la etiqueta de supervivencia [0 = 'No', 1 = 'Sí']
print("Porcentaje de sobrevivientes: ", round(549/891 * 100, 2), "%")
print("Porcentaje de no sobrevivientes: ", round(342/891 * 100, 2), "%")

plt.figure(figsize=(3,3))
sns.countplot(x=df['Survived'])
plt.show()



# Podemos ver la proporción de hombres y mujeres
print("Porcentaje de hombres (", (468+109), "): ", round(577/891 * 100, 2), "%")
print("Porcentaje de mujeres (", (81+233), "): ", round(314/891 * 100, 2), "%")
print("\n")

# Podemos ver la proporción de supervivencia entre hombres y mujeres
grouped = df.groupby(['Sex', 'Survived']).size().unstack()

print("Porcentaje de hombres que sobrevivieron: ", round(grouped[1][1]/(468+109) * 100, 2), "%")
print("Porcentaje de hombres que NO sobrevivieron: ", round(grouped[0][1]/(468+109) * 100, 2), "%")
print("\n")
print("Porcentaje de mujeres que sobrevivieron: ", round(grouped[1][0]/314 * 100, 2), "%")
print("Porcentaje de mujeres que NO sobrevivieron: ", round(grouped[0][0]/314 * 100, 2), "%")

# Crear 2 gráficos en la misma fila
fig, axs = plt.subplots(1, 2, figsize=(8, 4))  

# [1] Primer gráfico: conteo de hombres y mujeres
sns.countplot(x=df['Sex'], ax=axs[0])
axs[0].set_title("Distribución por sexo")
axs[0].set_xlabel("Sexo")
axs[0].set_ylabel("Cantidad")

# [2] Segundo gráfico: supervivencia por sexo
grouped.plot(kind='bar', stacked=True, ax=axs[1])
axs[1].set_xlabel("Sexo")
axs[1].set_ylabel("Cantidad")
axs[1].set_title("Supervivencia por sexo")
axs[1].legend(title='Supervivencia', labels=['No sobrevivió', 'Sobrevivió'])
axs[1].tick_params(axis='x', rotation=0)
axs[1].grid(True)

plt.tight_layout()  # Ajustar el espacio entre subgráficos
plt.show()



# Podemos ver la proporción de edades por grupo o categoría de edad
# Creamos categorías de edad para agrupar a los pasajeros por rangos de edad. 
# Esto puede ayudarnos a capturar el impacto de la edad en la supervivencia.
bins = [0, 18, 30, 50, 100]
labels = ["menores", "jóvenes adultos", "adultos", "ancianos"]
df["Categoria_Edad"] = pd.cut(df["Age"], bins=bins, labels=labels)

print("Menores (", (69+70), "):\t0 - 18 años\tNo sobrevivió: ", round((69)/(69+70)*100, 2), "%\t\tSobrevivió:", round((70)/(69+70)*100, 2), "%")
print("Jóvenes adultos (", (174+96), "):\t19 - 30 años\tNo sobrevivió: ", round((174)/(174+96)*100, 2), "%\t\tSobrevivió:", round((96)/(174+96)*100, 2), "%")
print("Adultos (", (139+102), "):\t\t30 - 50 años\tNo sobrevivió: ", round((139)/(139+102)*100, 2), "%\t\tSobrevivió:", round((102)/(139+102)*100, 2), "%")
print("Ancianos (", (42+22), "):\t\t50+ años\tNo sobrevivió: ", round((42)/(42+22)*100, 2), "%\t\tSobrevivió:", round((22)/(42+22)*100, 2), "%")
print("\n")

# Ahora agrupamos por Supervivencia y Categoría de Edad
grouped = df.groupby(['Categoria_Edad', 'Survived']).size().unstack()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Dos gráficos en la misma fila

# [1] Primer gráfico: distribución por categoría de edad
sns.countplot(x=df['Categoria_Edad'], ax=axs[0])
axs[0].set_title("Distribución por categoría de edad")
axs[0].set_xlabel("Categoría de edad")
axs[0].set_ylabel("Cantidad")

# [2] Segundo gráfico: supervivencia por categoría de edad
grouped.plot(kind='bar', stacked=True, ax=axs[1])
axs[1].set_xlabel("Categoría de edad")
axs[1].set_ylabel("Cantidad")
axs[1].set_title("Supervivencia por categoría de edad")
axs[1].legend(title='Supervivencia', labels=['No sobrevivió', 'Sobrevivió'])
axs[1].tick_params(axis='x', rotation=0)
axs[1].grid(True)

plt.tight_layout()  # Ajustar espacio entre gráficos para evitar solapamiento
plt.show()



# Podemos ver la proporción de pasajeros por clase
print("Porcentaje de personas en 1ª clase (", 184, "): ", round(184/891 * 100, 2), "%\tNo sobrevivió: ", round((80)/(80+136)*100, 2), "%\t\tSobrevivió:", round((136)/(80+136)*100, 2), "%")
print("Porcentaje de personas en 2ª clase (", 216, "): ", round(216/891 * 100, 2), "%\tNo sobrevivió: ", round((97)/(97+87)*100, 2), "%\t\tSobrevivió:", round((87)/(97+87)*100, 2), "%")
print("Porcentaje de personas en 3ª clase (", 491, "): ", round(491/891 * 100, 2), "%\tNo sobrevivió: ", round((372)/(372+119)*100, 2), "%\t\tSobrevivió:", round((119)/(372+119)*100, 2), "%")
print("\n")

# Ahora agrupamos por Supervivencia y Clase
grouped = df.groupby(['Pclass', 'Survived']).size().unstack()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Dos gráficos en la misma fila

# [1] Primer gráfico: distribución de pasajeros por clase
sns.countplot(x=df['Pclass'], ax=axs[0])
axs[0].set_title("Distribución por clase")
axs[0].set_xlabel("Clase de boleto")
axs[0].set_ylabel("Cantidad")

# [2] Segundo gráfico: supervivencia por clase
grouped.plot(kind='bar', stacked=True, ax=axs[1])
axs[1].set_xlabel("Clase de boleto")
axs[1].set_ylabel("Cantidad")
axs[1].set_title("Supervivencia por clase")
axs[1].legend(title='Supervivencia', labels=['No sobrevivió', 'Sobrevivió'])
axs[1].tick_params(axis='x', rotation=0)
axs[1].grid(True)

plt.tight_layout()  # Ajustar espacio entre subgráficos
plt.show()




# Podemos ver la proporción de pasajeros según el puerto de embarque
print("Porcentaje de embarque en C = Cherburgo (", (75+93), "): ", round((75+93)/891 * 100, 2), "%\t\tNo sobrevivió: ", round((75)/(75+93)*100, 2), "%\t\tSobrevivió:", round((93)/(75+93)*100, 2), "%")
print("Porcentaje de embarque en Q = Queenstown (", (47+30), "): ", round((47+30)/891 * 100, 2), "%\t\tNo sobrevivió: ", round((47)/(47+30)*100, 2), "%\t\tSobrevivió:", round((30)/(47+30)*100, 2), "%")
print("Porcentaje de embarque en S = Southampton (", (427+217), "): ", round((427+217)/891 * 100, 2), "%\tNo sobrevivió: ", round((427)/(427+217)*100, 2), "%\t\tSobrevivió:", round((217)/(427+217)*100, 2), "%")
print("\n")

# Ahora agrupamos por Supervivencia y Puerto de embarque
grouped = df.groupby(['Embarked', 'Survived']).size().unstack()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Dos gráficos en la misma fila

# [1] Primer gráfico: distribución de pasajeros por puerto de embarque
sns.countplot(x=df["Embarked"], ax=axs[0])
axs[0].set_title("Distribución por puerto de embarque")
axs[0].set_xlabel("Puerto de embarque")
axs[0].set_ylabel("Cantidad")

# [2] Segundo gráfico: supervivencia por puerto de embarque
grouped.plot(kind='bar', stacked=True, ax=axs[1])
axs[1].set_xlabel("Puerto de embarque")
axs[1].set_ylabel("Cantidad")
axs[1].set_title("Supervivencia por puerto de embarque")
axs[1].legend(title='Supervivencia', labels=['No sobrevivió', 'Sobrevivió'])
axs[1].tick_params(axis='x', rotation=0)
axs[1].grid(True)

plt.tight_layout()  # Ajustar espacio entre subgráficos
plt.show()



# Para tener una idea del tipo de datos con los que estamos trabajando,
# graficamos un histograma para cada uno de los ATRIBUTOS NUMÉRICOS.
# Un histograma nos muestra el número de instancias (en el eje vertical)
# que tienen un valor dentro de un rango determinado (eje horizontal).

df.hist(bins=50, figsize=(12,8))
plt.show()

# NOTAS:
# Los histogramas de 'Fare', 'Age', 'SibSp' y 'Parch' están sesgados hacia la izquierda
# (tienen colas largas): se extienden mucho más hacia la izquierda de la mediana que hacia la derecha.
# Esto puede dificultar que algunos modelos de Machine Learning detecten patrones.
# Posteriormente transformaremos estos atributos para que tengan una distribución
# más simétrica y con forma de campana (normal).



# Búsqueda de valores atípicos (outliers).
# Vamos a buscarlos de dos formas:
# 1. Con diagramas de dispersión (scatter plot) entre pares de atributos [A, B].
# 2. Usando diagramas de caja (boxplot) de la librería Seaborn.
# Los boxplots muestran claramente dónde existen valores atípicos en las columnas.

# AGE & FARE
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Boxplot para la variable Age
age_box_plot = sns.boxplot(x=df['Age'], ax=axs[0])
age_box_plot.set_title('Valores atípicos en Age')

# Boxplot para la variable Fare
fare_box_plot = sns.boxplot(x=df['Fare'], ax=axs[1])
fare_box_plot.set_title('Valores atípicos en Fare')

plt.tight_layout()
plt.show()



# Análisis de 'Fare' y 'Pclass'
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# [1] Diagrama de dispersión (scatter plot): primer gráfico
scatter_plot = df.plot(kind="scatter", x="Pclass", y="Fare", grid=True,
                             alpha=0.2, s=df["Fare"] / 2, label="Fare",
                             legend=True, sharex=False, colorbar=True,
                             cmap="jet", c="Fare", ax=axs[0])
scatter_plot.set_yticks(range(0, int(df["Fare"].max()) + 10, 20))
scatter_plot.set_xticks(range(1, 4))
scatter_plot.set_title('Relación entre Pclass y Fare')

# [2] Diagrama de caja (boxplot): segundo gráfico
box_plot = sns.boxplot(x=df['Fare'], ax=axs[1])
box_plot.set_title('Valores atípicos en Fare')
box_plot.set_xlabel('Fare')

plt.tight_layout()  # Ajustar espacios entre subgráficos
plt.show()

# OBSERVACIONES:
# A primera vista, se observa que los precios de 'Fare' en 2ª y 3ª clase son muy similares.
# En 1ª clase los precios son mucho más variados.
# Algunos boletos de 1ª clase se vendieron al mismo precio que los de 2ª y 3ª clase,
# pero también hubo boletos de 1ª clase con precios mucho más altos.



# Embarked <===> Fare

# Codificación del puerto de embarque:
# C = Cherbourg, Q = Queenstown, S = Southampton
# Se codifican como: 1 = Cherbourg, 2 = Queenstown, 3 = Southampton
df_copy = df.copy()
df_copy['Embarked'] = df_copy['Embarked'].replace(['C', 'Q', 'S'], [1, 2, 3])

# Análisis de 'Embarked' y 'Fare'
# [1] Diagrama de dispersión (scatter plot)
scatter_plot = df_copy.plot(kind="scatter", x="Embarked", y="Fare", grid=True,
                             alpha=0.2, s=df_copy["Fare"] / 2, label="Fare",
                             legend=True, sharex=False, colorbar=True,
                             cmap="jet", c="Fare")

scatter_plot.set_yticks(range(0, int(df_copy["Fare"].max()) + 10, 20))
scatter_plot.set_xticks(range(1, 4), labels=['Cherbourg', 'Queenstown', 'Southampton'])
scatter_plot.set_title('Relación entre Puerto de Embarque y Fare')

plt.show()

# OBSERVACIONES:
# - Se aprecia que los pasajeros embarcados en Cherbourg tienden a tener tarifas más altas.
# - Los pasajeros de Queenstown generalmente pagaron tarifas más bajas.
# - En Southampton hay una mayor variedad de tarifas, pero predominan las de costo bajo y medio.



# Sex <===> Fare

# Codificación del sexo:
# 'male' = Hombre, 'female' = Mujer
# Se codifican como: 1 = Hombre, 0 = Mujer
df['Sex'] = df['Sex'].replace(['male', 'female'], [1, 0])

# Análisis de 'Sex' y 'Fare'
# [1] Diagrama de dispersión (scatter plot)
scatter_plot = df.plot(kind="scatter", x="Sex", y="Fare", grid=True,
                             alpha=0.2, s=df["Fare"] / 2, label="Fare",
                             legend=True, sharex=False, colorbar=True,
                             cmap="jet", c="Fare")

scatter_plot.set_yticks(range(0, int(df["Fare"].max()) + 10, 20))
scatter_plot.set_xticks(range(0, 2), labels=['Hombre', 'Mujer'])
scatter_plot.set_title('Relación entre Sexo y Fare')

plt.show()

# OBSERVACIONES:
# - A primera vista, las mujeres parecen haber pagado tarifas más altas en promedio.
# - Los hombres presentan más casos en la franja de tarifas bajas.
# - Esto podría estar relacionado con la clase del boleto, ya que muchas pasajeras mujeres 
#   viajaban en primera clase (lo que eleva el costo del pasaje).



df.head()


# Función que calcula los límites de los outliers usando el rango intercuartílico (IQR)
def outlier_thresholds(df, col_name):
    # Se calculan los estadísticos descriptivos de la columna
    data_qtles = df.describe()
    
    # Primer cuartil (Q1 = 25%)
    q1 = data_qtles[col_name]['25%']
    
    # Tercer cuartil (Q3 = 75%)
    q3 = data_qtles[col_name]['75%']
    
    # Rango intercuartílico (IQR = Q3 - Q1)
    IQR = q3 - q1
    
    # Límite superior: Q3 + 1.5*IQR
    up_limit = q3 + 1.5 * IQR
    
    # Límite inferior: Q1 - 1.5*IQR
    low_limit = q1 - 1.5 * IQR
    
    # Se devuelven los límites inferior y superior
    return low_limit, up_limit



# Imprimir los límites (umbral inferior y superior) de outliers
# para todas las columnas seleccionadas del dataset

# Seleccionamos las columnas donde queremos buscar outliers
columns = ['Age', 'Fare']

# Recorremos cada columna y calculamos sus límites con la función definida antes
for i in columns:
    low, up = outlier_thresholds(df, i)
    print(f"Umbrales en la columna {i}: Límite inferior = {low}, Límite superior = {up}")



# Calcular los límites inferior y superior de la columna "Age"
low, up = outlier_thresholds(df, "Age")

'''
Usamos los límites obtenidos para filtrar el DataFrame,
dejando solo las filas donde la columna "Age" esté dentro de los límites.
De esta manera eliminamos los outliers en esa columna.
'''

# Filtrar las filas que estén dentro de los límites
df = df[(df['Age'] >= low) & (df['Age'] <= up)]



# Calcular los límites inferior y superior de la columna "Fare"
low, up = outlier_thresholds(df, "Fare")

# Filtrar las filas que estén dentro de los límites
# Esto elimina los outliers en la columna "Fare"
df = df[(df['Fare'] >= low) & (df['Fare'] <= up)]



# Mostrar el DataFrame ya sin outliers en las columnas 'Fare' y 'Age'
df



# Visualizando nuevamente los boxplots de seaborn para verificar que ya no hay outliers
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Boxplot de Age
age_box_plot = sns.boxplot(x=df['Age'], ax=axs[0])
age_box_plot.set_title('Age (sin outliers)')

# Boxplot de Fare
fare_box_plot = sns.boxplot(x=df['Fare'], ax=axs[1])
fare_box_plot.set_title('Fare (sin outliers)')

plt.tight_layout()
plt.show()



# Eliminamos las variables categóricas que no aportan a la correlación (solo se trabaja con atributos numéricos)
columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]

for column in columns_to_drop:
    if column in df.columns:   # Verificamos que la columna exista
        df.drop(columns=column, axis=1, inplace=True)

# Mostramos las primeras filas para confirmar que se eliminaron
df.head()



# Separar las variables predictoras (X) de la variable objetivo (y)
X = df.drop('Survived', axis=1)   # Todas las columnas menos 'Survived'
y = df['Survived']                # Columna objetivo



# Imprimir los atributos numéricos y categóricos del DataFrame
atributos = {'numéricos': [], 'categóricos': []}

for columna in df.columns:
    if pd.api.types.is_numeric_dtype(df[columna]):
        atributos['numéricos'].append(columna)
    else:
        atributos['categóricos'].append(columna)

print("Atributos Numéricos:")
print(atributos['numéricos'])

print("Atributos Categóricos:")
print(atributos['categóricos'])



df


# Matriz de correlación entre las variables numéricas
correlation_matrix = df.corr(numeric_only=True)

# Mostrar en consola la matriz
print(correlation_matrix)



# Visualización de la matriz de correlación con un mapa de calor
plt.figure(figsize=(8, 6))  # Definir el tamaño de la figura
sns.heatmap(
    correlation_matrix,      # Matriz de correlación calculada
    annot=True,              # Mostrar los valores dentro de cada celda
    cmap='coolwarm',         # Paleta de colores (rojo a azul)
    vmin=-1, vmax=1          # Escala de colores de -1 a 1
)
plt.title("Mapa de calor de correlaciones", fontsize=14)
plt.show()



# Nueva característica: tamaño de la familia
# Se construye sumando:
# - 'SibSp': número de hermanos/as o cónyuge a bordo
# - 'Parch': número de padres/madres o hijos/as a bordo
# Esto nos da el tamaño de la familia con la que viajaba cada pasajero.
df['family_size'] = df['SibSp'] + df['Parch']



# Crear categorías de tarifas ('Fare Categories') a partir de la columna 'Fare'.
# Usamos 'pd.cut' para dividir el rango de tarifas en intervalos (bins) con etiquetas (labels).

# Definimos los rangos de tarifas (en dólares):
bins = [0, 10, 30, 100, 250, 550]

# Definimos las etiquetas para cada rango:
labels = ["very_low_fare", "low_fare", "medium_fare", "high_fare", "very_high_fare"]

# Creamos una nueva columna 'fare_category' con las categorías asignadas
df["fare_category"] = pd.cut(df["Fare"], bins=bins, labels=labels)

# Imprimimos cuántas personas caen en cada categoría usando value_counts()
print("Distribución de pasajeros según categoría de tarifa:")
print(df["fare_category"].value_counts())



# Gráfico de barras para visualizar la distribución de las categorías de tarifa
plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura

# sns.countplot muestra la frecuencia de cada categoría en 'fare_category'
sns.countplot(
    x="fare_category",      # Eje X: categorías de tarifa
    data=df,                # Dataset
    order=labels,           # Asegura que se muestren en el mismo orden que definimos antes
    palette="viridis"       # Paleta de colores
)

# Etiquetas de los ejes
plt.xlabel("Fare Category")
plt.ylabel("Número de pasajeros")

# Rotar las etiquetas del eje X para mejor visibilidad
plt.xticks(rotation=45)

# Mostrar gráfico
plt.show()



# Reducimos las categorías de tarifas a solo 3 grupos
# Esto ayuda a que los datos no estén tan sesgados a la izquierda
# y que el modelo pueda trabajar con una mejor distribución.

bins = [0, 10, 30, 1000]   # Definimos los intervalos de las tarifas
labels = ["L", "M", "H"]   # L = Low, M = Medium, H = High

# Crear nueva columna con las categorías de tarifa
df["fare_category"] = pd.cut(df["Fare"], bins=bins, labels=labels)



# Creamos una variable binaria para identificar si el pasajero viaja solo o no.
# travel_alone = 1 si family_size == 0 (viaja solo)
# travel_alone = 0 si family_size > 0 (viaja acompañado)

df["travel_alone"] = np.where(df["family_size"] == 0, 1, 0)

# Estadísticas de control (en este caso son valores fijos de referencia del dataset Titanic)
print("Count of people Traveling Alone: ", 354)
print("Count of people Not-Traveling Alone: ", 537)



# Agrupamos por "viajar solo" y "sobrevivió"
grouped = df.groupby(['travel_alone', 'Survived']).size().unstack()

# Gráfico de barras apiladas
grouped.plot(kind='bar', stacked=True, figsize=(6, 4))
plt.xlabel('Travel Alone [0=No, 1=Yes]')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.xticks(rotation=0)  # Mantener etiquetas horizontales
plt.grid(True)
plt.show()



# Mostrar las últimas filas con las nuevas variables creadas
df.tail()



from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Pipeline para procesar variables numéricas
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])



from sklearn.preprocessing import OneHotEncoder

# Pipeline para procesar variables categóricas
cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])



# from sklearn.preprocessing import StandardScaler
# X = df.drop('Survived', axis = 1)
# y = df['Survived']
# scaler = StandardScaler()

# Escalando los datos de la tarifa (Fare) para que funcionen con 
# K-Vecinos, Máquinas de Vectores de Soporte (SVM) y Regresión Logística.
# feature = df['Fare']
# scaled_feature = scaler.fit_transform(feature.values.reshape(-1,1))
# df['Fare'] = scaled_feature

# Escalando los datos de la edad (Age) para que funcionen con 
# K-Vecinos, Máquinas de Vectores de Soporte (SVM) y Regresión Logística.
# feature2 = df['Age']
# scaled_feature2 = scaler.fit_transform(feature2.values.reshape(-1,1))
# df['Age'] = scaled_feature2

# Para hacer predicciones necesitamos transformar de vuelta los valores escalados:
# original_feature = scaler.inverse_transform(scaled_feature)



class_counts = df['Survived'].value_counts()
print(class_counts)

