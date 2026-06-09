# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_data.head()


sample_submission.head()


print("Train Data Info:")
print(train_data.info())

print("\nTest Data Info:")
print(test_data.info())

print("\nTraining Extra Info:")
print(train_extra.info())


print("\nMissing values in Train Data:")
print(train_data.isnull().sum())

print("\nMissing values in Test Data:")
print(test_data.isnull().sum())

print("\nMissing values in Training Extra:")
print(train_extra.isnull().sum())


def missing_percentage(df, name):
    missing = df.isnull().sum()
    total = len(df)
    percent_missing = (missing / total) * 100
    missing_df = pd.DataFrame({'Column': df.columns, 'Missing Values': missing, 'Percentage (%)': percent_missing})
    missing_df = missing_df[missing_df["Missing Values"] > 0]  # Filtrar solo columnas con nulos
    print(f"\nMissing Values in {name}:")
    print(missing_df.sort_values(by="Percentage (%)", ascending=False))

# Aplicamos la función a cada dataset
missing_percentage(train_data, "Train Data")
missing_percentage(test_data, "Test Data")
missing_percentage(train_extra, "Training Extra")



# Rellenamos valores nulos con "Desconocido"
categorical_cols = ["Brand", "Material", "Size", "Style", "Color"]
for col in categorical_cols:
    train_data[col] = train_data[col].fillna("Desconocido")
    test_data[col] = test_data[col].fillna("Desconocido")


train_data["Weight Capacity (kg)"] = train_data["Weight Capacity (kg)"].fillna(train_data["Weight Capacity (kg)"].median())
test_data["Weight Capacity (kg)"] = test_data["Weight Capacity (kg)"].fillna(test_data["Weight Capacity (kg)"].median())



# Rellenar valores nulos en booleanas con "Desconocido"
bool_cols = ["Laptop Compartment", "Waterproof"]
for col in bool_cols:
    train_data[col] = train_data[col].fillna("Desconocido")
    test_data[col] = test_data[col].fillna("Desconocido")



print("\nMissing values in Train Data:")
print(train_data.isnull().sum())

print("\nMissing values in Test Data:")
print(test_data.isnull().sum())

print("\nMissing values in Training Extra:")
print(train_extra.isnull().sum())


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(train_data["Price"], bins=10, kde=True, color='blue')
plt.title("Price Distribution")
plt.xlabel("Price ($)")

plt.subplot(1, 3, 2)
sns.histplot(train_data["Compartments"], bins=10, kde=True, color='green')
plt.title("Compartments Distribution")
plt.xlabel("Number of Compartments")

plt.subplot(1, 3, 3)
sns.histplot(train_data["Weight Capacity (kg)"], bins=10, kde=True, color='red')
plt.title("Weight Capacity Distribution")
plt.xlabel("Weight Capacity (kg)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=train_data["Price"], color='blue')
plt.title("Boxplot of Price")

plt.subplot(1, 3, 2)
sns.boxplot(x=train_data["Compartments"], color='green')
plt.title("Boxplot of Compartments")

plt.subplot(1, 3, 3)
sns.boxplot(x=train_data["Weight Capacity (kg)"], color='red')
plt.title("Boxplot of Weight Capacity")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(x='Brand', data=train_data, palette='viridis')
plt.title('Brand Distribution')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Material', data=train_data, palette='Set2')
plt.title('Material Distribution')
plt.xticks(rotation=45)
plt.show()


# Copiamos el dataset para no modificar el original
train_encoded = train_data.copy()

# Columnas categóricas a convertir
categorical_cols = ["Brand", "Material", "Size", "Style", "Color"]

# Aplicamos Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    train_encoded[col] = le.fit_transform(train_encoded[col])


train_encoded["Laptop Compartment"] = train_encoded["Laptop Compartment"].map({"Yes": 1, "No": 0})
train_encoded["Waterproof"] = train_encoded["Waterproof"].map({"Yes": 1, "No": 0})


print("\nDuplicated Rows in Train Data:", train_encoded.duplicated().sum())
print("Duplicated Rows in Test Data:", test_data.duplicated().sum())
print("Duplicated Rows in Training Extra:", train_extra.duplicated().sum())


# Histograma
plt.figure(figsize=(8, 5))
sns.histplot(train_encoded["Price"], bins=50, kde=True)
plt.title("Distribución de Precios")
plt.xlabel("Price")
plt.ylabel("Frecuencia")
plt.show()


# Matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(train_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()


# Función para detectar y eliminar outliers
def detect_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers in {col}: {len(outliers)}")
        
# Detectamos outliers en columnas numéricas
numeric_cols = ["Compartments", "Weight Capacity (kg)", "Price"]
detect_outliers(train_encoded, numeric_cols)



# Creamos un boxplot para Price vs Compartments
plt.figure(figsize=(10, 6))

# Boxplot de Precio vs Compartimentos
sns.boxplot(x=train_encoded["Compartments"], y=train_encoded["Price"], color="lightblue")

# Calcular Q1, Q3 y la Media
Q1 = train_encoded["Price"].quantile(0.25)
Q3 = train_encoded["Price"].quantile(0.75)
median = train_encoded["Price"].median()
mean = train_encoded["Price"].mean()

# Añadir las líneas para los cuartiles y la media
plt.axhline(y=Q1, color='r', linestyle='--', label=f'Q1: {Q1:.2f}')
plt.axhline(y=Q3, color='r', linestyle='--', label=f'Q3: {Q3:.2f}')
plt.axhline(y=median, color='g', linestyle='-', label=f'Median: {median:.2f}')
plt.axhline(y=mean, color='b', linestyle='-', label=f'Mean: {mean:.2f}')

# Añadir título y etiquetas
plt.title('Boxplot of Price vs Compartments')
plt.xlabel('Compartments')
plt.ylabel('Price')

# Añadir leyenda
plt.legend()

# Mostrar gráfico
plt.show()



# Verifica si hay valores nulos en las columnas
print(train_encoded[['Weight Capacity (kg)', 'Price']].isnull().sum())

# Elimina las filas con valores nulos
train_encoded = train_encoded.dropna(subset=['Weight Capacity (kg)', 'Price'])



# Verifica los tipos de datos
print(train_encoded[['Weight Capacity (kg)', 'Price']].dtypes)

# Convierte las columnas a numéricas si es necesario
train_encoded['Weight Capacity (kg)'] = pd.to_numeric(train_encoded['Weight Capacity (kg)'], errors='coerce')
train_encoded['Price'] = pd.to_numeric(train_encoded['Price'], errors='coerce')





