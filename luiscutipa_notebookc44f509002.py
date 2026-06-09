import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

Healt_data = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv') 
#conocer que contiene el dataset.
print(Healt_data.head()) 
print()
print(Healt_data.info())
print()
# identificacion de las columnas con datos faltantes.
print(Healt_data.isnull().sum())
print()
# imputación de la mediana a los datos faltantes.
Healt_data['Work Pressure'] = Healt_data['Work Pressure'].fillna(Healt_data['Work Pressure'].median())
Healt_data['Job Satisfaction'] = Healt_data['Job Satisfaction'].fillna(Healt_data['Job Satisfaction'].median())
Healt_data['Financial Stress'] = Healt_data['Financial Stress'].fillna(Healt_data['Financial Stress'].median())
print()
# identificacion de las columnas con mediana con datos faltantes.
print(Healt_data.isnull().sum())
print()
# imputación de la moda a los datos faltantes.
Healt_data['Dietary Habits'] = Healt_data['Dietary Habits'].fillna(Healt_data['Dietary Habits'].mode()[0])
Healt_data['Degree'] = Healt_data['Degree'].fillna(Healt_data['Degree'].mode()[0])
print()
# identificacion de las columnas con moda con datos faltantes.
print(Healt_data.isnull().sum())
print()
#se eliminaran las columas que no tienen valor para llegar al objetivo: id, Name, city, profession, etc.

Healt_data_clean = Healt_data.drop(columns=['id', 'Name', 'City', 'Profession', 'Academic Pressure', 'CGPA', 'Study Satisfaction'])
print(Healt_data_clean.head()) 

# valores nulos
print(Healt_data_clean.isnull().sum())

#saber si esta balaceado la columna depression
print(Healt_data['Depression'].value_counts(normalize=True))
# [V_1: Distribución de clase 'Depression']
plt.figure(figsize=(6, 4))
sns.countplot(x='Depression', data=Healt_data_clean)
plt.title('¿Cuántas personas tienen depresión?')
plt.xlabel('0 = No / 1 = Sí')
plt.ylabel('Cantidad')
plt.grid(True)
plt.tight_layout()
plt.show() 

# Preparar X y Y
X = Healt_data_clean.drop(columns=['Depression'])
y = Healt_data_clean['Depression']

# Convertir variables categóricas
X_encoded = pd.get_dummies(X)
print()
print()
print(X_encoded.head(10))
print()
print()
# Separar datos
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# Modelo con Xgboost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Matriz de confusion para validación
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(xgb_model, X_val, y_val)
plt.title("Matriz de Confusión - Validación")
plt.show()
print()

# Evaluar y visualizar el modelo
plt.figure(figsize=(10, 10))
plot_importance(xgb_model, max_num_features=10, importance_type='gain', title='Variables relevantes')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

Healt_data_test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv') 
#conocer que contiene el dataset.
print(Healt_data_test.head()) 
print()
print(Healt_data_test.info())
print()
# identificacion de las columnas con datos faltantes.
print(Healt_data_test.isnull().sum())
print()
# imputación de la mediana a los datos faltantes.
Healt_data_test['Work Pressure'] = Healt_data_test['Work Pressure'].fillna(Healt_data_test['Work Pressure'].median())
Healt_data_test['Job Satisfaction'] = Healt_data_test['Job Satisfaction'].fillna(Healt_data_test['Job Satisfaction'].median())
Healt_data_test['Financial Stress'] = Healt_data_test['Financial Stress'].fillna(Healt_data_test['Financial Stress'].median())
print()
# identificacion de las columnas con mediana con datos faltantes.
print(Healt_data_test.isnull().sum())
print()
# imputación de la moda a los datos faltantes.
Healt_data_test['Dietary Habits'] = Healt_data_test['Dietary Habits'].fillna(Healt_data_test['Dietary Habits'].mode()[0])
Healt_data_test['Degree'] = Healt_data_test['Degree'].fillna(Healt_data_test['Degree'].mode()[0])
print()
# identificacion de las columnas con moda con datos faltantes.
print(Healt_data_test.isnull().sum())
print()
#se eliminaran las columas que no tienen valor para llegar al objetivo: id, Name, city y profession
Healt_data_clean_test = Healt_data_test.drop(columns=['id', 'Name', 'City', 'Profession', 'Academic Pressure', 'CGPA', 'Study Satisfaction'])
print(Healt_data_clean_test.head()) 
print()
# valores nulos
print(Healt_data_clean_test.isnull().sum())
print()
# Codificar variables categóricas con get_dummies
test_encoded = pd.get_dummies(Healt_data_clean_test)
# Alinear columnas con las del entrenamiento
test_encoded = test_encoded.reindex(columns=X_train.columns, fill_value=0)
# Predicción
y_test_pred = xgb_model.predict(test_encoded)
# Resultados
print()
resultado = pd.DataFrame({
    'Prediction': y_test_pred
})
print()
print(resultado.head(10))






