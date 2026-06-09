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





import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

train.head()


#informaciÃ³n general
train.info()


# estadistica basica
train.describe()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="viridis", rc={"figure.figsize":(12,6)})


# Diccionario para traducir columnas al espaÃ±ol
traduccion = {
    'age': 'Edad',
    'alcohol_consumption_per_week': 'Consumo de alcohol semanal',
    'physical_activity_minutes_per_week': 'Minutos de actividad fÃ­sica/semana',
    'diet_score': 'Puntaje de dieta',
    'sleep_hours_per_day': 'Horas de sueÃ±o/dÃ­a',
    'screen_time_hours_per_day': 'Horas frente a pantallas/dÃ­a',
    'bmi': 'Ã�ndice de masa corporal (IMC)',
    'waist_to_hip_ratio': 'RelaciÃ³n cintura-cadera',
    'systolic_bp': 'PresiÃ³n sistÃ³lica',
    'diastolic_bp': 'PresiÃ³n diastÃ³lica',
    'heart_rate': 'Frecuencia cardÃ­aca',
    'cholesterol_total': 'Colesterol total',
    'hdl_cholesterol': 'Colesterol HDL (bueno)',
    'ldl_cholesterol': 'Colesterol LDL (malo)',
    'triglycerides': 'TriglicÃ©ridos',
    'family_history_diabetes': 'Historial familiar de diabetes',
    'hypertension_history': 'Historial de hipertensiÃ³n',
    'cardiovascular_history': 'Historial cardiovascular',
    'diagnosed_diabetes': 'Diabetes diagnosticada'
}





# SelecciÃ³n de columnas numÃ©ricas (excepto id)
num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(['id'])

# Crear copia traduciendo nombres
train_es = train[num_cols].rename(columns=traduccion)

plt.figure(figsize=(20,20))
train_es.hist(
    bins=30,
    figsize=(20,20),
    color='teal',
    edgecolor='black'
)

plt.suptitle("DistribuciÃ³n de Variables NumÃ©ricas", fontsize=22, y=1.02)

plt.figtext(
    0.5, 
    0.94,
    "Cada histograma muestra la distribuciÃ³n de una variable numÃ©rica.\n"
    "Este anÃ¡lisis permite identificar patrones de salud, valores extremos y variabilidad.",
    ha="center",
    fontsize=14
)

plt.tight_layout()
plt.show()









# SelecciÃ³n de columnas numÃ©ricas (sin id)
num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(['id'])

# Matriz de correlaciÃ³n
corr = train[num_cols].corr()

plt.figure(figsize=(18,14))
sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=False,
    linewidths=0.5
)
plt.title("Mapa de calor de correlaciones entre variables numÃ©ricas", fontsize=18)
plt.show()



target_corr = corr['diagnosed_diabetes'].abs().sort_values(ascending=False)
target_corr



# ======================================
# CORRELACIÃ“N ENTRE VARIABLES CATEGÃ“RICAS (CRAMÃ‰R'S V)
# ======================================


from scipy.stats import chi2_contingency

# --------------------------------------
# 1. Seleccionar variables categÃ³ricas
# --------------------------------------
cat_cols = train.select_dtypes(include=['object']).columns

print("Variables categÃ³ricas detectadas:\n", cat_cols.tolist())


# --------------------------------------
# 2. FunciÃ³n para calcular CramÃ©râ€™s V
# --------------------------------------
def cramers_v(confusion_matrix):
    """
    Calcula CramÃ©r's V a partir de una tabla cruzada.
    Devuelve un valor entre 0 y 1 indicando la fuerza de la asociaciÃ³n.
    """
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]  # estadÃ­stico chi-cuadrado
        n = confusion_matrix.sum().sum()              # nÃºmero de observaciones
        r, k = confusion_matrix.shape                 # dimensiones
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))
    except:
        # Si algo falla (por ejemplo, una categorÃ­a vacÃ­a)
        return np.nan


# --------------------------------------
# 3. Crear matriz de correlaciones
# --------------------------------------
cramer_matrix = pd.DataFrame(
    np.zeros((len(cat_cols), len(cat_cols))),
    index=cat_cols,
    columns=cat_cols
)

# Confirmamos creaciÃ³n (para evitar NameError)
print("\nMatriz de correlaciones creada con tamaÃ±o:",
      cramer_matrix.shape)


# --------------------------------------
# 4. Calcular CramÃ©r's V para cada par de variables
# --------------------------------------
for col1 in cat_cols:
    for col2 in cat_cols:
        tabla = pd.crosstab(train[col1], train[col2])
        cramer_matrix.loc[col1, col2] = cramers_v(tabla)


# --------------------------------------
# 5. Visualizar en un heatmap
# --------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    cramer_matrix,
    annot=True,      # muestra valores
    fmt=".2f",       # formato de decimales
    cmap="coolwarm", # paleta de colores
    linewidths=0.5
)

plt.title("Matriz de CorrelaciÃ³n CategÃ³rica (CramÃ©r's V)", fontsize=16)
plt.xlabel("Variables categÃ³ricas")
plt.ylabel("Variables categÃ³ricas")
plt.show()



# vamos a contar cuÃ¡ntos casos hay en cada clase
train['diagnosed_diabetes'].value_counts()


sns.countplot(data=train, x='diagnosed_diabetes')
plt.title('DistribuciÃ³n de la variable objetivo (diabetes)')
plt.show()


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

df = train.copy()

# Codificamos categÃ³ricas
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

mi_scores.head(20)



from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Copia del dataset
df = train.copy()

# Codificar categÃ³ricas
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# SelecciÃ³n de muestra para acelerar el cÃ¡lculo
df_sample = df.sample(80000, random_state=42)

X = df_sample.drop('diagnosed_diabetes', axis=1)
y = df_sample['diagnosed_diabetes']

# Entrenar el modelo
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X, y)

# Importancia de features
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

importances_sorted.head(20)



# Nos quedamos con el Top 20
top_n = 20
feat_importances = importances_sorted.head(top_n)

plt.figure(figsize=(10, 6))

# GrÃ¡fico de barras horizontal para mejor legibilidad
plt.barh(feat_importances.index, feat_importances.values)

plt.gca().invert_yaxis()  # Para que la mÃ¡s importante quede arriba
plt.title('Top 20 caracterÃ­sticas mÃ¡s importantes segÃºn Random Forest')
plt.xlabel('Importancia')
plt.ylabel('CaracterÃ­sticas')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 1. Definimos la variable objetivo
target = 'diagnosed_diabetes'

# 2. Lista de caracterÃ­sticas seleccionadas (sin 'id')
selected_features = [
    'physical_activity_minutes_per_week',
    'age',
    'family_history_diabetes',
    'bmi',
    'triglycerides',
    'systolic_bp',
    'cholesterol_total',
    'ldl_cholesterol',
    'waist_to_hip_ratio',
    'hdl_cholesterol',
    'diet_score',
    'screen_time_hours_per_day',
    'sleep_hours_per_day',
    'heart_rate',
    'diastolic_bp',
    'smoking_status',
    'ethnicity',
    'education_level',
    'income_level',
    'gender'
]

# 3. Creamos X e y
X = train[selected_features].copy()
y = train[target]

# 4. Identificamos variables numÃ©ricas y categÃ³ricas dentro de X
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Variables numÃ©ricas:", num_features)
print("Variables categÃ³ricas:", cat_features)

# 5. Dividimos en entrenamiento y prueba con estratificaciÃ³n
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Definimos los transformadores para numÃ©ricas y categÃ³ricas
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# 7. Creamos el ColumnTransformer para aplicar el preprocesado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# (Opcional) 8. Definimos un pipeline base con solo el preprocesador
# MÃ¡s adelante aÃ±adiremos el modelo (LogisticRegression, RandomForest, etc.)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Ajustamos el preprocesador usando solo los datos de entrenamiento
baseline_pipeline.fit(X_train)

print("âœ… Preprocesado completado: escalado numÃ©rico y one-hot en categÃ³ricas.")



from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline completo: preprocesado + modelo
logreg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=500,
        solver='liblinear'  # robusto para datasets grandes
    ))
])

# Entrenamos
logreg_model.fit(X_train, y_train)

print("âœ… Modelo de RegresiÃ³n LogÃ­stica entrenado correctamente.")



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predicciones
y_pred = logreg_model.predict(X_test)
y_proba = logreg_model.predict_proba(X_test)[:, 1]

# MÃ©tricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("ğŸ“Œ Resultados del modelo baseline (RegresiÃ³n LogÃ­stica):")
print(f"Accuracy:        {acc:.4f}")
print(f"Precision:       {prec:.4f}")
print(f"Recall:          {rec:.4f}")
print(f"F1-score:        {f1:.4f}")
print(f"ROC-AUC:         {roc_auc:.4f}")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de ConfusiÃ³n - RegresiÃ³n LogÃ­stica")
plt.xlabel("PredicciÃ³n")
plt.ylabel("Actual")
plt.show()



from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')  # lÃ­nea diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - RegresiÃ³n LogÃ­stica')
plt.legend()
plt.grid(True)
plt.show()


