import os
import pandas as pd
import logging
import matplotlib.pyplot as plt


def cargar_datos(nombre_archivo):
    """Esta función carga los datos contenidos en el archivo

    Args:
        nombre_archivo (_type_): El nombre del archivo a cargar
    """
    # El archivo debe estar en la carpeta data
    try:
        base = os.path.dirname(os.getcwd())
        url = os.path.join(base, "data", nombre_archivo)
        
        data = pd.read_csv(url, sep=",", encoding="utf-8")
    
    except:
        pass
    return data


base = os.path.dirname(os.getcwd())
url_train = os.path.join(base, "data", "train.csv")
url_test = os.path.join(base, "data", "train.csv")

train = pd.read_csv(url_train, sep=",", encoding="utf-8")
test = pd.read_csv(url_test, sep=",", encoding="utf-8")


train.shape


train.columns


train.info()


# Valores faltantes
train.isnull().sum()


# Valores duplicados
train.duplicated().sum()


descriptiva = train.describe().T
descriptiva


Q3 = train.quantile(0.75)
Q1 = train.quantile(0.25)
IQR = Q3 - Q1 
lower_limit = Q1 - IQR*1.5 
Upper_limit = Q3 + IQR*1.5 
# Inicializar una serie booleana para mantener el rastreo de las filas sin valores atípicos 
sin_atipicos = pd.Series([True] * len(train)) 
# Iterar sobre cada columna y actualizar la serie booleana si una fila contiene valores atípicos 
for col in train.columns: 
    valores_if = ((train[col] >= lower_limit[col]) & (train[col] <= Upper_limit[col])) 
    sin_atipicos = sin_atipicos & valores_if 
# Filtrar las filas que no contienen valores atípicos 
df_sin_atipicos = train[sin_atipicos] 


# Porcentaje de valores normales
len(df_sin_atipicos) / len(train) * 100


df_sin_atipicos


data = train.drop(["id", "rainfall"], axis=1)


data = train.drop(["id", "rainfall"], axis=1)
row = 4
col = 3

fig, ax = plt.subplots(row, col, figsize=(15, 15))
ax = ax.flatten()

for i, col in enumerate(data):
    ax[i].boxplot(data[col])
    ax[i].set_title(col)
    ax[i].yaxis.grid(True)
    
for j in range(len(data.columns), len(ax)):
    ax[j].set_visible(False)
    
plt.tight_layout()
plt.show()


row = 4
col = 3 

fig, ax = plt.subplots(row, col, figsize=(15 ,15))
ax = ax.flatten()

for i, col in enumerate(data):
    ax[i].violinplot(data[col],
                     showmeans=False,
                     showmedians=True)
    ax[i].set_title(col)
    ax[i].yaxis.grid(True)
    
for j in range(len(data.columns), len(ax)):
    ax[j].set_visible(False)
    
plt.tight_layout()
plt.show()


from scipy.stats import shapiro

for col in data.columns:
    stat, value_p = shapiro(data[col])
    print(f"Shapiro-Wilk para {col}: Estadístico={stat:.4f}, p-valor={value_p:.4f}")


from scipy.stats import skew
from scipy.stats import skewtest

for col in data.columns:
    skewness  = skew(data[col])
    stat, p = skewtest(data[col])
    print(f"{col}, p-valor: {round(p, 2)}, Asimetría (skewness): {round(skewness, 2)}")


from scipy.stats import mannwhitneyu

grupo0 = train[train["rainfall"] == 0].drop(labels=["id"], axis=1)
grupo1 = train[train["rainfall"] == 1].drop(labels=["id"], axis=1)

for col in data.columns:
    # Prueba de Mann-Whitney U
    stat, p = mannwhitneyu(grupo0[col], grupo1[col], alternative='two-sided')

    print(f"Prueba para {col}")
    print(f'Estadístico U: {stat}')
    print(f'P-valor: {p}')

    if p < 0.05:
        print(f"Hay una diferencia significativa entre los grupos.")
    else:
        print(f"No hay una diferencia significativa entre los grupos.")


grupo0.shape


grupo1.shape


import seaborn as sns

plt.figure(figsize=(15, 15))

data = train.drop(labels="id", axis=1)
corr = data.corr("spearman")

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlaciòn entre variables")
plt.show()


corr


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = data.copy()  # Copia el DataFrame original para no modificarlo directamente

# Convierte las columnas numéricas a float antes de aplicar StandardScaler
data_scaled.iloc[:, :-1] = data_scaled.iloc[:, :-1].astype(float)

# Aplica la transformación
data_scaled.iloc[:, :-1] = scaler.fit_transform(data_scaled.iloc[:, :-1])



from pandas.plotting import parallel_coordinates

# Las características en el gráfico deben tener la misma escala, para que el gráfico sea significativo
plt.figure(figsize=(15, 15))
pll = parallel_coordinates(data_scaled, "rainfall", colormap='tab20')
plt.show()


from scipy.stats import norm

# Parámetros estadísticos
Z = norm.ppf(0.975)  # Nivel de confianza 95% → Z = 1.96
E = 0.05  # Margen de error 5%
σ = 0.5  # Suposición conservadora

# Tamaño de muestra para población infinita
n = (Z**2 * σ**2) / (E**2)

# Poblaciones finitas
N_grupo0 = 540
N_grupo1 = 1650

# Ajuste para población finita
n_grupo0 = n / (1 + ((n - 1) / N_grupo0))
n_grupo1 = n / (1 + ((n - 1) / N_grupo1))

print(f"Tamaño de muestra recomendado para grupo0: {round(n_grupo0)}")
print(f"Tamaño de muestra recomendado para grupo1: {round(n_grupo1)}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Contar las instancias por cada clase en la variable objetivo
print(data['rainfall'].value_counts())

# Visualización con gráfico de barras
plt.figure(figsize=(6, 4))
sns.countplot(x=data["rainfall"], hue=data["rainfall"], palette="viridis")
plt.title("Distribución de la variable objetivo")
plt.show()



# Calcular porcentaje de cada clase
clase_0 = (data['rainfall'].value_counts()[0] / len(data)) * 100
clase_1 = (data['rainfall'].value_counts()[1] / len(data)) * 100

print(f"Clase 0: {clase_0:.2f}% del total")
print(f"Clase 1: {clase_1:.2f}% del total")


ratio = data['rainfall'].value_counts().max() / data['rainfall'].value_counts().min()
print(f"Ratio de desbalanceo: {ratio:.2f}")


import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt


base = os.path.dirname(os.getcwd())
url_train = os.path.join(base, "data", "train.csv")
url_test = os.path.join(base, "data", "test.csv")

train = pd.read_csv(url_train, sep=",", encoding="utf-8")
test = pd.read_csv(url_test, sep=",", encoding="utf-8")


import logging

logging.basicConfig(level="INFO",
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H-%M-%S",
                    handlers = [logging.StreamHandler()])

logger = logging.getLogger(__name__)


# Utilizamos las clases bases BaseEstimator y TransformerMixin para crear los transformadores
# Personalizados
from sklearn.base import BaseEstimator, TransformerMixin


class CorreccionTipograficos(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para corregir erorres tipogràficos en los nombres de las columnas

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
        
    Methods:
    fit(X, y=None):
        Método requerido por `scikit-learn`, no realiza ninguna operación en el ajuste.
    
    transform(X):
        Corrige errores tipográficos en las columnas del DataFrame.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        logger.info("CorreccionTipograficos - fit ejecutado")
        return self 
      
    def transform(self, X):
        X = X.copy() 
        X = X.rename(columns={"temparature": "temperature"})
        return pd.DataFrame(X)


class DropColumn(BaseEstimator, TransformerMixin):
    """_summary_
    Transformador personalizado para eliminar columnas irrelevantes 
    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
        
    Methods:
    fit(X, y=None):
        Método requerido por `scikit-learn`, no realiza ninguna operación en el ajuste.
    
    transform(X):
        Elimina las columnas irrelevantes del DataFrame.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        logger.info("DropColumn - fit ejecutado")
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        return pd.DataFrame(X)  # Mantener estructura de DataFrame


class DropDuplicated(BaseEstimator, TransformerMixin):
    """_summary_
    Transformador personalizado para eliminar los datos duplicados  
    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
        
    Methods:
    fit(X, y=None):
        Método requerido por `scikit-learn`, no realiza ninguna operación en el ajuste.
    
    transform(X):
        Elimina los datos duplicados del DataFrame.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        logger.info("DropDuplicated - fit ejecutado")
        return self
    
    def transform(self, X):
        return X.drop_duplicates()


class DropNan(BaseEstimator, TransformerMixin):
    """_summary_
    Transformador personalizado para eliminar los datos nulos  
    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
        
    Methods:
    fit(X, y=None):
        Método requerido por `scikit-learn`, no realiza ninguna operación en el ajuste.
    
    transform(X):
        Elimina los datos nulos  del DataFrame.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        logger.info("DropNan - fit ejecutado")
        return self
    
    def transform(self, X):
        X = X.dropna()
        return pd.DataFrame(X, columns=X.columns)
        


X = train.drop(labels=["rainfall"], axis=1)
y = train["rainfall"]


X["dewpoint_ratio"] = X["dewpoint"] / X["temparature"]
X["cloud_humidity_index"] = X["cloud"] * X["humidity"]
X["temp_range"] = X["maxtemp"] - X["mintemp"]
X["temp_pressure_ratio"] = X["maxtemp"] / X["pressure"]


from sklearn.model_selection import train_test_split
random_seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_seed)


## Imputar datos con knn
from sklearn.impute import KNNImputer

preprocessor = [
    ('tipograficos', CorreccionTipograficos()),
    ('drop_column', DropColumn(columns_to_drop=["id", "day"])),
    ('drop_duplicated', DropDuplicated()),
    ('impute_knn', KNNImputer(n_neighbors=5))
]


from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


base_clf = DecisionTreeClassifier()

pipeline_lr = ImbPipeline(preprocessor + [  # Preprocesador directamente
    ('scaler', StandardScaler()),  # Normalización
    ('smote', SMOTE(sampling_strategy=0.8, random_state=random_seed)),  # SMOTE después de normalizar
    ('model_lr', LogisticRegression(random_state=random_seed))
])

pipeline_rf = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_rf', RandomForestClassifier(random_state=random_seed))
])

pipeline_sgd = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_sgd', SGDClassifier(random_state=random_seed))
])

pipeline_svc = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_svc', SVC(probability=True, random_state=random_seed))
])

pipeline_AdaBoostClassifier = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_AdaBoost', AdaBoostClassifier(estimator=base_clf, random_state=random_seed))
])

pipeline_MLP = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_MLP', MLPClassifier(random_state=random_seed, max_iter=1000))
])

pipeline_xgb = ImbPipeline(preprocessor + [ 
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.7, random_state=random_seed)),
    ('model_xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42))
])


list_pipelines = [
    pipeline_lr,
    pipeline_rf,
    pipeline_sgd,
    pipeline_svc,
    pipeline_AdaBoostClassifier,
    pipeline_MLP,
    pipeline_xgb
    ]

best_roc_auc = 0.0
best_classifier = 0
best_pipeline = ""

# Diccionario de pipelines y tipo de clasificador
pipe_dict = {0: "LogisticRegression",
             1: "RandomForestClassifier",
             2: "SGDClassifier",
             3: "SVC",
             4: "AdaBoostClassifier",
             5: "MLPClassifier",
             6: "XGBClassifier"}


# Entrenamiento de pipelines
for pipe in list_pipelines:
  pipe.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score

for i, model in enumerate(list_pipelines):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"{pipe_dict[i]} roc_auc en prueba: {roc_auc:.2f}")


# Selección del mejor modelo
for i, model in enumerate(list_pipelines): # Toma tanto el indixe de la lista como el modelo
  y_pred = model.predict(X_test)
  roc_auc = roc_auc_score(y_test, y_pred)
  if roc_auc > best_roc_auc: # Verifica si el score del modelo actual es mejor que el anterior de ser asi se actualiza las variables
  # best_cof, best_pipeline y best_regressor. De no serlo no se actualizan quedando el valor más alto.
    best_roc_auc = roc_auc
    best_pipeline = model
    best_classifier = i
print("El clasificador con el mejor roc_auc_score es: {}".format(pipe_dict[best_classifier]))


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import scipy 

parameters = {
    "model_svc__C": scipy.stats.loguniform(1e-3, 1e3),  # Rango amplio para la regularización
    "model_svc__kernel": ["linear", "poly", "rbf", "sigmoid"],  # Diferentes núcleos
    "model_svc__degree": scipy.stats.randint(2, 5),  # Solo aplica para kernel 'poly'
    "model_svc__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1, 10],  # Aplica para 'rbf', 'poly' y 'sigmoid'
    "model_svc__coef0": scipy.stats.uniform(0, 1),  # Solo para 'poly' y 'sigmoid'
    "model_svc__shrinking": [True, False],  # Prueba con y sin reducción de soporte
}

clf = RandomizedSearchCV(
    pipeline_svc,
    parameters,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    refit="roc_auc",
    verbose=0,
    random_state=random_seed
)

clf.fit(X_train, y_train)


clf.best_score_


clf.best_params_


test["dewpoint_ratio"] = test["dewpoint"] / test["temparature"]
test["cloud_humidity_index"] = test["cloud"] * test["humidity"]
test["temp_range"] = test["maxtemp"] - test["mintemp"]
test["temp_pressure_ratio"] = test["maxtemp"] / test["pressure"]


from sklearn.model_selection import cross_val_score
best_svc = clf.best_estimator_
scores = cross_val_score(best_svc, X, y, cv=5, scoring="roc_auc")
scores


best_svc = clf.best_estimator_
best_svc.fit(X, y)

# Predicción de clases
y_pred = best_svc.predict(test)

# Si entrenaste con probability=True, puedes obtener probabilidades
if hasattr(best_svc, "predict_proba"):
    y_prob = best_svc.predict_proba(test)[:, 1]


df_results = pd.DataFrame({
    "probability": y_prob
}, index=test["id"][:len(y_prob)])  # Usamos los índices de test como índice del nuevo DataFrame

# Renombramos el índice para mayor claridad
df_results.index.name = "id"

# Mostrar el DataFrame resultante
print(df_results.head())



base = os.path.dirname(os.getcwd())
path = os.path.join(base, "data", "predictions", "predicciones.csv")
df_results.to_csv(path, sep=",", encoding="utf-8")

