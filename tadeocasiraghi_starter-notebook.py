COMP_DIR = "/kaggle/input/copy-hackaton-udesa-ort"
TRAIN = f"{COMP_DIR}/train.csv"
TEST  = f"{COMP_DIR}/test.csv"
SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"


import pandas as pd

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
sample = pd.read_csv(SAMPLE_SUB)


# visualizamos (primeras filas)
train.head()


# algunas estadÃ­sticas bÃ¡sicas
train.describe()


# Definimos el target y el ID Ãºnico (importante para la submission)
ID_COL = "track_id"
TARGET = "is_hit"


# Generamos el dummy submission para entrar a la competencia
dummy = test[[ID_COL]].copy()
dummy[TARGET] = 0  # predicciones 0 o 1 son vÃ¡lidas para hacer un submission
dummy.to_csv("/kaggle/working/submission.csv", index=False)
print("Dummy submission created:", dummy.head(), sep="\n")


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Limpieza de Nulos
train_df = train.dropna()

# 2. SeparaciÃ³n de caracterÃ­sticas (X) y objetivo (y)
X = train_df.drop([TARGET,ID_COL], axis=1)
y = train_df[TARGET]

# 3. CodificaciÃ³n One-Hot para variables categÃ³ricas
# pd.get_dummies convierte columnas categÃ³ricas en 0s y 1s
X_encoded = pd.get_dummies(X, drop_first=True)

# 4. DivisiÃ³n en conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
     X_encoded, y, test_size=0.2, random_state=42, stratify=y)

print(f'\nTamaÃ±o del set de entrenamiento: {X_train.shape[0]} muestras')
print(f'TamaÃ±o del set de prueba: {X_test.shape[0]} muestras')

# # create model instance
xgb = XGBClassifier(n_estimators=500, max_depth=2, learning_rate=0.01, objective='binary:logistic')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)


plt.figure()

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=plt.gca())
plt.title('Matriz de ConfusiÃ³n - XGB')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')

plt.tight_layout()
plt.show()


# 1. Limpieza de Nulos
test_df = test.dropna()

# 2. SeparaciÃ³n de caracterÃ­sticas (X) y objetivo (y)
test_X = test_df.drop([ID_COL], axis=1)

# 3. CodificaciÃ³n One-Hot para variables categÃ³ricas
# pd.get_dummies convierte columnas categÃ³ricas en 0s y 1s
test_X_encoded = pd.get_dummies(test_X, drop_first=True)
test_predict = xgb.predict(test_X_encoded)

# Generamos el dummy submission para entrar a la competencia
dummy = test[[ID_COL]].copy()
dummy[TARGET] = test_predict

dummy.to_csv("/kaggle/working/submission.csv", index=False)
print("Dummy submission created:", dummy.head(), sep="\n")

