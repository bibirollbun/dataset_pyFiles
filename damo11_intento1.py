#traer librerias necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

#pip install optuna #para encontrar paràmetros òptimos posteriormente

#leer datos
telefonia_train = pd.read_excel("/kaggle/input/retencion-en-telefonia-movil-2501/traintelco.xlsx")
telefonia_test = pd.read_excel("/kaggle/input/retencion-en-telefonia-movil-2501/testelco.xlsx")

#ver las columnas de train (para ver que variables hay)
telefonia_train.columns

#ver los tipos de datos de las columnas
print(telefonia_train.info())

#convertir variables a categoricas
columnas_categoricas = {
    "tipo cliente": "tipocli",
    "Factura online": "factonline",
    "Plan de datos": "plandatos"
}
for old, new in columnas_categoricas.items():
    telefonia_train[new] = telefonia_train[old].astype("category")
    telefonia_test[new] = telefonia_test[old].astype("category")

telefonia_train["result"] = telefonia_train["resultado"].astype("category")

#ver la informaciòn de los tipos de datos en las columnas (ver si sì sirviò la conversiòn)
print(telefonia_train.info())

#añadir columnas para adaptar los datos de mejor manera al contexto dado
fecha_corte = pd.to_datetime("2019-01-01")
telefonia_train["edad"] = ((fecha_corte - pd.to_datetime(telefonia_train["Fecha de nacimiento"])).dt.days / 365.25).round() #cantidad de dias en un año  (decidiendo no considerar a los años bisisestos)
telefonia_test["edad"] = ((fecha_corte - pd.to_datetime(telefonia_test["Fecha de nacimiento"])).dt.days / 365.25).round()

telefonia_train["antiguedad"] = (fecha_corte - pd.to_datetime(telefonia_train["Fecha inicio contrato"])).dt.days / 30 #cantidad de dias promedio en un mes 
telefonia_test["antiguedad"] = (fecha_corte - pd.to_datetime(telefonia_test["Fecha inicio contrato"])).dt.days / 30

telefonia_train["facturacion mensual promedio"] = telefonia_train["facturación"] / 6
telefonia_test["facturacion mensual promedio"] = telefonia_test["facturación"] / 6

telefonia_train["minutos mensuales promedio"] = telefonia_train["minutos"] / 43800 #minutos promedio por mes
telefonia_test["minutos mensuales promedio"] = telefonia_test["minutos"] / 43800

#quitar outliers 
columnas_numericas = ['facturación', 'mora', 'minutos', 'Antigüedad Equipo', 'minutos mensuales promedio', 'facturacion mensual promedio', "edad", "antiguedad"]

def quitar_outliers_iqr(df, columnas):
    df_limpio = df.copy()
    for col in columnas:
        Q1 = df_limpio[col].quantile(0.25)
        Q3 = df_limpio[col].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df_limpio[col] >= Q1 - 1.5 * IQR) & (df_limpio[col] <= Q3 + 1.5 * IQR)# Filtro para mantener solo los valores dentro de un rango considerado aceptable
        df_limpio = df_limpio[filtro]
    return df_limpio

telefonia_train = quitar_outliers_iqr(telefonia_train, columnas_numericas)

#ver como quedò el conjunto de datos
telefonia_train.head()

print(telefonia_train.info())

#eliminar columnas que se volvieron irrelevantes / redundantes
drop_cols_train = ["Fecha de nacimiento", "Fecha inicio contrato", "tipo cliente", "Factura online", "resultado"]
drop_cols_test = ["Fecha de nacimiento", "Fecha inicio contrato", "tipo cliente", "Factura online"]
telefonia_train_2 = telefonia_train.drop(columns=drop_cols_train)
telefonia_test_2 = telefonia_test.drop(columns=drop_cols_test)

#estandarizaciòn a las columnas
columnas_numericas_train = telefonia_train_2.select_dtypes(include=['float64', 'int64']).columns
columnas_numericas_test = telefonia_test_2.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()

telefonia_train_2[columnas_numericas_train] = scaler.fit_transform(telefonia_train_2[columnas_numericas_train])
telefonia_test_2[columnas_numericas_test] = scaler.transform(telefonia_test_2[columnas_numericas_test])

#reducir variables del conjunto de datos
columns_to_keep = ['facturación', 'edad', 'tipocli', 'antiguedad', 'mora', 'result', 'facturacion mensual promedio', 'minutos mensuales promedio']

selected_features = ['facturación', 'edad', 'tipocli', 'antiguedad', 'mora', 'facturacion mensual promedio', 'minutos mensuales promedio']
telefonia_train_2 = telefonia_train_2[columns_to_keep]

telefonia_test_2 = telefonia_test_2[selected_features]

#observar como quedò el conjunto de datos
telefonia_train_2

#separar los datos para train y test del modelo 
train_data, test_data = train_test_split(telefonia_train_2, test_size=0.3, random_state=42, stratify=telefonia_train_2["result"])

#realizar variables dummies para evitar malinterpretaciones de datos
X_train = pd.get_dummies(train_data.drop(columns="result"), drop_first=True)
y_train = train_data["result"].astype(int)
X_test = pd.get_dummies(test_data.drop(columns="result"), drop_first=True)
y_test = test_data["result"].astype(int)

#asegurar que train y test tengan las mismas columnas
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#ver nulos en train
telefonia_train_2.isnull().sum()

#ver nulos en test
telefonia_test_2.isnull().sum()

#ver cantidad de datos en train
print(f"Cantidad de valores en el conjunto de datos de entrenamiento: {len(train_data)}")

#ver cantidad de datos en test
print(f"Cantidad de valores en el conjunto de datos de test: {len(test_data)}")

#encontrar parametros òptimos para el modelo
##import optuna####
##
###def objective(trial):
##    gb = GradientBoostingClassifier(
###        n_estimators=trial.suggest_int('n_estimators', 100, 500),
###        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
##        max_depth=trial.suggest_int('max_depth', 2, 6),
##        random_state=42
##    )
##    gb.fit(X_train, y_train)
#    return gb.score(X_test, y_test)  # Usa un conjunto de validación
    
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=100)
#print("Mejores parámetros:", study.best_params)

#crear y entrenar modelo
gb_modelo = GradientBoostingClassifier(n_estimators=278, learning_rate=0.12534051449832964, max_depth=2, random_state=42) 
gb_modelo.fit(X_train, y_train)

#predecir en el conjunto de validaciòn 
y_pred_gb = gb_modelo.predict(X_test)
y_proba_gb = gb_modelo.predict_proba(X_test)[:, 1]

#calcular mètricas para Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, y_proba_gb)

#ver las mètricas calculadas
print("\nMétricas del modelo Gradient Boosting:")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall: {recall_gb:.4f}")
print(f"F1-score: {f1_gb:.4f}")
print("Matriz de Confusión:")
print(conf_matrix_gb)
print(f"AUC-ROC: {roc_auc_gb:.9f}")

#graficar la curva ROC para Gradient Boosting
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, label=f'(AUC = {roc_auc_gb:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

#realizar predicciones en "testelco"
telefonia_test_pred = telefonia_test_2.copy()

X_test_pred = pd.get_dummies(telefonia_test_pred)

for col in X_train.columns:
    if col not in X_test_pred.columns:
        X_test_pred[col] = X_train[col].mean()

X_test_pred = X_test_pred[X_train.columns]

predic_proba = gb_modelo.predict_proba(X_test_pred)[:, 1]

resultados_df = pd.DataFrame({
    'id': telefonia_test['id'],
    'resultado': predic_proba
})

try:
    resultados_df.to_csv('samplesubtelco.csv', index=False)
    print("Archivo 'samplesubtelco.csv' guardado exitosamente.")
except Exception as e:
    print(f"Error al guardar el archivo: {e}")

