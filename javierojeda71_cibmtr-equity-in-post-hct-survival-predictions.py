import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar datos
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


# Filtrar columnas con más de 24000 valores no nulos
threshold = 24000
columnas_a_conservar = train.columns[train.notnull().sum() > threshold]
train_filtrado = train[columnas_a_conservar]
test_filtrado = test[columnas_a_conservar.drop(['efs', 'efs_time', 'ID'], errors='ignore')]

# Separar variables independientes y dependientes
x_train = train_filtrado.drop(['efs', 'efs_time', 'ID'], axis=1, errors='ignore')
y_train = train_filtrado[['efs', 'efs_time']]
x_test = test_filtrado

# Identificar columnas con valores nulos
coltofix = x_train.columns[x_train.isnull().any()]
coltofixtest = x_test.columns[x_test.isnull().any()]

# Reemplazar valores nulos (moda para categóricas, media para numéricas)
for col in coltofix:
    if x_train[col].dtype == "object":
        x_train[col].fillna(x_train[col].mode()[0], inplace=True)
    else:
        x_train[col].fillna(x_train[col].mean(), inplace=True)

# Aplicar el mismo tratamiento al dataset de test
for col in coltofixtest:
    if x_test[col].dtype == "object":
        x_test[col].fillna(x_test[col].mode()[0], inplace=True)
    else:
        x_test[col].fillna(x_test[col].mean(), inplace=True)

# Normalizar columnas numéricas excluyendo 'year_hct'
numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).drop(['year_hct'], errors='ignore').columns
numerical_cols_test = x_test.select_dtypes(include=['int64', 'float64']).drop(['year_hct'], errors='ignore').columns
scaler = StandardScaler()
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols_test] = scaler.transform(x_test[numerical_cols_test])

# Codificación one-hot para variables categóricas
categorical_cols = x_train.select_dtypes(include=['object']).columns
categorical_cols_test = x_test.select_dtypes(include=['object']).columns
x_train_encoded = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=categorical_cols_test, drop_first=True)


# Convertir booleanos a enteros
bool_cols = x_train_encoded.select_dtypes(include=['bool']).columns
bool_cols_test = x_test_encoded.select_dtypes(include=['bool']).columns
x_train_encoded[bool_cols] = x_train_encoded[bool_cols].astype(int)
x_test_encoded[bool_cols_test] = x_test_encoded[bool_cols_test].astype(int)

# DataFrames finales
df_train_final = x_train_encoded
df_test_final = x_test_encoded

varrel = ['donor_age', 'age_at_hct', 'comorbidity_score', 'prim_disease_hct_IEA', 'prim_disease_hct_IIS', 'prim_disease_hct_HIS', 'prim_disease_hct_HD', 'cardiac_Yes', 'prod_type_PB', 'graft_type_Peripheral blood', 'conditioning_intensity_RIC', 'prim_disease_hct_PCD', 'gvhd_proph_TDEPLETION +- other', 'in_vivo_tcd_Yes', 'gvhd_proph_CSA + MMF +- others(not FK)', 'dri_score_N/A - pediatric', 'prior_tumor_Yes', 'pulm_severe_Yes', 'dri_score_N/A - non-malignant indication', 'hepatic_severe_Yes', 'dri_score_TBD cytogenetics', 'gvhd_proph_FKalone', 'prim_disease_hct_SAA', 'prim_disease_hct_Other leukemia', 'prim_disease_hct_IMD', 'gvhd_proph_CSA alone', 'dri_score_N/A - disease not classifiable', 'gvhd_proph_CDselect alone', 'prim_disease_hct_Solid tumor', 'dri_score_Very high', 'diabetes_Not done']
lista = []
for var in df_test_final.columns:
    if var in varrel:
        lista.append(var)
lista






import pandas as pd
from lifelines import CoxPHFitter

# Preparar datos de entrenamiento
df_train = df_train_final[lista].copy()
df_train['T'] = y_train["efs_time"]
df_train['E'] = y_train["efs"]

# Entrenar el modelo de regresión de Cox
cph = CoxPHFitter().fit(df_train, duration_col='T', event_col='E')

# Hacer predicciones de riesgo en el conjunto de prueba
df_test = df_test_final[lista].copy()
df_test['prediction'] = cph.predict_partial_hazard(df_test)

# Crear y guardar el archivo de submission
submission = pd.DataFrame({'ID': test['ID'], 'prediction': df_test['prediction']})
submission.to_csv('/kaggle/working/submission.csv', index=False)



