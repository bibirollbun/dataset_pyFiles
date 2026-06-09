# IMPORTAR LIBRERÍAS Y CARGAR DATOS
import pandas as pd                         
import matplotlib.pyplot as plt            

df = pd.read_csv('/kaggle/input/horse/train.csv')              
print("Dimensiones:", df.shape)             
print("\nPrimeras 5 filas:")                
display(df.head())                          
print("\nValores nulos por columna:")
print(df.isnull().sum())                    

# IDENTIFICAR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  
for drop in ['id', 'hospital_number', 'outcome']:  
    if drop in numeric_cols:                 
        numeric_cols.remove(drop)           
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()                                           
if 'outcome' in categorical_cols:           
    categorical_cols.remove('outcome')      
    
print("                    ")
print("Columnas numéricas:", numeric_cols)
print("                    ")
print("Columnas categóricas:", categorical_cols)

# IMPUTAR Y ESCALAR VARIABLES NUMÉRICAS
for col in numeric_cols:                   
    mean = df[col].mean()                   
    print(mean)
    df[col] = df[col].fillna(mean)        
    std = df[col].std()                     
    df[col + '_scaled'] = (df[col] - mean) / std  
                                       
# CODIFICAR CATEGÓRICAS 
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# VISUALIZACIÓN DE UN ATRIBUTO EJEMPLO
col = 'rectal_temp'                 
plt.figure(figsize=(5,3))                   
df[col].hist(bins=20)                      
plt.title(f'Distribución de {col}')         
plt.xlabel(col)                            
plt.ylabel('Frecuencia')                   
plt.show()                                  

col_1 = 'pulse'                  
plt.figure(figsize=(5,3))                  
df[col_1].hist(bins=20)                      
plt.title(f'Distribución de {col_1}')        
plt.xlabel(col_1)                           
plt.ylabel('Frecuencia')                    
plt.show()   

col_2 = 'respiratory_rate'                  
plt.figure(figsize=(5,3))                   
df[col_2].hist(bins=20)                      
plt.title(f'Distribución de {col_2}')        
plt.xlabel(col_2)                           
plt.ylabel('Frecuencia')                    
plt.show()   

col_3 = 'nasogastric_reflux_ph'                  
plt.figure(figsize=(5,3))                   
df[col_3].hist(bins=20)                       
plt.title(f'Distribución de {col_3}')        
plt.xlabel(col_3)                           
plt.ylabel('Frecuencia')                    
plt.show()   

col_4 = 'age'                  
plt.figure(figsize=(5,3))                   
df[col_3].hist(bins=20)                       
plt.title(f'Distribución de {col_4}')         
plt.xlabel(col_4)                             
plt.ylabel('Frecuencia')                    
plt.show()   

col_5 = 'pain'                 
plt.figure(figsize=(5,3))                   
df[col_3].hist(bins=20)                       
plt.title(f'Distribución de {col_5}')         
plt.xlabel(col_5)                             
plt.ylabel('Frecuencia')                    
plt.show()   


# 1 Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Cargar datos
df = pd.read_csv('/kaggle/input/horse/train.csv')  
print("Datos cargados:", df.shape)

# Preparar x y
X = df.drop(['id', 'hospital_number', 'outcome'], axis=1)  
y = df['outcome'].map({'died': 0, 'euthanized': 1, 'lived': 2})  

# Codificar variables categóricas
X = pd.get_dummies(X, drop_first=True)  
print("Columnas tras codificar:", X.shape[1])

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,      
    stratify=y,        
    random_state=42
)
print("Train:", X_train.shape, "Validation:", X_val.shape)

# Entrenar Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("RE entrenado.")

# Entrenar Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=50,   
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
print("GB entrenado.")



# Importar las métricas que vamos a usar
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report)

# Hacer predicciones sobre el conjunto de validación
y_pred_rf = rf.predict(X_val)  
y_pred_gb = gb.predict(X_val) 

# Función para imprimir resultados de forma ordenada
def evaluar(modelo_nombre, y_true, y_pred):
    print(f"\n--- {modelo_nombre} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro'):.3f}")
    print(f"F1-score : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print("\nReporte detallado:")
    print(classification_report(y_true, y_pred))

# Ejecutar la evaluación para cada modelo
evaluar("RF",      y_val, y_pred_rf)
evaluar("GB",  y_val, y_pred_gb)






# Cargar test
df_test = pd.read_csv('/kaggle/input/horse/test.csv')

# Preprocesar (igual que train)
X_test = df_test.drop(['id','hospital_number'], axis=1)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Predecir con el mejor modelo (por ejemplo rf)
y_test_pred = rf.predict(X_test)

# Mapear de vuelta a etiquetas
inv_map = {0:'died', 1:'euthanized', 2:'lived'}
y_test_labels = [inv_map[i] for i in y_test_pred]

# Crear CSV de envío
submission = pd.DataFrame({
    'id': df_test['id'],
    'outcome': y_test_labels
})
submission.to_csv('submission.csv', index=False)

print(submission)  




