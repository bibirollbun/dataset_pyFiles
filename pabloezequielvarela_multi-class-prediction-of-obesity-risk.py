import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


# Separar características y etiquetas
X = train.drop(columns=['NObeyesdad', 'id'])  # Asumimos que 'id' no aporta valor predictivo
y = train['NObeyesdad']

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identificar columnas categóricas y numéricas automáticamente
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Definir los modelos que queremos probar
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42)
}

# Entrenar y evaluar cada modelo
results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"{name} - Accuracy: {acc:.4f}")

# Mostrar resultados
print("\nModel Comparison (Validation Accuracy):")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")


# Eliminar columnas no deseadas (por ejemplo, 'id')
columns_to_drop = ['NObeyesdad', 'id']  # Ajusta según tu dataset

X = train.drop(columns=columns_to_drop, errors='ignore')  # Usa errors='ignore' por seguridad
y = train['NObeyesdad']
X_test = test.drop(columns=columns_to_drop, errors='ignore')  # Solo si 'id' está en test


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)


categorical_cols = X.select_dtypes(include=['object']).columns.tolist()  # Fixed tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough',   # Opcional: útil si hay columnas no especificadas
    verbose_feature_names_out=False  # Evita prefijos largos al obtener nombres de columnas
)


models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000,
        random_state=42
    ),
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale', 
        decision_function_shape='ovr',
        probability=True,     # Necesario si usas predict_proba() después
        class_weight='balanced'  # Manejo de clases desbalanceadas
    )
}


trained_pipelines = {}  # Para guardar los pipelines entrenados

results = {}

print("Training and evaluating models...\n")

for name, model in models.items():
    try:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Entrenar y predecir
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        # Calcular accuracy
        acc = accuracy_score(y_val, y_pred)
        results[name] = acc
        trained_pipelines[name] = pipeline
        
        print(f"{name:<20} Validation Accuracy: {acc:.4f}")
    
    except Exception as e:
        print(f"{name} failed with error: {str(e)}")
        



print("Generating submissions...\n")

for name, model in models.items():
    try:
        # Crear un nuevo pipeline por modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Entrenar con todo el conjunto de entrenamiento
        pipeline.fit(X, y)
        
        # Predecir en test
        preds = pipeline.predict(X_test)
        
        # Crear DataFrame de submission
        if 'id' in test.columns:
            submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': preds})
        else:
            submission = pd.DataFrame({'id': range(len(preds)), 'NObeyesdad': preds})
            print(f"[Warning] 'id' column not found in test data for {name}, using default indices.")
        
        # Guardar CSV
        filename = f'submission_{name.lower().replace(" ", "_")}.csv'
        submission.to_csv(filename, index=False)
        print(f"✔️ Submission file '{filename}' generated successfully.")
    
    except Exception as e:
        print(f"❌ Error generating submission for {name}: {str(e)}")
        continue

print("\n✅ All submissions processed!")
        continue


import os

source_path = '/kaggle/working/submission_svm.csv'
target_path = '/kaggle/working/submission.csv'

if os.path.exists(source_path):
    if os.path.exists(target_path):
        os.remove(target_path)
    
    os.rename(source_path, target_path)
    print(f"✅ Renamed '{source_path}' to '{target_path}'")
else:
    print(f"❌ Error: {source_path} does not exist")

print("\nCurrent files in /kaggle/working:")
print(os.listdir('/kaggle/working'))

