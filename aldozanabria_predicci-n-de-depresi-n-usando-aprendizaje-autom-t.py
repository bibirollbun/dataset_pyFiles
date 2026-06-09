# 1. Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# AGREGAR ESTAS LIBRERÍAS PARA VISUALIZACIÓN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 2. Cargar los datos 
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')

# 3. Definir variables 
target = "Depression"
features_num = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
features_cat = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                'Sleep Duration', 'Dietary Habits', 'Degree',
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# 4. Preprocesar datos 
train = train.drop(columns=['id', 'Name'])
X = train.drop(columns=[target])
y = train[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

#rellenar los valores nulos en las columnas numéricas con la mediana de esa columna. La mediana es robusta a valores atípicos (outliers).
num_imputer = SimpleImputer(strategy='median')
#rellenar los valores nulos en las columnas categóricas con el valor más frecuente (moda) de esa columna.
cat_imputer = SimpleImputer(strategy='most_frequent')
#convertir las columnas categóricas en un formato numérico (dummies binarias)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

#Aplica la imputación de medianas a las características numéricas del conjunto de entrenamiento.
X_num_train = pd.DataFrame(num_imputer.fit_transform(X_train[features_num]), columns=features_num)
#plica la imputación de la moda y luego la codificación One-Hot a las características categóricas del conjunto de entrenamiento.
X_cat_train = pd.DataFrame(cat_imputer.fit_transform(X_train[features_cat]), columns=features_cat)
#Aplica la imputación a las características numéricas del conjunto de validación.
X_cat_train_enc = pd.DataFrame(encoder.fit_transform(X_cat_train), columns=encoder.get_feature_names_out(features_cat))

X_num_val = pd.DataFrame(num_imputer.transform(X_val[features_num]), columns=features_num)
X_cat_val = pd.DataFrame(cat_imputer.transform(X_val[features_cat]), columns=features_cat)
X_cat_val_enc = pd.DataFrame(encoder.transform(X_cat_val), columns=encoder.get_feature_names_out(features_cat))

#Combina las características numéricas preprocesadas y las características categóricas codificadas para formar los DataFrames finales que se usarán para entrenar y validar los modelos.
X_train_final = pd.concat([X_num_train.reset_index(drop=True), X_cat_train_enc.reset_index(drop=True)], axis=1)
X_val_final = pd.concat([X_num_val.reset_index(drop=True), X_cat_val_enc.reset_index(drop=True)], axis=1)

missing_cols = set(X_train_final.columns) - set(X_val_final.columns)
for col in missing_cols:
    X_val_final[col] = 0
X_val_final = X_val_final[X_train_final.columns]

# Entrenar y comparar modelos 
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42), #aqui se incremento ya que teniamos errores 
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# M RECOLECTAR PREDICCIONES Y PROBABILIDADES
results = []
predictions_dict = {}  
probabilities_dict = {}  

for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_val_final)
    
    # LÍNEAS PARA RECOLECTAR DATOS
    predictions_dict[name] = y_pred
    
    # Calcular probabilidades si el modelo las soporta
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val_final)[:, 1]
        probabilities_dict[name] = y_proba
        roc_auc = roc_auc_score(y_val, y_proba)
    else:
        probabilities_dict[name] = None
        roc_auc = None
    
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    results.append({
        'Modelo': name,
        'Accuracy': round(accuracy_score(y_val, y_pred), 4),
        'Precision': round(precision_score(y_val, y_pred), 4),
        'Recall': round(recall_score(y_val, y_pred), 4),
        'F1-Score': round(f1_score(y_val, y_pred), 4),
        'ROC-AUC': round(roc_auc, 4) if roc_auc else 'N/A'  
    })

results_df = pd.DataFrame(results)
print(results_df)

#  NOMBRES DE MODELOS
model_names = list(models.keys())

# ================== FUNCIONES DE VISUALIZACIÓN ==================

def plot_metrics_comparison(results_df):
    """Gráfico de barras comparativo de métricas por modelo"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparación de Métricas por Modelo', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Filtrar modelos que tienen la métrica
        data_to_plot = results_df[results_df[metric] != 'N/A']
        
        bars = ax.bar(data_to_plot['Modelo'], data_to_plot[metric], 
                     color=sns.color_palette("husl", len(data_to_plot)))
        
        ax.set_title(f'{metric} por Modelo', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotar etiquetas del eje x
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(y_val, predictions_dict, model_names):
    """Matrices de confusión para todos los modelos"""
    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Matrices de Confusión por Modelo', fontsize=16, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, name in enumerate(model_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        y_pred = predictions_dict[name]
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Depresión', 'Depresión'],
                   yticklabels=['No Depresión', 'Depresión'])
        
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_ylabel('Valores Reales')
        ax.set_xlabel('Predicciones')
    
    # Ocultar subplots vacíos
    for idx in range(n_models, rows * cols):
        row, col = idx // cols, idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(y_val, probabilities_dict, model_names):
    """Curvas ROC para modelos que soportan predict_proba"""
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("husl", len(model_names))
    
    for idx, name in enumerate(model_names):
        if probabilities_dict[name] is not None:
            y_proba = probabilities_dict[name]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos', fontweight='bold')
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    """Gráfico de importancia de características para modelos que lo soporten"""
    if hasattr(model, 'feature_importances_'):
        # Obtener importancia de características
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importancia de la Característica', fontweight='bold')
        plt.title(f'Top {top_n} Características Más Importantes (Random Forest)', fontweight='bold')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("El modelo no soporta feature_importances_")

# ================== GENERAR VISUALIZACIONES ==================

print("\n" + "="*60)
print("GENERANDO VISUALIZACIONES")
print("="*60)

print("\n1. Comparación de métricas...")
plot_metrics_comparison(results_df)

print("\n2. Matrices de confusión...")
plot_confusion_matrices(y_val, predictions_dict, model_names)

print("\n3. Curvas ROC...")
plot_roc_curves(y_val, probabilities_dict, model_names)

print("\n4. Importancia de características (Random Forest)...")
plot_feature_importance(models['Random Forest'], X_train_final.columns)

print("\n" + "="*60)
print("¡VISUALIZACIONES COMPLETADAS!")
print("="*60)


# 1. Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# AGREGAR ESTA LIBRERÍA PARA MANEJAR DESBALANCE DE CLASES
from imblearn.over_sampling import SMOTE # <-- CAMBIO 1: Importar SMOTE

# AGREGAR ESTAS LIBRERÍAS PARA VISUALIZACIÓN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 2. Cargar los datos
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')

# 3. Definir variables
target = "Depression"
features_num = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
features_cat = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                'Sleep Duration', 'Dietary Habits', 'Degree',
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# 4. Preprocesar datos
train = train.drop(columns=['id', 'Name'])
X = train.drop(columns=[target])
y = train[target]

# CAMBIO 2: Añadir stratify=y en train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # <-- CAMBIO 2

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_num_train = pd.DataFrame(num_imputer.fit_transform(X_train[features_num]), columns=features_num)
X_cat_train = pd.DataFrame(cat_imputer.fit_transform(X_train[features_cat]), columns=features_cat)
X_cat_train_enc = pd.DataFrame(encoder.fit_transform(X_cat_train), columns=encoder.get_feature_names_out(features_cat))

X_num_val = pd.DataFrame(num_imputer.transform(X_val[features_num]), columns=features_num)
X_cat_val = pd.DataFrame(cat_imputer.transform(X_val[features_cat]), columns=features_cat)
X_cat_val_enc = pd.DataFrame(encoder.transform(X_cat_val), columns=encoder.get_feature_names_out(features_cat))

X_train_final = pd.concat([X_num_train.reset_index(drop=True), X_cat_train_enc.reset_index(drop=True)], axis=1)
X_val_final = pd.concat([X_num_val.reset_index(drop=True), X_cat_val_enc.reset_index(drop=True)], axis=1)

missing_cols = set(X_train_final.columns) - set(X_val_final.columns)
for col in missing_cols:
    X_val_final[col] = 0
X_val_final = X_val_final[X_train_final.columns]

# --- CAMBIO 3: Aplicar SMOTE al conjunto de entrenamiento después del preprocesamiento ---
print("\n--- Balanceando el conjunto de entrenamiento con SMOTE ---")
smote = SMOTE(random_state=42)
X_train_final_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)
print(f"Número de muestras antes de SMOTE: {len(y_train)}")
print(f"Distribución de clases antes de SMOTE:\n{y_train.value_counts(normalize=True)}")
print(f"Número de muestras después de SMOTE: {len(y_train_balanced)}")
print(f"Distribución de clases después de SMOTE:\n{y_train_balanced.value_counts(normalize=True)}")
print("----------------------------------------------------------\n")
# --------------------------------------------------------------------------------------


# Entrenar y comparar modelos
# CAMBIO 4: Usar class_weight='balanced' o scale_pos_weight en los modelos aplicables
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'), # <-- CAMBIO 4a
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'), # <-- CAMBIO 4b
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42,
                                 use_label_encoder=False, eval_metric='logloss',
                                 scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1]))), # <-- CAMBIO 4c
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42), # No tiene 'class_weight' directo
    'Naive Bayes': GaussianNB(), # No tiene 'class_weight' directo
    'KNN': KNeighborsClassifier(n_neighbors=5) # No tiene 'class_weight' directo
}

# MODIFICAR ESTA PARTE PARA RECOLECTAR PREDICCIONES Y PROBABILIDADES
results = []
predictions_dict = {}
probabilities_dict = {}

for name, model in models.items():
    # CAMBIO 5: Entrenar el modelo con los datos balanceados por SMOTE
    model.fit(X_train_final_balanced, y_train_balanced) # <-- CAMBIO 5
    y_pred = model.predict(X_val_final)
    
    # AGREGAR ESTAS LÍNEAS PARA RECOLECTAR DATOS
    predictions_dict[name] = y_pred
    
    # Calcular probabilidades si el modelo las soporta
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val_final)[:, 1]
        probabilities_dict[name] = y_proba
        roc_auc = roc_auc_score(y_val, y_proba)
    else:
        probabilities_dict[name] = None
        roc_auc = None
    
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    results.append({
        'Modelo': name,
        'Accuracy': round(accuracy_score(y_val, y_pred), 4),
        'Precision': round(precision_score(y_val, y_pred), 4),
        'Recall': round(recall_score(y_val, y_pred), 4),
        'F1-Score': round(f1_score(y_val, y_pred), 4),
        'ROC-AUC': round(roc_auc, 4) if roc_auc else 'N/A'
    })

results_df = pd.DataFrame(results)
print("\n--- Resultados de los Modelos (después de balanceo) ---")
print(results_df)
print("------------------------------------------------------\n")

# AGREGAR NOMBRES DE MODELOS
model_names = list(models.keys())

