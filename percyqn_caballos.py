# 0. BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- FUNCIONES DE VISUALIZACIÓN ---
def visualizar_caracteristicas_numericas(df, features, output_path):
    os.makedirs(output_path, exist_ok=True)
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df[feature], color='skyblue')
            plt.title(f'Boxplot de {feature}')
            plt.ylabel(feature)
            plt.subplot(1, 2, 2)
            sns.histplot(df[feature].dropna(), kde=True, color='lightcoral', bins=30)
            plt.title(f'Distribución de {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'distribucion_boxplot_{feature}.png'))
            plt.close()
            print(f"  Visualización para '{feature}' guardada en {output_path}.")
        else:
            print(f"  Advertencia: La característica numérica '{feature}' no se encontró en el DataFrame.")

def visualizar_caracteristicas_categoricas(df, features, output_path):
    os.makedirs(output_path, exist_ok=True)
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=feature, data=df, palette='pastel', order=df[feature].value_counts(dropna=False).index)
            plt.title(f'Distribución de {feature}')
            plt.xlabel(feature)
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'distribucion_categorica_{feature}.png'))
            plt.close()
            print(f"  Visualización para '{feature}' guardada en {output_path}.")
        else:
            print(f"  Advertencia: La característica categórica '{feature}' no se encontró en el DataFrame.")

# --- FUNCIÓN DE LIMPIEZA DE OUTLIERS ---
def eliminar_outliers_iqr(df, features_numericas):
    df_limpio = df.copy()
    print("\nEliminando outliers usando el método IQR...")
    for feature in features_numericas:
        if feature in df_limpio.columns and pd.api.types.is_numeric_dtype(df_limpio[feature]):
            Q1 = df_limpio[feature].quantile(0.25)
            Q3 = df_limpio[feature].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            original_rows = len(df_limpio)
            df_limpio = df_limpio[(df_limpio[feature] >= limite_inferior) & (df_limpio[feature] <= limite_superior) | (df_limpio[feature].isnull())]
            rows_removed = original_rows - len(df_limpio)
            if rows_removed > 0:
                print(f"  Outliers eliminados de '{feature}': {rows_removed} filas.")
        elif feature not in df_limpio.columns:
             print(f"  Advertencia: La característica numérica '{feature}' para eliminación de outliers no se encontró.")
    return df_limpio

# --- FIN DE FUNCIONES ---

# 1. CARGA DE DATOS
print("1. Cargando datos...")
INPUT_PATH = '/kaggle/input/playground-series-s3e22/'
OUTPUT_PATH = '/kaggle/working/analisis_simple/'
VIZ_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'visualizaciones')

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VIZ_OUTPUT_PATH, exist_ok=True)

try:
    train_data_original = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
    test_data = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))
    test_ids = test_data['id'].copy()
    print(f"Datos cargados: {train_data_original.shape[0]} filas en entrenamiento, {test_data.shape[0]} filas en prueba.")
except FileNotFoundError:
    print(f"Error: Asegúrate de que 'train.csv' y 'test.csv' estén en el directorio {INPUT_PATH}")
    print("Asegúrate de haber añadido el conjunto de datos 'Playground Series S3E22' a tu notebook de Kaggle.")
    train_data_original = pd.DataFrame()
    test_data = pd.DataFrame()
    test_ids = pd.Series(dtype='int')

# Definir características numéricas y objetivo
numeric_features = ['rectal_temp', 'pulse', 'respiratory_rate', 'packed_cell_volume', 'total_protein']
TARGET_FEATURE = 'outcome'
ID_COLUMN = 'id'

# CORRECCIÓN: Identificar características categóricas dinámicamente
if not train_data_original.empty:
    all_columns = train_data_original.columns.tolist()
    # Excluir ID, objetivo y características numéricas ya definidas
    excluded_columns = [ID_COLUMN, TARGET_FEATURE] + numeric_features
    categorical_features = [col for col in all_columns if col not in excluded_columns]
    print(f"\nCaracterísticas numéricas identificadas: {numeric_features}")
    print(f"Características categóricas identificadas: {categorical_features}")
else:
    categorical_features = [] # Lista vacía si no se cargaron datos
    print("Advertencia: No se pudieron cargar los datos, lista de características categóricas está vacía.")

# 2. ANÁLISIS EXPLORATORIO BÁSICO (sobre datos originales)
print("\n2. Exploración de datos (sobre datos originales)...")

if not train_data_original.empty:
    if TARGET_FEATURE in train_data_original.columns:
        plt.figure()
        sns.countplot(x=TARGET_FEATURE, data=train_data_original, palette='viridis')
        plt.title('Distribución de la Variable Objetivo (Original)')
        plt.xlabel('Resultado')
        plt.ylabel('Cantidad')
        plt.savefig(os.path.join(VIZ_OUTPUT_PATH, 'distribucion_objetivo_original.png'))
        plt.close()
        print("  Visualización de la distribución del objetivo (original) guardada.")
    else:
        print(f"  Advertencia: La columna '{TARGET_FEATURE}' no se encontró para visualizar su distribución.")

    missing_values_train = train_data_original.isnull().sum()
    missing_percentage_train = (missing_values_train / len(train_data_original)) * 100
    print(f"\nValores faltantes en el conjunto de entrenamiento original:\n{missing_values_train[missing_values_train > 0]}")
    print(f"\nPorcentaje de valores faltantes en entrenamiento:\n{missing_percentage_train[missing_percentage_train > 0]}")

    plt.figure(figsize=(12, 6))
    sns.heatmap(train_data_original.isnull(), cbar=False, cmap='viridis')
    plt.title('Mapa de Calor de Valores Faltantes (Entrenamiento Original)')
    plt.savefig(os.path.join(VIZ_OUTPUT_PATH, 'missing_values_heatmap_train.png'))
    plt.close()
    print("  Mapa de calor de valores faltantes (entrenamiento) guardado.")

    if not test_data.empty:
        missing_values_test = test_data.isnull().sum()
        print(f"\nValores faltantes en el conjunto de prueba:\n{missing_values_test[missing_values_test > 0]}")

    print("\nGenerando visualizaciones de características...")
    # Filtrar características existentes antes de visualizar
    numeric_features_exist = [f for f in numeric_features if f in train_data_original.columns]
    categorical_features_exist = [f for f in categorical_features if f in train_data_original.columns]
    visualizar_caracteristicas_numericas(train_data_original, numeric_features_exist, VIZ_OUTPUT_PATH)
    visualizar_caracteristicas_categoricas(train_data_original, categorical_features_exist, VIZ_OUTPUT_PATH)
else:
    print("  No se pueden realizar análisis exploratorios porque train_data_original está vacío.")


# 3. PREPROCESAMIENTO
print("\n3. Preprocesando...")
if not train_data_original.empty and TARGET_FEATURE in train_data_original.columns and categorical_features:
    # Crear una copia para el preprocesamiento
    train_data = train_data_original.copy()

    # >>> INICIO: Eliminación de Outliers (solo en train_data) <<<
    print("Shape de train_data antes de eliminar outliers:", train_data.shape)
    train_data = eliminar_outliers_iqr(train_data, numeric_features_exist) # Usar solo las existentes
    print("Shape de train_data después de eliminar outliers:", train_data.shape)
    # >>> FIN: Eliminación de Outliers <<<

    # Re-alinear 'y' con 'train_data' después de eliminar outliers
    y = train_data[TARGET_FEATURE]
    train_data_features = train_data.drop(columns=[TARGET_FEATURE, ID_COLUMN], errors='ignore')

    # Asegurarse de que las listas de features solo contengan columnas existentes en train_data_features
    numeric_features_proc = [f for f in numeric_features if f in train_data_features.columns]
    categorical_features_proc = [f for f in categorical_features if f in train_data_features.columns]

    # Separar características numéricas y categóricas de train_data_features
    X_num_train = train_data_features[numeric_features_proc].copy()
    X_cat_train = train_data_features[categorical_features_proc].copy()

    # Hacer lo mismo para los datos de prueba
    if not test_data.empty:
        test_data_features = test_data.drop(columns=[ID_COLUMN], errors='ignore')
        # Asegurarse de que las listas de features solo contengan columnas existentes en test_data_features
        numeric_features_test = [f for f in numeric_features if f in test_data_features.columns]
        categorical_features_test = [f for f in categorical_features if f in test_data_features.columns]
        X_num_test = test_data_features[numeric_features_test].copy()
        X_cat_test = test_data_features[categorical_features_test].copy()
    else:
        X_num_test = pd.DataFrame(columns=numeric_features_proc)
        X_cat_test = pd.DataFrame(columns=categorical_features_proc)

    # Imputación
    print("  Imputando valores faltantes...")
    if not X_num_train.empty:
        num_imputer = SimpleImputer(strategy='median')
        X_num_train = pd.DataFrame(num_imputer.fit_transform(X_num_train), columns=numeric_features_proc, index=X_num_train.index)
        if not X_num_test.empty:
            X_num_test = pd.DataFrame(num_imputer.transform(X_num_test), columns=numeric_features_proc, index=X_num_test.index)

    if not X_cat_train.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_train = pd.DataFrame(cat_imputer.fit_transform(X_cat_train), columns=categorical_features_proc, index=X_cat_train.index)
        if not X_cat_test.empty:
            X_cat_test = pd.DataFrame(cat_imputer.transform(X_cat_test), columns=categorical_features_proc, index=X_cat_test.index)

    # Codificación One-Hot
    print("  Codificando características categóricas...")
    if not X_cat_train.empty:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_train_enc = pd.DataFrame(encoder.fit_transform(X_cat_train), columns=encoder.get_feature_names_out(categorical_features_proc), index=X_cat_train.index)
        if not X_cat_test.empty:
            X_cat_test_enc = pd.DataFrame(encoder.transform(X_cat_test), columns=encoder.get_feature_names_out(categorical_features_proc), index=X_cat_test.index)
        else:
            X_cat_test_enc = pd.DataFrame(columns=encoder.get_feature_names_out(categorical_features_proc))
    else:
        X_cat_train_enc = pd.DataFrame(index=X_num_train.index) # DataFrame vacío con índice correcto si no hay categóricas
        X_cat_test_enc = pd.DataFrame(index=X_num_test.index)  # DataFrame vacío con índice correcto si no hay categóricas

    # Concatenar características numéricas y categóricas codificadas
    print("  Concatenando características procesadas...")
    X_train_full = pd.concat([X_num_train, X_cat_train_enc], axis=1)

    # Manejo de X_test_full
    if not X_num_test.empty or not X_cat_test_enc.empty:
        num_test_processed = X_num_test if not X_num_test.empty else pd.DataFrame(index=X_cat_test_enc.index if not X_cat_test_enc.empty else None)
        cat_test_processed = X_cat_test_enc if not X_cat_test_enc.empty else pd.DataFrame(index=X_num_test.index if not X_num_test.empty else None)
        X_test_full = pd.concat([num_test_processed, cat_test_processed], axis=1)
    else:
        X_test_full = pd.DataFrame(columns=X_train_full.columns)

    # Asegurar que las columnas de X_test_full coincidan con X_train_full
    train_cols = X_train_full.columns
    if not X_test_full.empty:
        test_cols = X_test_full.columns
        missing_cols_test = set(train_cols) - set(test_cols)
        for c in missing_cols_test:
            X_test_full[c] = 0
        extra_cols_test = set(test_cols) - set(train_cols)
        X_test_full = X_test_full.drop(columns=list(extra_cols_test), errors='ignore')
        X_test_full = X_test_full[train_cols]
    else:
        X_test_full = pd.DataFrame(columns=train_cols)

    print(f"Dimensiones finales: X_train_full: {X_train_full.shape}, X_test_full: {X_test_full.shape}, y: {y.shape}")

    # 4. DIVISIÓN
    print("\n4. Dividiendo datos para validación...")
    if X_train_full.shape[0] != len(y):
        print(f"Error: Desajuste en el número de filas entre X_train_full ({X_train_full.shape[0]}) y y ({len(y)}).")
        modelos = {}
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, stratify=y, random_state=42)
        print(f"Datos divididos: Entrenamiento: {X_train.shape}, Validación: {X_val.shape}")

        # 5. MODELOS
        print("\n5. Entrenando modelos...")
        modelos = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42),
            'NaiveBayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        results = {}
        for nombre, modelo in modelos.items():
            print(f"Entrenando modelo: {nombre}")
            try:
                modelo.fit(X_train, y_train)
                y_pred_val = modelo.predict(X_val)
                acc = accuracy_score(y_val, y_pred_val)
                results[nombre] = acc
                print(f"  Accuracy {nombre} (Validación): {acc:.4f}")
                labels_present = np.unique(np.concatenate((y_val.astype(str), y_pred_val.astype(str))))
                print(f"  Reporte de Clasificación - {nombre} (Validación):\n{classification_report(y_val, y_pred_val, labels=labels_present, zero_division=0)}")

                cm = confusion_matrix(y_val, y_pred_val, labels=modelo.classes_)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=modelo.classes_, yticklabels=modelo.classes_)
                plt.title(f'Matriz de Confusión (Validación): {nombre}')
                plt.xlabel('Predicción')
                plt.ylabel('Valor real')
                plt.tight_layout()
                plt.savefig(os.path.join(VIZ_OUTPUT_PATH, f'confusion_val_{nombre}.png'))
                plt.close()
                print(f"  Matriz de confusión para '{nombre}' guardada.")
            except Exception as e:
                print(f"  Error entrenando o evaluando el modelo {nombre}: {e}")

        # 6. PREDICCIÓN FINAL CON MEJOR MODELO
        print("\n6. Realizando predicción final con el mejor modelo...")
        if results:
            mejor_modelo_nombre = max(results, key=results.get)
            print(f"Mejor modelo según Accuracy de validación: {mejor_modelo_nombre} ({results[mejor_modelo_nombre]:.4f})")
            mejor_modelo = modelos[mejor_modelo_nombre]

            print(f"Re-entrenando {mejor_modelo_nombre} con todos los datos de entrenamiento procesados (X_train_full, y)...")
            if X_train_full.shape[0] == len(y):
                mejor_modelo.fit(X_train_full, y)
                print("  Modelo re-entrenado.")

                if not X_test_full.empty:
                    print("  Generando predicciones en el conjunto de prueba...")
                    test_pred = mejor_modelo.predict(X_test_full)
                    submission = pd.DataFrame({'id': test_ids, TARGET_FEATURE: test_pred})
                    submission_path = os.path.join(OUTPUT_PATH, 'submission.csv')
                    submission.to_csv(submission_path, index=False)
                    print(f"Archivo de sumisión guardado en: {submission_path}")
                else:
                    print("  No se pueden generar predicciones porque X_test_full está vacío o no se pudo procesar.")

                # 7. GUARDADO DEL MEJOR MODELO
                print("\n7. Guardando el mejor modelo entrenado...")
                mejor_modelo_path = os.path.join(OUTPUT_PATH, 'mejor_modelo.pkl')
                try:
                    with open(mejor_modelo_path, 'wb') as f:
                        pickle.dump(mejor_modelo, f)
                    print(f"Mejor modelo guardado en: {mejor_modelo_path}")
                except Exception as e:
                    print(f"  Error guardando el modelo: {e}")
            else:
                print("  Error: Desajuste entre X_train_full e y antes del re-entrenamiento final. Modelo no guardado.")
        else:
            print("  No se entrenaron modelos con éxito, no se puede seleccionar el mejor modelo ni generar sumisión.")

        print("\nEntrenamiento y análisis completos.")
else:
    print("\nNo se pudo continuar con el preprocesamiento y entrenamiento debido a datos de entrenamiento originales faltantes, columna objetivo faltante o ninguna característica categórica identificada.")



