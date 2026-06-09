import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Lectura de BD
df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')



class SurvivalAnalyzer:
    def __init__(self, df):
        self.df = df
        # Definir tipos de variables
        self.categorical_cols = [
            'dri_score', 'psych_disturb', 'cyto_score', 'diabetes',
            'tbi_status', 'arrhythmia', 'graft_type', 'renal_issue',
            'pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match',
            'rituximab', 'prod_type', 'conditioning_intensity', 'ethnicity',
            'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
            'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue',
            'sex_match', 'race_group', 'hepatic_mild', 'tce_div_match',
            'donor_related', 'melphalan_dose', 'cardiac', 'pulm_moderate'
        ]
        self.numerical_cols = [
            'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
            'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
            'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
            'hla_match_dqb1_low', 'year_hct', 'donor_age', 'age_at_hct',
            'comorbidity_score', 'karnofsky_score', 'efs_time'
        ]


    def encode_categorical_variables(self):
        """Codifica las variables categóricas en el DataFrame"""
        le = LabelEncoder()
        for col in self.categorical_cols:
            if col in self.df.columns:
                # Convertir a tipo string para evitar errores
                self.df[col] = le.fit_transform(self.df[col].astype(str))

    def analyze_basic_info(self):
        """Analiza información básica del dataset"""
        print("=== Información Básica del Dataset ===")
        print(f"\nNúmero total de pacientes: {len(self.df)}")
        print(f"Número de variables: {len(self.df.columns)}")

        # Análisis de valores faltantes
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Valores Faltantes': missing_data,
            'Porcentaje': missing_percent
        }).sort_values('Porcentaje', ascending=False)

        print("\nVariables con valores faltantes (>0%):")
        print(missing_summary[missing_summary['Valores Faltantes'] > 0])

    def analyze_survival_outcomes(self):
        """Analiza los resultados de supervivencia"""
        print("\n=== Análisis de Supervivencia ===")

        # Estadísticas básicas de supervivencia
        print("\nEstadísticas de tiempo de supervivencia (efs_time):")
        print(self.df['efs_time'].describe())

        # Distribución de eventos
        event_dist = self.df['efs'].value_counts(normalize=True) * 100
        print("\nDistribución de eventos (efs):")
        print(event_dist)

        # Visualizaciones
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Histograma de tiempos de supervivencia
        sns.histplot(data=self.df, x='efs_time', bins=50, ax=ax1)
        ax1.set_title('Distribución de Tiempos de Supervivencia')
        ax1.set_xlabel('Tiempo (días)')

        # Gráfico de eventos
        sns.countplot(data=self.df, x='efs', ax=ax2)
        ax2.set_title('Distribución de Eventos')

        plt.tight_layout()
        plt.show()

    def analyze_categorical_features(self):
        """Analiza las variables categóricas principales"""
        print("\n=== Análisis de Variables Categóricas ===")

        important_cats = ['dri_score', 'cyto_score', 'tbi_status',
                         'conditioning_intensity', 'donor_related']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, col in enumerate(important_cats):
            if col in self.df.columns:
                # Calcular proporción de eventos por categoría
                props = self.df.groupby(col)['efs'].mean()
                print(f"\nTasa de eventos por {col}:")
                print(props)

                # Visualizar
                sns.countplot(data=self.df, x=col, hue='efs', ax=axes[idx])
                axes[idx].set_title(f'Distribución de {col} por Evento')
                axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def analyze_numerical_features(self):
        """Analiza las variables numéricas principales"""
        print("\n=== Análisis de Variables Numéricas ===")

        important_nums = ['age_at_hct', 'donor_age', 'comorbidity_score',
                         'karnofsky_score', 'efs_time']

        # Estadísticas descriptivas
        print("\nEstadísticas descriptivas de variables numéricas principales:")
        print(self.df[important_nums].describe())

        # Visualizaciones
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, col in enumerate(important_nums):
            if col in self.df.columns:
                # Box plot por evento
                sns.boxplot(data=self.df, x='efs', y=col, ax=axes[idx])
                axes[idx].set_title(f'{col} por Evento')

        plt.tight_layout()
        plt.show()

        # Matriz de correlación
        corr_matrix = self.df[important_nums].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación - Variables Numéricas Principales')
        plt.show()

    def analyze_risk_factors(self):
        """Analiza los factores de riesgo separando variables categóricas y numéricas"""
        print("\n=== Análisis de Factores de Riesgo ===")

        correlations = []


        df_encoded = self.df.copy()  # Suponiendo que df ya está codificado

        for col in df_encoded.columns:
            if col not in ['ID', 'efs_time'] and pd.api.types.is_numeric_dtype(df_encoded[col]):
                corr = df_encoded[col].corr(df_encoded['efs'])
                correlations.append((col, abs(corr)))

        # Ordenar e imprimir correlaciones
        correlations.sort(key=lambda x: x[1], reverse=True)

        print("\nCorrelaciones numéricas más fuertes con el evento:")
        for col, corr in correlations[:5]:
            print(f"{col}: {corr:.3f}")


    def visualize_risk_factors(self):
        """Visualiza la relación entre factores de riesgo y supervivencia"""
        print("\n=== Visualización de Factores de Riesgo ===")

        for col in self.categorical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 5))
                sns.barplot(data=self.df, x=col, y='efs')
                plt.title(f'Tasa de Eventos por {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()


    def full_analysis(self):
        """Realiza el análisis completo del dataset"""
        print("======= Análisis Completo de Supervivencia Post-HCT =======")

        self.analyze_basic_info()
        self.analyze_survival_outcomes()
        self.analyze_categorical_features()
        self.analyze_numerical_features()
        self.analyze_risk_factors()
        self.visualize_risk_factors()


analyzer = SurvivalAnalyzer(df)


# Ejecutar el análisis completo
analyzer.full_analysis()


pip install pandas numpy scikit-learn matplotlib seaborn xgboost



import pandas as pd

# Ruta al archivo CSV
file_path = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path)




# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Identificar variables categóricas y numéricas
categorical_cols = [
    'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
    'arrhythmia', 'graft_type', 'renal_issue', 'pulm_severe', 'prim_disease_hct',
    'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type', 'conditioning_intensity',
    'ethnicity', 'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
    'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
    'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose',
    'cardiac', 'pulm_moderate'
]

numerical_cols = [
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6',
    'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6',
    'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low',
    'year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score', 'karnofsky_score', 'efs_time'
]

# Codificar variables categóricas
le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Separar características (X) y variable objetivo (y)
X = df[categorical_cols + numerical_cols]
y = df['efs']  # Variable objetivo (evento de supervivencia)

# Paso 1: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Paso 2: Manejo de valores faltantes
# Imputación para valores numéricos
num_imputer = SimpleImputer(strategy='mean')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Imputación para valores categóricos
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Paso 3: Escalar las variables numéricas
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Paso 4: Entrenar el modelo RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=20,        # Número de árboles en el bosque
    max_depth=2,             # Profundidad máxima de los árboles
    min_samples_split=10,     # Mínimo número de muestras para dividir un nodo
    random_state=8346         # Semilla para reproducibilidad
)
model.fit(X_train, y_train)

# Paso 5: Evaluar el modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Evento', 'Evento'], yticklabels=['No Evento', 'Evento'])
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()

# Paso 6: Identificar características importantes
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Visualizar las características más importantes
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Características más importantes')
plt.show()





