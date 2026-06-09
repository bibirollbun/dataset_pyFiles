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


# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


!pip install ydata-profiling -q
from ydata_profiling import ProfileReport


# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Cargar los datos
train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
print(" Datos cargados correctamente")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


#MUY GRANDE EL DATASET XD SE ME TRABA
# Cargar los datos
train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')

# MUESTREO ESTRATIFICADO para trabajar más rápido
# Esto mantiene la proporción del target 
from sklearn.model_selection import train_test_split

# Tomar 100,000 filas manteniendo la proporción del target
train_sample, _ = train_test_split(
    train, 
    train_size=100000, 
    stratify=train['target'], 
    random_state=42
)

test_sample = test.sample(n=50000, random_state=42)


# los datasets REDUCIDOS
train = train_sample.copy()
test = test_sample.copy()

print("Datos cargados correctamente")
print(f"Train shape (muestra): {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Proporción target en muestra: {train['target'].mean():.2%}")


#Información básica
print("="*50)
print("INFORMACIÓN DEL DATASET")
print("="*50)
train.info()


#Primeras filas
train.head()


#Estadísticas descriptivas
train.describe().T


#Analizar el TARGET 
print("\n" + "="*50)
print("ANÁLISIS DEL TARGET")
print("="*50)
target_dist = train['target'].value_counts()
print(target_dist)
print(f"\nProporción de reclamaciones: {train['target'].mean():.2%}")



#Visualización del target
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
train['target'].value_counts().plot(kind='bar', ax=ax[0], color=['green', 'red'])
ax[0].set_title('Distribución del Target')
ax[0].set_xlabel('Target (0=No claim, 1=Claim)')
ax[0].set_ylabel('Frecuencia')

train['target'].value_counts(normalize=True).plot(kind='pie', ax=ax[1], autopct='%1.1f%%')
ax[1].set_title('Proporción del Target')
plt.tight_layout()
plt.show()


#Identificar valores faltantes (-1)
print("\n" + "="*50)
print("VALORES FALTANTES (representados como -1)")
print("="*50)

# Contar -1 por columna
missing_counts = (train == -1).sum()
missing_percent = (missing_counts / len(train) * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts[missing_counts > 0],
    'Percentage': missing_percent[missing_percent > 0]
})
print(missing_df.head(10))


# Visualizar las 10 columnas con más valores faltantes
plt.figure(figsize=(12, 6))
missing_percent[missing_percent > 0].head(10).plot(kind='barh')
plt.title('Top 10 Features con Valores Faltantes (-1)')
plt.xlabel('Porcentaje de valores faltantes')
plt.show()


print("\n" + "="*70)
print(" ANÁLISIS DETALLADO DE FEATURES")
print("="*70)

# Separar tipos de features
bin_features = [col for col in train.columns if '_bin' in col]
cat_features = [col for col in train.columns if '_cat' in col]
num_features = [col for col in train.columns if col not in bin_features + cat_features + ['id', 'target']]

print(f"\n Tipos de características:")
print(f"  - Binarias: {len(bin_features)}")
print(f"  - Categóricas: {len(cat_features)}")
print(f"  - Numéricas: {len(num_features)}")


# 1. ANÁLISIS DE FEATURES BINARIAS
# ------------------------------------------------------------
print("\n" + "-"*70)
print("1️  ANÁLISIS DE FEATURES BINARIAS")
print("-"*70)

# Mostrar distribución de TODAS las features binarias
bin_distribution = pd.DataFrame({
    'Feature': bin_features,
    'Zeros': [train[col].value_counts().get(0, 0) for col in bin_features],
    'Ones': [train[col].value_counts().get(1, 0) for col in bin_features],
    'Ratio_1s': [train[col].mean() for col in bin_features]
})
bin_distribution['Balance'] = bin_distribution['Ratio_1s'].apply(
    lambda x: 'Balanceado' if 0.3 <= x <= 0.7 else 'Desbalanceado'
)
bin_distribution = bin_distribution.sort_values('Ratio_1s', ascending=False)

print("\n Distribución de Features Binarias:")
print(bin_distribution.to_string(index=False))


# Visualizar distribución de binarias
fig, axes = plt.subplots(3, 6, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(bin_features):
    if idx < len(axes):
        counts = train[col].value_counts()
        axes[idx].bar([0, 1], [counts.get(0, 0), counts.get(1, 0)], 
                      color=['#e74c3c', '#2ecc71'])
        axes[idx].set_title(col, fontsize=9)
        axes[idx].set_xlabel('Valor')
        axes[idx].set_ylabel('Frecuencia')
        axes[idx].set_xticks([0, 1])

# Ocultar ejes sobrantes
for idx in range(len(bin_features), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribución de TODAS las Features Binarias', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 2. ANÁLISIS DE FEATURES CATEGÓRICAS
# ------------------------------------------------------------
print("\n" + "-"*70)
print("  ANÁLISIS DE FEATURES CATEGÓRICAS")
print("-"*70)

# Cardinalidad y estadísticas
cat_stats = pd.DataFrame({
    'Feature': cat_features,
    'Cardinality': [train[col].nunique() for col in cat_features],
    'Missing_Pct': [(train[col] == -1).sum() / len(train) * 100 for col in cat_features],
    'Most_Common': [train[col].mode()[0] if len(train[col].mode()) > 0 else None for col in cat_features],
    'Most_Common_Freq': [train[col].value_counts().iloc[0] / len(train) * 100 for col in cat_features]
})
cat_stats = cat_stats.sort_values('Cardinality', ascending=False)

print("\n Estadísticas de Features Categóricas:")
print(cat_stats.to_string(index=False))


# Clasificar por cardinalidad
cat_stats['Cardinality_Type'] = cat_stats['Cardinality'].apply(
    lambda x: 'Baja (≤10)' if x <= 10 else ('Media (11-50)' if x <= 50 else 'Alta (>50)')
)

print("\n Clasificación por Cardinalidad:")
print(cat_stats['Cardinality_Type'].value_counts())


# Visualizar cardinalidad
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Cardinalidad
cat_stats.plot(x='Feature', y='Cardinality', kind='barh', ax=ax1, 
               color='steelblue', legend=False)
ax1.set_title('Cardinalidad de Features Categóricas', fontsize=12, fontweight='bold')
ax1.set_xlabel('Número de Categorías Únicas')
ax1.invert_yaxis()

# Gráfico 2: Missing values
cat_stats.plot(x='Feature', y='Missing_Pct', kind='barh', ax=ax2, 
               color='coral', legend=False)
ax2.set_title('Porcentaje de Valores Faltantes', fontsize=12, fontweight='bold')
ax2.set_xlabel('Porcentaje (%)')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()




#  ANÁLISIS DE FEATURES NUMÉRICAS
# ------------------------------------------------------------
print("\n" + "-"*70)
print(" ANÁLISIS DE FEATURES NUMÉRICAS")
print("-"*70)

# Estadísticas descriptivas
num_stats = train[num_features].describe().T
num_stats['skewness'] = train[num_features].skew()
num_stats['kurtosis'] = train[num_features].kurtosis()
num_stats['missing_pct'] = (train[num_features] == -1).sum() / len(train) * 100

print("\n Estadísticas Descriptivas de Features Numéricas:")
print(num_stats[['mean', 'std', 'min', 'max', 'skewness', 'missing_pct']].round(3))


# Identificar features con alta asimetría 
high_skew = num_stats[abs(num_stats['skewness']) > 2].sort_values('skewness', ascending=False)
print(f"\n  Features con Alta Asimetría (|skewness| > 2): {len(high_skew)}")
if len(high_skew) > 0:
    print(high_skew[['skewness']].head(10))

# Visualizar distribuciones de features numéricas clave
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(num_features[:9]):  # Primeras 9
    train[col].hist(bins=50, ax=axes[idx], color='teal', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}\nSkew: {train[col].skew():.2f}', fontsize=10)
    axes[idx].set_xlabel('Valor')
    axes[idx].set_ylabel('Frecuencia')

plt.suptitle('Distribuciones de Features Numéricas (Top 9)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#________________________________________________________________________________________
# Correlación con el target (para variables numéricas)
correlations = train[num_features + ['target']].corr()['target'].drop('target').sort_values(ascending=False)
print("\n Top 10 correlaciones con el target:")
print(correlations.head(10))

plt.figure(figsize=(10, 8))
correlations.head(15).plot(kind='barh')
plt.title('Top 15 Features - Correlación con Target')
plt.xlabel('Correlación')
plt.tight_layout()
plt.show()


# VISUALIZACIONES: FEATURES VS TARGET
# ============================================================

print("\n" + "="*70)
print(" ANÁLISIS DE FEATURES VS TARGET")
print("="*70)

# ------------------------------------------------------------
# GRÁFICO 1: Top Features Numéricas vs Target (Boxplots)
# ------------------------------------------------------------
print("\n  Boxplots: Top 6 Features Numéricas vs Target")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
top_features = correlations.head(6).index

for idx, col in enumerate(top_features):
    ax = axes[idx // 3, idx % 3]
    
    # Crear boxplot separado por target
    data_to_plot = [
        train[train['target'] == 0][col],
        train[train['target'] == 1][col]
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['No Claim (0)', 'Claim (1)'],
                    patch_artist=True, showmeans=True)
    
    # Colorear boxes
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_title(f'{col}\nCorr: {correlations[col]:.4f}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)

plt.suptitle('Top 6 Features Numéricas vs Target', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


# GRÁFICO 2: Tasa de Reclamación por Features Categóricas
# ------------------------------------------------------------
print("\n  Tasa de Reclamación por Categorías (Top 4 Features Categóricas)")

# Seleccionar 4 categóricas con menos missing values
cat_for_analysis = cat_stats[cat_stats['Missing_Pct'] < 10].head(4)['Feature'].tolist()

if len(cat_for_analysis) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(cat_for_analysis):
        # Calcular tasa de reclamación por categoría
        target_rate = train.groupby(col)['target'].agg(['mean', 'count']).reset_index()
        target_rate = target_rate[target_rate['count'] > 100]  # Filtrar categorías con pocos datos
        target_rate = target_rate.sort_values('mean', ascending=False).head(15)
        
        # Crear gráfico de barras
        bars = axes[idx].bar(range(len(target_rate)), target_rate['mean'], 
                             color='coral', edgecolor='black')
        axes[idx].set_xticks(range(len(target_rate)))
        axes[idx].set_xticklabels(target_rate[col], rotation=45, ha='right')
        axes[idx].set_title(f'Tasa de Reclamación - {col}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Tasa de Reclamación')
        axes[idx].axhline(y=train['target'].mean(), color='red', linestyle='--', 
                         label=f'Media Global: {train["target"].mean():.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Tasa de Reclamación por Categorías (Top 15 Categorías)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print(" No hay suficientes features categóricas con <10% missing para análisis detallado")



# Gráfico 3: Heatmap de correlaciones entre top features 
plt.figure(figsize=(10, 8)) 
top_10_features = correlations.head(10).index.tolist() 
corr_matrix = train[top_10_features + ['target']].corr() 
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlaciones entre Top 10 Features y Target')
plt.tight_layout() 
plt.show()


# GRÁFICO 4: Features Binarias vs Target
# ------------------------------------------------------------
print("\n  Tasa de Reclamación para Features Binarias")

# Calcular tasa de reclamación para cada valor de las binarias
bin_target_rates = []
for col in bin_features:
    rate_0 = train[train[col] == 0]['target'].mean()
    rate_1 = train[train[col] == 1]['target'].mean()
    diff = abs(rate_1 - rate_0)
    bin_target_rates.append({
        'Feature': col,
        'Rate_0': rate_0,
        'Rate_1': rate_1,
        'Difference': diff
    })

bin_target_df = pd.DataFrame(bin_target_rates).sort_values('Difference', ascending=False)

# Visualizar top 10 binarias con mayor diferencia
fig, ax = plt.subplots(figsize=(14, 6))
top_10_bin = bin_target_df.head(10)

x = np.arange(len(top_10_bin))
width = 0.35

bars1 = ax.bar(x - width/2, top_10_bin['Rate_0'], width, label='Valor = 0', color='#2ecc71')
bars2 = ax.bar(x + width/2, top_10_bin['Rate_1'], width, label='Valor = 1', color='#e74c3c')

ax.set_xlabel('Feature')
ax.set_ylabel('Tasa de Reclamación')
ax.set_title('Top 10 Features Binarias: Tasa de Reclamación por Valor', 
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_10_bin['Feature'], rotation=45, ha='right')
ax.axhline(y=train['target'].mean(), color='blue', linestyle='--', 
           label=f'Media Global: {train["target"].mean():.3f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n Top 10 Features Binarias con Mayor Diferencia en Tasa de Reclamación:")
print(bin_target_df.head(10).to_string(index=False))


# ANÁLISIS DE INTERACCIONES ENTRE FEATURES
# ============================================================

print("\n" + "="*70)
print(" ANÁLISIS DE INTERACCIONES ENTRE FEATURES")
print("="*70)

# Crear interacciones entre las top features
top_5_features = correlations.head(5).index.tolist()

interaction_results = []

print("\n Creando y evaluando interacciones...")
for i in range(len(top_5_features)):
    for j in range(i+1, len(top_5_features)):
        feat1 = top_5_features[i]
        feat2 = top_5_features[j]
        
        # Crear 3 tipos de interacciones
        # 1. Multiplicación
        interaction_name_mult = f'{feat1}_x_{feat2}'
        train[interaction_name_mult] = train[feat1] * train[feat2]
        corr_mult = train[[interaction_name_mult, 'target']].corr().iloc[0, 1]
        
        # 2. Suma
        interaction_name_sum = f'{feat1}_+_{feat2}'
        train[interaction_name_sum] = train[feat1] + train[feat2]
        corr_sum = train[[interaction_name_sum, 'target']].corr().iloc[0, 1]
        
        # 3. División (con manejo de divisiones por cero)
        interaction_name_div = f'{feat1}_/_{feat2}'
        train[interaction_name_div] = train[feat1] / (train[feat2] + 1e-5)
        corr_div = train[[interaction_name_div, 'target']].corr().iloc[0, 1]
        
        interaction_results.append({
            'Interaction': interaction_name_mult,
            'Type': 'Multiplicación',
            'Correlation': corr_mult
        })
        interaction_results.append({
            'Interaction': interaction_name_sum,
            'Type': 'Suma',
            'Correlation': corr_sum
        })
        interaction_results.append({
            'Interaction': interaction_name_div,
            'Type': 'División',
            'Correlation': corr_div
        })

# Convertir a DataFrame
interaction_df = pd.DataFrame(interaction_results)
interaction_df = interaction_df.sort_values('Correlation', key=abs, ascending=False)

print("\n Top 15 Interacciones Más Correlacionadas con Target:")
print(interaction_df.head(15).to_string(index=False))


# Comparar con correlaciones originales
print("\n Comparación con Features Originales:")
print(f"  - Mejor correlación original: {correlations.max():.4f} ({correlations.idxmax()})")
print(f"  - Mejor correlación interacción: {interaction_df['Correlation'].abs().max():.4f}")
print(f"    ({interaction_df.loc[interaction_df['Correlation'].abs().idxmax(), 'Interaction']})")



# Visualizar top 10 interacciones
fig, ax = plt.subplots(figsize=(14, 6))
top_10_interactions = interaction_df.head(10)
colors = ['#3498db' if x == 'Multiplicación' else '#e74c3c' if x == 'Suma' else '#f39c12' 
          for x in top_10_interactions['Type']]

ax.barh(range(len(top_10_interactions)), top_10_interactions['Correlation'], color=colors)
ax.set_yticks(range(len(top_10_interactions)))
ax.set_yticklabels(top_10_interactions['Interaction'], fontsize=9)
ax.set_xlabel('Correlación con Target')
ax.set_title('Top 10 Interacciones de Features', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Leyenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Multiplicación'),
    Patch(facecolor='#e74c3c', label='Suma'),
    Patch(facecolor='#f39c12', label='División')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

print("\n Análisis de interacciones completado")
print(f" Total de interacciones creadas: {len(interaction_results)}")


print(" Generando reporte exploratorio ...")

# Muestra  pequeña solo para el reporte PORQUE SE ME TRABA KAGGLE XD
train_profile = train.sample(n=5000, random_state=42)

profile = ProfileReport(
    train_profile, 
    title="Reporte Exploratorio - Porto Seguro",
    explorative=True,
    minimal=True  # Cambiar a True si sigue lento
)
profile.to_notebook_iframe()


profile.to_file("reporte_exploratorio.html")


# ============================================================================
# PASO 3: FEATURE ENGINEERING
# ============================================================================

print("="*70)
print(" PREPARACIÓN PARA FEATURE ENGINEERING")
print("="*70)

# ----------------------------------------------------------------------------
# 1. RECARGAR DATOS LIMPIOS (sin transformaciones del EDA)
# ----------------------------------------------------------------------------
print("\n Recargando datos originales...")

train_original = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test_original = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')

# Aplicar muestreo estratificado (mismo que en EDA)
from sklearn.model_selection import train_test_split

train, _ = train_test_split(
    train_original,
    train_size=100000,
    stratify=train_original['target'],
    random_state=42  # Mismo seed para reproducibilidad
)

test = test_original.sample(n=50000, random_state=42)

print(f" Datos recargados:")
print(f"   - Train: {train.shape}")
print(f"   - Test: {test.shape}")
print(f"   - Proporción target: {train['target'].mean():.2%}")


# ----------------------------------------------------------------------------
# 2. SEPARAR COMPONENTES
# ----------------------------------------------------------------------------
print("\n Separando IDs, target y features...")

# Guardar IDs (para submission final)
train_ids = train['id'].copy()
test_ids = test['id'].copy()

# Guardar target
y_train = train['target'].copy()

# Crear DataFrames solo con features (X)
X_train = train.drop(['id', 'target'], axis=1).copy()
X_test = test.drop(['id'], axis=1).copy()

print(f" Separación completa:")
print(f"   - X_train: {X_train.shape}")
print(f"   - y_train: {y_train.shape}")
print(f"   - X_test: {X_test.shape}")


#----------------------------------------------------------------------------
# 3. VERIFICACIONES DE INTEGRIDAD
# ----------------------------------------------------------------------------
print("\n Verificando integridad de datos...")

# Verificar dimensiones correctas
assert X_train.shape == (100000, 57), f" X_train debería ser (100000, 57), es {X_train.shape}"
assert X_test.shape == (50000, 57), f" X_test debería ser (50000, 57), es {X_test.shape}"
assert y_train.shape == (100000,), f" y_train debería ser (100000,), es {y_train.shape}"
assert len(train_ids) == 100000, f" train_ids debería ser 100000, es {len(train_ids)}"
assert len(test_ids) == 50000, f" test_ids debería ser 50000, es {len(test_ids)}"

# Verificar que no hay data leakage (columnas extra en test)
assert set(X_train.columns) == set(X_test.columns), " Columnas diferentes en train y test"

# Verificar proporción del target
assert 0.035 <= y_train.mean() <= 0.038, f" Proporción target sospechosa: {y_train.mean():.2%}"

print(" Todas las verificaciones pasaron correctamente")


# ----------------------------------------------------------------------------
# 4. IDENTIFICAR TIPOS DE FEATURES
# ----------------------------------------------------------------------------
print("\n Identificando tipos de features...")

# Identificar features por tipo
bin_features = [col for col in X_train.columns if '_bin' in col]
cat_features = [col for col in X_train.columns if '_cat' in col]
num_features = [col for col in X_train.columns 
                if col not in bin_features + cat_features]

print(f" Features identificadas:")
print(f"   - Binarias: {len(bin_features)}")
print(f"   - Categóricas: {len(cat_features)}")
print(f"   - Numéricas: {len(num_features)}")
print(f"   - Total: {len(bin_features) + len(cat_features) + len(num_features)}")

# Mostrar nombres
print(f"\n Binarias: {bin_features[:5]}... ({len(bin_features)} total)")
print(f"   Categóricas: {cat_features[:5]}... ({len(cat_features)} total)")
print(f"   Numéricas: {num_features[:5]}... ({len(num_features)} total)")

print("\n" + "="*70)
print(" PREPARACIÓN COMPLETA - LISTO PARA FEATURE ENGINEERING")
print("="*70)


 #============================================================================
#  CREACIÓN DE FEATURES: MISSING COUNT INDICATOR
# ============================================================================

print("\n" + "="*70)
print(" FEATURE 1: INDICADOR DE VALORES FALTANTES")
print("="*70)

# Contar valores faltantes por fila (-1 representa missing)
X_train['missing_count'] = (X_train == -1).sum(axis=1)
X_test['missing_count'] = (X_test == -1).sum(axis=1)

print(f" Feature 'missing_count' creada")
print(f"\n Estadísticas de missing_count:")
print(X_train['missing_count'].describe())



# ----------------------------------------------------------------------------
# VISUALIZACIÓN
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Distribución de missing count
axes[0].hist(X_train['missing_count'], bins=range(0, 10), 
             color='coral', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribución de Missing Count', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Número de valores faltantes por fila')
axes[0].set_ylabel('Frecuencia')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Missing count vs Target
missing_by_target = pd.DataFrame({
    'No Claim': X_train.loc[y_train == 0, 'missing_count'],
    'Claim': X_train.loc[y_train == 1, 'missing_count']
})

bp = missing_by_target.plot(
    kind='box',
    ax=axes[1],
    patch_artist=True,
    color=dict(boxes='lightgreen', whiskers='green', medians='red', caps='green')
)
axes[1].set_title('Missing Count vs Target', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Missing Count')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------
# ANÁLISIS DE CORRELACIÓN
# ----------------------------------------------------------------------------
correlation_missing = pd.concat(
    [X_train['missing_count'], y_train], 
    axis=1
).corr().iloc[0, 1]

print(f"\n Correlación con target: {correlation_missing:.4f}")

# Interpretación
if abs(correlation_missing) < 0.01:
    print("     Correlación muy débil - Feature poco predictiva")
elif abs(correlation_missing) < 0.05:
    print("    Correlación débil pero puede aportar en ensemble")
else:
    print("    Correlación significativa - Feature útil")

print("\n" + "="*70)


# ============================================================================
# 3.2 ELIMINACIÓN DE FEATURES PROBLEMÁTICAS
# ============================================================================

print("\n" + "="*70)
print("ELIMINACIÓN DE FEATURES PROBLEMÁTICAS")
print("="*70)

# ----------------------------------------------------------------------------
# Features a eliminar basadas en EDA
# ----------------------------------------------------------------------------

# 1. Features con exceso de missing values (>40%)
high_missing_features = ['ps_car_03_cat', 'ps_car_05_cat']

# 2. Features con muy bajo poder predictivo (opcional)
# low_predictive_features = []  # Puedes agregar aquí si decides eliminar más

# Combinar todas las features a eliminar
features_to_drop = high_missing_features

print(f"\n Features a eliminar: {len(features_to_drop)}")
for feat in features_to_drop:
    missing_pct = (X_train[feat] == -1).sum() / len(X_train) * 100
    print(f"   - {feat}: {missing_pct:.2f}% missing")

# ----------------------------------------------------------------------------
# Eliminar features
# ----------------------------------------------------------------------------

print(f"\n Eliminando features...")
print(f"   Antes: X_train {X_train.shape}, X_test {X_test.shape}")

X_train = X_train.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)

print(f"   Después: X_train {X_train.shape}, X_test {X_test.shape}")

# Actualizar listas de features categóricas
cat_features = [col for col in cat_features if col not in features_to_drop]

print(f"\n Features categóricas actualizadas: {len(cat_features)}")
print(f"   {cat_features}")

# ----------------------------------------------------------------------------
# Verificación
# ----------------------------------------------------------------------------

assert X_train.shape[1] == X_test.shape[1], " Train y test tienen diferente número de columnas"
assert X_train.shape[1] == 57 - len(features_to_drop) + 1, f" Número de columnas incorrecto"  # 57 originales - 2 eliminadas + 1 missing_count

print("\n" + "="*70)
print(f" ELIMINACIÓN COMPLETA: {X_train.shape[1]} features restantes")
print("="*70)


# ============================================================================
#  IMPUTACIÓN DE VALORES FALTANTES
# ============================================================================

print("\n" + "="*70)
print(" PASO 3: IMPUTACIÓN DE VALORES FALTANTES")
print("="*70)

# Identificar features con missing values (-1)
features_with_missing = []
for col in X_train.columns:
    missing_count = (X_train[col] == -1).sum()
    if missing_count > 0:
        missing_pct = missing_count / len(X_train) * 100
        features_with_missing.append({
            'feature': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'type': 'categorical' if col in cat_features else 'numerical'
        })

missing_df = pd.DataFrame(features_with_missing).sort_values('missing_pct', ascending=False)

print(f"\n Features con valores faltantes: {len(missing_df)}")
print(missing_df.to_string(index=False))


# ----------------------------------------------------------------------------
# Imputar CATEGÓRICAS con MODA
# ----------------------------------------------------------------------------
print(f"\n  Imputando features CATEGÓRICAS...")

cat_with_missing = missing_df[missing_df['type'] == 'categorical']['feature'].tolist()

for col in cat_with_missing:
    # Calcular moda (excluyendo -1)
    mode_value = X_train[X_train[col] != -1][col].mode()[0]
    
    # Imputar
    X_train.loc[X_train[col] == -1, col] = mode_value
    X_test.loc[X_test[col] == -1, col] = mode_value
    
    print(f"    {col}: imputado con moda = {mode_value}")


#----------------------------------------------------------------------------
# Imputar NUMÉRICAS con MEDIANA
# ----------------------------------------------------------------------------
print(f"\n Imputando features NUMÉRICAS...")

num_with_missing = missing_df[missing_df['type'] == 'numerical']['feature'].tolist()

for col in num_with_missing:
    # Calcular mediana (excluyendo -1)
    median_value = X_train[X_train[col] != -1][col].median()
    
    # Imputar
    X_train.loc[X_train[col] == -1, col] = median_value
    X_test.loc[X_test[col] == -1, col] = median_value
    
    print(f"    {col}: imputado con mediana = {median_value:.4f}")



# ----------------------------------------------------------------------------
# Verificar que no quedan valores -1
# ----------------------------------------------------------------------------
remaining_missing_train = (X_train == -1).sum().sum()
remaining_missing_test = (X_test == -1).sum().sum()

print(f"\n Verificación:")
print(f"   - Missing en train: {remaining_missing_train}")
print(f"   - Missing en test: {remaining_missing_test}")

if remaining_missing_train == 0 and remaining_missing_test == 0:
    print("    Todos los valores faltantes han sido imputados")
else:
    print(f"     Aún quedan {remaining_missing_train + remaining_missing_test} valores faltantes")

print("\n" + "="*70)
print(" IMPUTACIÓN COMPLETA")
print("="*70)


# ============================================================================
#  CREACIÓN DE INTERACCIONES
# ============================================================================

print("\n" + "="*70)
print(" PASO 4: CREACIÓN DE FEATURES DE INTERACCIÓN")
print("="*70)

# Top features para interacciones (de tu EDA)
top_features_for_interaction = ['ps_car_13', 'ps_reg_02', 'ps_car_12', 'ps_reg_03', 'ps_car_15']

interaction_count = 0

# Crear interacciones
for i in range(len(top_features_for_interaction)):
    for j in range(i+1, len(top_features_for_interaction)):
        feat1 = top_features_for_interaction[i]
        feat2 = top_features_for_interaction[j]
        
        # Multiplicación
        new_feat_mult = f'{feat1}_x_{feat2}'
        X_train[new_feat_mult] = X_train[feat1] * X_train[feat2]
        X_test[new_feat_mult] = X_test[feat1] * X_test[feat2]
        interaction_count += 1
        
        # División (con protección)
        new_feat_div = f'{feat1}_div_{feat2}'
        X_train[new_feat_div] = X_train[feat1] / (X_train[feat2] + 1e-5)
        X_test[new_feat_div] = X_test[feat1] / (X_test[feat2] + 1e-5)
        interaction_count += 1

print(f" {interaction_count} features de interacción creadas")




# Evaluar correlaciones
new_interaction_features = [col for col in X_train.columns if '_x_' in col or '_div_' in col]
interaction_corrs = pd.concat([X_train[new_interaction_features], y_train], axis=1).corr()['target'].drop('target')
interaction_corrs = interaction_corrs.sort_values(ascending=False, key=abs)

print(f"\n Top 5 Interacciones más correlacionadas:")
print(interaction_corrs.head(5))

print("\n" + "="*70)
print(f" INTERACCIONES COMPLETAS: {X_train.shape[1]} features totales")
print("="*70)


# ============================================================================
#  ENCODING DE VARIABLES CATEGÓRICAS
# ============================================================================

print("\n" + "="*70)
print("  PASO 5: ENCODING DE VARIABLES CATEGÓRICAS")
print("="*70)

# ----------------------------------------------------------------------------
# Analizar cardinalidad de features categóricas
# ----------------------------------------------------------------------------

print("\n Análisis de Cardinalidad:")
cat_cardinality = []
for col in cat_features:
    cardinality = X_train[col].nunique()
    cat_cardinality.append({
        'feature': col,
        'cardinality': cardinality,
        'encoding': 'One-Hot' if cardinality <= 10 else 'Label'
    })

cardinality_df = pd.DataFrame(cat_cardinality).sort_values('cardinality', ascending=False)
print(cardinality_df.to_string(index=False))


# ----------------------------------------------------------------------------
# Estrategia de Encoding
# ----------------------------------------------------------------------------

# Features para One-Hot Encoding (≤10 categorías)
low_card_features = cardinality_df[cardinality_df['cardinality'] <= 10]['feature'].tolist()

# Features para Label Encoding (>10 categorías)
high_card_features = cardinality_df[cardinality_df['cardinality'] > 10]['feature'].tolist()

print(f"\n Estrategia de Encoding:")
print(f"   - One-Hot Encoding: {len(low_card_features)} features")
print(f"   - Label Encoding: {len(high_card_features)} features")


# ----------------------------------------------------------------------------
# Label Encoding para alta cardinalidad
# ----------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

if len(high_card_features) > 0:
    print(f"\n Aplicando Label Encoding...")
    
    label_encoders = {}
    
    for col in high_card_features:
        le = LabelEncoder()
        
        # Fit en train
        X_train[col] = le.fit_transform(X_train[col])
        
        # Transform en test (manejar categorías no vistas)
        X_test[col] = X_test[col].map(lambda x: x if x in le.classes_ else -1)
        X_test.loc[X_test[col] == -1, col] = le.transform([le.classes_[0]])[0]  # Asignar moda
        X_test[col] = le.transform(X_test[col])
        
        label_encoders[col] = le
        
        print(f"    {col}: {len(le.classes_)} categorías → [0, {len(le.classes_)-1}]")



# ----------------------------------------------------------------------------
# One-Hot Encoding para baja cardinalidad
# ----------------------------------------------------------------------------

if len(low_card_features) > 0:
    print(f"\n Aplicando One-Hot Encoding...")
    
    # Crear dummies
    X_train_dummies = pd.get_dummies(
        X_train[low_card_features], 
        columns=low_card_features, 
        drop_first=True,  # Evitar multicolinealidad
        prefix=low_card_features
    )
    
    X_test_dummies = pd.get_dummies(
        X_test[low_card_features], 
        columns=low_card_features, 
        drop_first=True,
        prefix=low_card_features
    )
    
    # Asegurar que train y test tengan las mismas columnas
    # (test puede no tener todas las categorías)
    missing_cols = set(X_train_dummies.columns) - set(X_test_dummies.columns)
    for col in missing_cols:
        X_test_dummies[col] = 0
    
    # Reordenar columnas
    X_test_dummies = X_test_dummies[X_train_dummies.columns]
    
    # Eliminar features originales
    X_train = X_train.drop(columns=low_card_features)
    X_test = X_test.drop(columns=low_card_features)
    
    # Concatenar dummies
    X_train = pd.concat([X_train, X_train_dummies], axis=1)
    X_test = pd.concat([X_test, X_test_dummies], axis=1)
    
    print(f"    {len(low_card_features)} features → {X_train_dummies.shape[1]} columnas dummy")


# ----------------------------------------------------------------------------
# Actualizar lista de features numéricas
# ----------------------------------------------------------------------------

# Todas las features categóricas ahora son numéricas
num_features_final = [col for col in X_train.columns if col not in bin_features]

print(f"\n Actualización de features:")
print(f"   - Features binarias: {len(bin_features)} (sin cambios)")
print(f"   - Features numéricas: {len(num_features_final)}")
print(f"   - Total features: {X_train.shape[1]}")

# ----------------------------------------------------------------------------
# Verificaciones finales
# ----------------------------------------------------------------------------

print(f"\n Verificaciones finales:")

assert X_train.shape[1] == X_test.shape[1], " Train y test tienen diferente número de columnas"
print(f"    Train y test tienen {X_train.shape[1]} columnas (OK)")

assert X_train.shape[0] == 100000, f" X_train tiene {X_train.shape[0]} filas, debería tener 100000"
print(f"    Train tiene 100,000 filas (OK)")

assert X_test.shape[0] == 50000, f" X_test tiene {X_test.shape[0]} filas, debería tener 50000"
print(f"    Test tiene 50,000 filas (OK)")

# Verificar que no hay valores faltantes
assert X_train.isnull().sum().sum() == 0, " Hay valores NaN en train"
assert X_test.isnull().sum().sum() == 0, " Hay valores NaN en test"
print(f"    No hay valores faltantes (OK)")

print("\n" + "="*70)
print(f" ENCODING COMPLETO: {X_train.shape[1]} features finales")
print("="*70)



# ----------------------------------------------------------------------------
# Resumen final
# ----------------------------------------------------------------------------

print(f"\n RESUMEN FINAL DE FEATURE ENGINEERING:")
print(f"   1. Missing Count: +1 feature")
print(f"   2. Eliminación: -2 features problemáticas")
print(f"   3. Imputación: 9 features tratadas")
print(f"   4. Interacciones: +20 features")
print(f"   5. Encoding: {len(cat_features)} categóricas → numéricas")
print(f"\n    Features originales: 57")
print(f"    Features finales: {X_train.shape[1]}")
print(f"    Aumento neto: +{X_train.shape[1] - 57} features")


# LIBRERÍAS NECESARIAS (sin instalación adicional)
# ----------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer

# Para SMOTE, usar imbalanced-learn (ya viene instalado en Kaggle)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print(" imblearn disponible - SMOTE habilitado")
except ImportError:
    print(" imblearn no disponible - solo usaremos class_weight='balanced'")
    SMOTE_AVAILABLE = False

print(" Todas las librerías cargadas correctamente")


# SEPARAR FEATURES POR TIPO PARA ESCALADO SELECTIVO
# ----------------------------------------------------------------------------
print("\n Identificando features para escalado...")

# Features que NO necesitan escalado (ya están en 0-1)
no_scale_features = bin_features + [col for col in X_train.columns if '_x_' in col or '_div_' in col or col == 'missing_count']

# Features que SÍ necesitan escalado (numéricas originales y algunas derivadas)
scale_features = [col for col in X_train.columns if col not in no_scale_features]

print(f"    Features a escalar: {len(scale_features)}")
print(f"    Features sin escalar: {len(no_scale_features)} (binarias + interacciones)")


# FUNCIÓN DE MÉTRICA GINI
# ----------------------------------------------------------------------------
def normalized_gini(y_true, y_pred_proba):
    """
    Calcula el coeficiente de Gini normalizado.
    Gini = 2 * AUC - 1
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1

# Crear scorer para usar en cross_val_score
gini_scorer = make_scorer(normalized_gini, needs_proba=True)

print("\n Métrica Gini configurada")


# 3. ESTRATEGIA DE ESCALADO
# ----------------------------------------------------------------------------
print("\n Preparando transformaciones...")

# Opción 1: Escalar TODAS las features (más simple)
# StandardScaler es sensible a outliers, RobustScaler es más robusto
scaler = StandardScaler()  # o RobustScaler()

# Aplicar escalado
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Escalar solo las features que lo necesitan
X_train_scaled[scale_features] = scaler.fit_transform(X_train[scale_features])
X_test_scaled[scale_features] = scaler.transform(X_test[scale_features])

print(f"    {len(scale_features)} features escaladas con StandardScaler")


#  SPLIT PARA VALIDACIÓN
# ----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

print("\n Creando datos de entrenamiento y datos de validacion manteniendo la proporcion")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, 
    y_train, 
    test_size=0.2, 
    stratify=y_train, 
    random_state=42
)

print(f"    Train: {X_train_split.shape}")
print(f"    Validation: {X_val_split.shape}")
print(f"    Proporción target train: {y_train_split.mean():.2%}")
print(f"    Proporción target val: {y_val_split.mean():.2%}")

print("\n" + "="*70)
print(" PREPARACIÓN PARA MODELADO COMPLETA")
print("="*70)


# Pipeline 1: Logistic Regression
# ----------------------------------------------------------------------------
pipeline_lr = Pipeline([
    ('classifier', LogisticRegression(
        class_weight='balanced',  # Compensa el desbalance
        max_iter=1000,
        random_state=42,
        solver='liblinear'  # Mejor para datasets pequeños
    ))
])
pipeline_lr


# Pipeline 2: Random Forest 
# ----------------------------------------------------------------------------
pipeline_rf = Pipeline([
    ('classifier', RandomForestClassifier(
        n_estimators=200,        # Aumentado de 100
        max_depth=20,            # Aumentado de 10
        min_samples_split=5,     # Mejor generalización
        class_weight='balanced',  # Compensa el desbalance
        random_state=42,
        n_jobs=-1
    ))
])
pipeline_rf


pipelines = {
    'Logistic Regression': pipeline_lr,
    'Random Forest': pipeline_rf
}

print(f"\n Total pipelines: {len(pipelines)}")
print("\n  Nota: Usando class_weight='balanced' como alternativa a SMOTE")


# Configurar Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



results = []

for name, pipeline in pipelines.items():
    print(f"\n{'='*70}")
    print(f" Evaluando: {name}")
    print(f"{'='*70}")
    
    # Cross-validation con Gini
    cv_scores_gini = cross_val_score(
        pipeline, 
        X_train_scaled,
        y_train, 
        cv=skf, 
        scoring=gini_scorer,
        n_jobs=-1,
        verbose=0
    )
    
    # Cross-validation con ROC-AUC
    cv_scores_auc = cross_val_score(
        pipeline, 
        X_train_scaled, 
        y_train, 
        cv=skf, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Guardar resultados
    results.append({
        'Model': name,
        'Gini_Mean': cv_scores_gini.mean(),
        'Gini_Std': cv_scores_gini.std(),
        'AUC_Mean': cv_scores_auc.mean(),
        'AUC_Std': cv_scores_auc.std()
    })
    
    print(f"\n Resultados:")
    print(f"   Gini: {cv_scores_gini.mean():.4f} (±{cv_scores_gini.std():.4f})")
    print(f"   ROC-AUC: {cv_scores_auc.mean():.4f} (±{cv_scores_auc.std():.4f})")
    print(f"   Gini por fold: {cv_scores_gini.round(4)}")



results_df = pd.DataFrame(results).sort_values('Gini_Mean', ascending=False)

print("\n" + "="*70)
print(" RANKING DE MODELOS")
print("="*70)
print(results_df.to_string(index=False))

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Gini Score
results_df.plot(x='Model', y='Gini_Mean', kind='barh', ax=ax1, 
                color='steelblue', xerr='Gini_Std', legend=False)
ax1.set_title('Gini Coefficient por Modelo', fontweight='bold', fontsize=12)
ax1.set_xlabel('Gini Score')
ax1.grid(True, alpha=0.3)

# Gráfico 2: ROC-AUC
results_df.plot(x='Model', y='AUC_Mean', kind='barh', ax=ax2, 
                color='coral', xerr='AUC_Std', legend=False)
ax2.set_title('ROC-AUC por Modelo', fontweight='bold', fontsize=12)
ax2.set_xlabel('ROC-AUC Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Identificar mejor modelo
best_model_name = results_df.iloc[0]['Model']
best_gini = results_df.iloc[0]['Gini_Mean']
best_auc = results_df.iloc[0]['AUC_Mean']

print("\n" + "="*70)
print(" MEJOR MODELO IDENTIFICADO")
print("="*70)
print(f"   Modelo: {best_model_name}")
print(f"   Gini: {best_gini:.4f}")
print(f"   ROC-AUC: {best_auc:.4f}")
print("="*70)


# Pipeline RF con datos originales (sin escalar)
# ----------------------------------------------------------------------------
pipeline_rf_unscaled = Pipeline([
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_rf_unscaled


# Evaluar con datos SIN escalar
cv_scores_gini_unscaled = cross_val_score(
    pipeline_rf_unscaled, 
    X_train,  #  SIN ESCALAR
    y_train, 
    cv=skf, 
    scoring=gini_scorer,
    n_jobs=-1,
    verbose=0
)


cv_scores_auc_unscaled = cross_val_score(
    pipeline_rf_unscaled, 
    X_train,  #  SIN ESCALAR
    y_train, 
    cv=skf, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
print(f"\n Resultados RF (SIN escalar):")
print(f"   Gini: {cv_scores_gini_unscaled.mean():.4f} (±{cv_scores_gini_unscaled.std():.4f})")
print(f"   ROC-AUC: {cv_scores_auc_unscaled.mean():.4f} (±{cv_scores_auc_unscaled.std():.4f})")
print(f"   Gini por fold: {cv_scores_gini_unscaled.round(4)}")


# Comparación: Escalado vs Sin Escalar
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print(" COMPARACIÓN: Random Forest")
print("="*70)

comparison = pd.DataFrame({
    'Versión': ['RF con escalado', 'RF sin escalar'],
    'Gini_Mean': [0.1419, cv_scores_gini_unscaled.mean()],
    'Gini_Std': [0.0140, cv_scores_gini_unscaled.std()],
    'AUC_Mean': [0.5710, cv_scores_auc_unscaled.mean()],
    'AUC_Std': [0.0070, cv_scores_auc_unscaled.std()]
})

print(comparison.to_string(index=False))

# Calcular mejora
mejora_gini = ((cv_scores_gini_unscaled.mean() - 0.1419) / 0.1419) * 100
mejora_auc = ((cv_scores_auc_unscaled.mean() - 0.5710) / 0.5710) * 100

print(f"\n Mejora con datos sin escalar:")
print(f"   Gini: {mejora_gini:+.2f}%")
print(f"   ROC-AUC: {mejora_auc:+.2f}%")


# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Comparación Gini
versions = ['RF\ncon escalado', 'RF\nsin escalar']
gini_values = [0.1419, cv_scores_gini_unscaled.mean()]
colors = ['coral' if g < 0.15 else 'steelblue' if g < 0.20 else 'green' for g in gini_values]

ax1.bar(versions, gini_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Gini Score')
ax1.set_title('Random Forest: Comparación Gini', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for i, v in enumerate(gini_values):
    ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# Gráfico 2: Comparación ROC-AUC
auc_values = [0.5710, cv_scores_auc_unscaled.mean()]
colors_auc = ['coral' if a < 0.58 else 'steelblue' if a < 0.60 else 'green' for a in auc_values]

ax2.bar(versions, auc_values, color=colors_auc, alpha=0.7, edgecolor='black')
ax2.set_ylabel('ROC-AUC Score')
ax2.set_title('Random Forest: Comparación ROC-AUC', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(auc_values):
    ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()




# ============================================================================
# RANKING FINAL DE TODOS LOS MODELOS
# ============================================================================

print("\n" + "="*70)
print(" RANKING FINAL DE TODOS LOS MODELOS")
print("="*70)

all_results = pd.DataFrame([
    {
        'Model': 'Logistic Regression',
        'Data': 'Escalado',
        'Gini_Mean': 0.2420,
        'AUC_Mean': 0.6210
    },
    {
        'Model': 'Random Forest',
        'Data': 'Escalado',
        'Gini_Mean': 0.1419,
        'AUC_Mean': 0.5710
    },
    {
        'Model': 'Random Forest',
        'Data': 'Sin Escalar',
        'Gini_Mean': cv_scores_gini_unscaled.mean(),
        'AUC_Mean': cv_scores_auc_unscaled.mean()
    }
]).sort_values('Gini_Mean', ascending=False)

print(all_results.to_string(index=False))

# Identificar el VERDADERO mejor modelo
best_model_final = all_results.iloc[0]

print("\n" + "="*70)
print(" CAMPEÓN DEFINITIVO")
print("="*70)
print(f"   Modelo: {best_model_final['Model']}")
print(f"   Datos: {best_model_final['Data']}")
print(f"   Gini: {best_model_final['Gini_Mean']:.4f}")
print(f"   ROC-AUC: {best_model_final['AUC_Mean']:.4f}")
print("="*70)

# Actualizar el mejor pipeline según resultado
if best_model_final['Model'] == 'Random Forest' and best_model_final['Data'] == 'Sin Escalar':
    print("\n Random Forest SIN ESCALAR es el GANADOR")
    best_pipeline = pipeline_rf_unscaled
    best_model_name = 'Random Forest (sin escalar)'
    X_train_final = X_train  # Usar datos sin escalar
    X_test_final = X_test
else:
    print("\n Logistic Regression sigue siendo el GANADOR")
    best_pipeline = pipelines['Logistic Regression']
    best_model_name = 'Logistic Regression'
    X_train_final = X_train_scaled  # Usar datos escalados
    X_test_final = X_test_scaled


print(" ENTRENAMIENTO FINAL - LOGISTIC REGRESSION")
print("="*70)

# Confirmando el mejor modelo
best_pipeline = pipelines['Logistic Regression']
best_model_name = 'Logistic Regression'
X_train_final = X_train_scaled  # Usar datos ESCALADOS
X_test_final = X_test_scaled

print(f"\n Modelo seleccionado: {best_model_name}")
print(f"   Gini (CV): 0.2420")
print(f"   ROC-AUC: 0.6210")
print(f"   Features: {X_train_final.shape[1]}")


# Entrenar con TODO el conjunto de entrenamiento
# ----------------------------------------------------------------------------
#ESTO PORQUE EB CV YA VALIDO QUE EL MODELO GENERALIZA
print("\n Entrenando con 100,000 muestras...")

best_pipeline.fit(X_train_final, y_train)

print(" Entrenamiento completo")


print(" EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
print("="*70)

y_val_pred_proba = best_pipeline.predict_proba(X_val_split)[:, 1]
y_val_pred = best_pipeline.predict(X_val_split)

# Métricas
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

final_gini = normalized_gini(y_val_split, y_val_pred_proba)
final_auc = roc_auc_score(y_val_split, y_val_pred_proba)

print(f"\n Métricas en Validación:")
print(f"   Gini:    {final_gini:.4f}")
print(f"   ROC-AUC: {final_auc:.4f}")


print("\n Classification Report:")
print(classification_report(y_val_split, y_val_pred, 
                          target_names=['No Claim (0)', 'Claim (1)'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(y_val_split, y_val_pred)
print("\n Confusion Matrix:")
print(f"\n                 Predicted")
print(f"                 No    Claim")
print(f"Actual  No     {cm[0,0]:6,} {cm[0,1]:6,}")
print(f"        Claim  {cm[1,0]:6,} {cm[1,1]:6,}")


# Calcular métricas adicionales
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\n Métricas Detalladas:")
print(f"   Sensitivity (Recall):  {sensitivity:.4f}   % de claims detectados")
print(f"   Specificity:           {specificity:.4f}   % de no-claims correctos")
print(f"   Precision:             {precision:.4f}   % de predicciones claim correctas")
print(f"   F1-Score:              {f1:.4f}")


# Visualizaciones
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'],
            cbar_kws={'label': 'Count'})
axes[0,0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

# 2. Distribución de probabilidades predichas
axes[0,1].hist(y_val_pred_proba[y_val_split == 0], bins=50, alpha=0.7, 
               label='No Claim (0)', color='green', edgecolor='black')
axes[0,1].hist(y_val_pred_proba[y_val_split == 1], bins=50, alpha=0.7, 
               label='Claim (1)', color='red', edgecolor='black')
axes[0,1].set_title('Distribución de Probabilidades Predichas', fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('Probabilidad Predicha')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val_split, y_val_pred_proba)
axes[1,0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {final_auc:.4f})')
axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1,0].set_title('ROC Curve', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_val_split, y_val_pred_proba)
axes[1,1].plot(recall_vals, precision_vals, linewidth=2, label=f'PR Curve')
axes[1,1].axhline(y=y_val_split.mean(), color='r', linestyle='--', 
                  label=f'Baseline ({y_val_split.mean():.3f})')
axes[1,1].set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Recall')
axes[1,1].set_ylabel('Precision')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print(" GENERANDO PREDICCIONES PARA TEST SET")
print("="*70)

test_predictions = best_pipeline.predict_proba(X_test_final)[:, 1]

print(f"\n {len(test_predictions):,} predicciones generadas")
print(f"\n Estadísticas de las predicciones:")
print(f"   Min:     {test_predictions.min():.6f}")
print(f"   Q1:      {np.percentile(test_predictions, 25):.6f}")
print(f"   Median:  {np.median(test_predictions):.6f}")
print(f"   Mean:    {test_predictions.mean():.6f}")
print(f"   Q3:      {np.percentile(test_predictions, 75):.6f}")
print(f"   Max:     {test_predictions.max():.6f}")
print(f"   Std:     {test_predictions.std():.6f}")


# Distribución de predicciones
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
ax1.hist(test_predictions, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(test_predictions.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {test_predictions.mean():.4f}')
ax1.axvline(np.median(test_predictions), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {np.median(test_predictions):.4f}')
ax1.set_title('Distribución de Predicciones - Test Set', fontweight='bold', fontsize=12)
ax1.set_xlabel('Probabilidad Predicha')
ax1.set_ylabel('Frecuencia')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
ax2.boxplot(test_predictions, vert=True)
ax2.set_title('Box Plot - Predicciones Test', fontweight='bold', fontsize=12)
ax2.set_ylabel('Probabilidad Predicha')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Crear archivo de submission
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print(" CREANDO ARCHIVO DE SUBMISSION")
print("="*70)

submission = pd.DataFrame({
    'id': test_ids,
    'target': test_predictions
})

# Guardar
submission.to_csv('submission.csv', index=False)

print(f"\n Archivo 'submission.csv' creado exitosamente")
print(f"   Shape: {submission.shape}")
print(f"   Columnas: {list(submission.columns)}")

print(f"\n Primeras 10 predicciones:")
print(submission.head(10).to_string(index=False))

print(f"\n Últimas 10 predicciones:")
print(submission.tail(10).to_string(index=False))


# Verificaciones finales
# ----------------------------------------------------------------------------
print("\n Verificaciones finales:")

checks = {
    'Número correcto de predicciones': len(submission) == len(test_ids),
    'Sin IDs duplicados': submission['id'].nunique() == len(test_ids),
    'Predicciones en rango [0,1]': submission['target'].between(0, 1).all(),
    'Sin valores faltantes': not submission.isnull().any().any(),
    'IDs coinciden con test': (submission['id'].values == test_ids.values).all()
}

for check, passed in checks.items():
    status = "BIEN" if passed else "MAL"
    print(f"   {status} {check}")

if all(checks.values()):
    print("\n Todas las verificaciones pasaron - Listo para submission")
else:
    print("\n Algunas verificaciones fallaron - Revisar")


# Resumen final
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print(" PROCESO COMPLETO")
print("="*70)

print(f"\n RESUMEN FINAL:")
print(f"   Modelo:              {best_model_name}")
print(f"   Gini (CV):           0.2420")
print(f"   ROC-AUC (CV):        0.6210")
print(f"   Gini (Validación):   {final_gini:.4f}")
print(f"   ROC-AUC (Validación): {final_auc:.4f}")
print(f"   Features usadas:     {X_train_final.shape[1]}")
print(f"   Predicciones:        {len(test_predictions):,}")
print(f"   Archivo:             submission.csv")

print("\n LISTO PARA SUBIR A LA COMPETENCIA KAGGLE")

print("\n" + "="*70)

