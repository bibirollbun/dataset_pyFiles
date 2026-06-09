# LibrerÃ­as bÃ¡sicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# LibrerÃ­as de modelado y mÃ©tricas
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ConfiguraciÃ³n general
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")  # Estilo seaborn para grÃ¡ficos
sns.set_context("notebook")  # TamaÃ±o y contexto de los grÃ¡ficos

# Cargar datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col = "id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv", index_col = "id")



print("Gracias a dios ... no tenemos valores faltantes \n ")
num_cols = df_train.select_dtypes(include=np.number).columns
cat_cols = df_train.select_dtypes(include="object").columns

# Porcentaje de valores faltantes por columna
missing_pct = df_train[num_cols].isnull().mean() * 100
missing_pct = missing_pct.sort_values(ascending=False)

print("Porcentaje de valores faltantes por variable:")
print(missing_pct)




print("Transformamos variables categoricas textuales a numericas ... \n")
from sklearn.preprocessing import LabelEncoder

cat_cols = df_train.select_dtypes(include=['object']).columns

df_train_le = df_train.copy()
df_test_le = df_test.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_train_le[col] = le.fit_transform(df_train_le[col])
    df_test_le[col] = le.transform(df_test_le[col])

df_train_le


%%time

def plot_baseline_roc_statsmodels(df):
    print("--- ðŸ“Š Statsmodels ---")
    
    features = df_train.columns[1:-1]
    X = df[features].fillna(0)
    y = df['diagnosed_diabetes']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Agregar constante para intercept
    X_train_sm = sm.add_constant(X_train)
    X_val_sm = sm.add_constant(X_val)
    
    # Ajustar el modelo
    model = sm.Logit(y_train, X_train_sm)
    result = model.fit()
    
    print(result.summary())
    
    # Predecir probabilidades
    y_pred_proba = result.predict(X_val_sm)
    
    # Calcular ROC AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot con colores radicalmente diferentes
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='mediumvioletred', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de falsos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
    plt.ylabel('Ratio de verdaderos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
    plt.title('Evaluacion de curva ROC', fontsize=16, fontweight='bold', color='purple')
    plt.legend(loc="lower right", facecolor='lavender', edgecolor='purple')
    plt.grid(True, linestyle=':', color='orchid', alpha=0.5)
    plt.show()
    
    print(f"Baseline AUC Score: {roc_auc:.5f}")

plot_baseline_roc_statsmodels(df_train_le)



# %%time

# def plot_baseline_roc_statsmodels(df):
#     print("--- ðŸ“Š Statsmodels ---")
    
#     features = [
#     'age',
#     'physical_activity_minutes_per_week',
#     'diet_score',
#     'screen_time_hours_per_day',
#     'bmi',
#     'systolic_bp',
#     'diastolic_bp',
#     'heart_rate',
#     'cholesterol_total',
#     'hdl_cholesterol',
#     'ldl_cholesterol',
#     'triglycerides',
#     'gender',
#     'education_level',
#     'income_level',
#     'smoking_status',
#     'family_history_diabetes',
#     'cardiovascular_history'
# ]

#     X = df[features].fillna(0)
#     y = df['diagnosed_diabetes']
    
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Agregar constante para intercept
#     X_train_sm = sm.add_constant(X_train)
#     X_val_sm = sm.add_constant(X_val)
    
#     # Ajustar el modelo
#     model = sm.Logit(y_train, X_train_sm)
#     result = model.fit()
    
#     print(result.summary())
    
#     # Predecir probabilidades
#     y_pred_proba = result.predict(X_val_sm)
    
#     # Calcular ROC AUC
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     roc_auc = auc(fpr, tpr)
    
#     # Plot con colores radicalmente diferentes
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='mediumvioletred', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Ratio de falsos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
#     plt.ylabel('Ratio de verdaderos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
#     plt.title('Evaluacion de curva ROC', fontsize=16, fontweight='bold', color='purple')
#     plt.legend(loc="lower right", facecolor='lavender', edgecolor='purple')
#     plt.grid(True, linestyle=':', color='orchid', alpha=0.5)
#     plt.show()
    
#     print(f"Baseline AUC Score: {roc_auc:.5f}")

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix

def plot_baseline_roc_statsmodels(df):
    print("--- ðŸ“Š Statsmodels ---")
    
    features = [
        'age',
        'physical_activity_minutes_per_week',
        'diet_score',
        'screen_time_hours_per_day',
        'bmi',
        'systolic_bp',
        'diastolic_bp',
        'heart_rate',
        'hdl_cholesterol',
        'ldl_cholesterol',
        'triglycerides',
        'gender',
        'education_level',
        'income_level',
        'smoking_status',
        'family_history_diabetes',
        'cardiovascular_history'
    ]

    X = df[features].fillna(0)
    y = df['diagnosed_diabetes']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Agregar constante para intercept
    X_train_sm = sm.add_constant(X_train)
    X_val_sm = sm.add_constant(X_val)
    
    # Ajustar el modelo
    model = sm.Logit(y_train, X_train_sm)
    result = model.fit()
    
    print(result.summary())
    
    # Predecir probabilidades
    y_pred_proba = result.predict(X_val_sm)
    
    # Calcular ROC AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='mediumvioletred', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de falsos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
    plt.ylabel('Ratio de verdaderos positivos', fontsize=14, fontweight='bold', color='darkmagenta')
    plt.title('Evaluacion de curva ROC', fontsize=16, fontweight='bold', color='purple')
    plt.legend(loc="lower right", facecolor='lavender', edgecolor='purple')
    plt.grid(True, linestyle=':', color='orchid', alpha=0.5)
    plt.show()
    
    # Imprimir mÃ©tricas adicionales
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    precision = precision_score(y_val, y_pred_class)
    recall = recall_score(y_val, y_pred_class)
    f1 = f1_score(y_val, y_pred_class)
    cm = confusion_matrix(y_val, y_pred_class)
    
    print(f"Baseline AUC Score: {roc_auc:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print("Confusion Matrix:")
    print(cm)

plot_baseline_roc_statsmodels(df_train_le)



sns.countplot(data=df_train, x="diagnosed_diabetes")
plt.title("Distribucion de variable objetivo")
plt.show()



from scipy.stats import normaltest, ttest_ind, f_oneway, levene

def eda_numeric(df, num_cols, sample_size=1000):
    """
    FunciÃ³n de EDA para variables numÃ©ricas:
    - EstadÃ­sticas descriptivas
    - Medidas de dispersiÃ³n (skew, kurtosis)
    - Correlaciones (Pearson y Spearman)
    - DetecciÃ³n de outliers (Z-score e IQR)
    - Histogramas
    - Test de normalidad (Dâ€™Agostino KÂ²)
    - ComparaciÃ³n de medias y ANOVA
    """
    
    print("--------- EstadÃ­sticas descriptivas ---------")
    display(df[num_cols].describe().T)
    
    print("--------- Medidas de dispersiÃ³n robustas ---------")
    display(df[num_cols].agg(['skew','kurt']))
    
    print("\n--------- CorrelaciÃ³n Pearson ---------")
    corr_matrix = df[num_cols].corr(method="pearson")
    display(corr_matrix)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", mask=mask, cbar_kws={"shrink":0.8}, linewidths=.5)
    plt.title("Matriz de CorrelaciÃ³n Pearson", fontsize=12)
    plt.show()
    
    print("\n--------- CorrelaciÃ³n Spearman ---------")
    corr_matrix = df[num_cols].corr(method="spearman")
    display(corr_matrix)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="crest", mask=mask, cbar_kws={"shrink":0.8}, linewidths=.5)
    plt.title("Matriz de CorrelaciÃ³n Spearman", fontsize=12)
    plt.show()
    
    print("\n--------- Outliers (Z-score) ---------")
    z_score = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    outlier_mask = np.abs(z_score) > 3
    outlier_summary = pd.DataFrame({
        'num_outliers': outlier_mask.sum(),
        '%_outliers': 100 * outlier_mask.mean()
    }).sort_values('%_outliers', ascending=False)
    display(outlier_summary)
    
    print("\n--------- Outliers (IQR) ---------")
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df[num_cols] < (Q1 - 1.5*IQR)) | (df[num_cols] > (Q3 + 1.5*IQR))).sum()
    display(outliers_iqr)
    
    print("\n--------- Histogramas ---------")
    sampled = df[num_cols].sample(sample_size, random_state=42)
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(sampled[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    
    print("\n--------- Test de normalidad (Dâ€™Agostino KÂ²) ---------")
    for col in num_cols:
        stat, p = normaltest(sampled[col])
        print(f"{col}: stat={stat:.3f}, p={p:.3f} {'(Normal)' if p>0.05 else '(No Normal)'}")
    
    print("\n--------- ComparaciÃ³n de medias y ANOVA ---------")
    if 'gender' in df.columns and 'age' in df.columns:
        group1 = df[df['gender']=='Female']['age']
        group2 = df[df['gender']=='Male']['age']
        stat, p = ttest_ind(group1, group2)
        print(f"T-test (edad por gÃ©nero): stat={stat:.3f}, p={p:.3f}")
        stat, p = levene(group1, group2)
        print(f"Levene test (igualdad de varianzas): stat={stat:.3f}, p={p:.3f}")
    
    if 'ethnicity' in df.columns and 'diet_score' in df.columns:
        groups = [df[df['ethnicity']==g]['diet_score'] for g in df['ethnicity'].unique()]
        stat, p = f_oneway(*groups)
        print(f"ANOVA (diet_score por etnicidad): stat={stat:.3f}, p={p:.3f}")


# Definir columnas numÃ©ricas
num_cols = ['age','bmi','systolic_bp','diastolic_bp','physical_activity_minutes_per_week',
            'diet_score','sleep_hours_per_day','screen_time_hours_per_day',
            'waist_to_hip_ratio','heart_rate','cholesterol_total','hdl_cholesterol',
            'ldl_cholesterol','triglycerides']

# Llamar funciÃ³n EDA
eda_numeric(df_train, num_cols)



plt.figure(figsize=(16,12))
for i, col in enumerate(num_cols[:9]):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df_train[col], color = "black" )
    plt.title(col)
plt.tight_layout()
plt.show()



def remove_quantile_outliers_robust(
    df,
    low_q=0.01,
    high_q=0.99,
    min_violations=1
):
    """
    Elimina observaciones que caen fuera del rango [low_q, high_q]
    en al menos `min_violations` variables numÃ©ricas.
    """

    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=["int", "float"]).columns

    # lÃ­mites por columna (calculados UNA sola vez)
    bounds = {
        col: (
            df_clean[col].quantile(low_q),
            df_clean[col].quantile(high_q)
        )
        for col in num_cols
    }

    # conteo de violaciones por fila
    violation_count = np.zeros(len(df_clean))

    for col, (low, high) in bounds.items():
        violation_count += (
            (df_clean[col] < low) | (df_clean[col] > high)
        ).astype(int)

    # mÃ¡scara final
    mask = violation_count < min_violations

    return df_clean.loc[mask].reset_index(drop=True)

df_train_clean = remove_quantile_outliers_robust(df_train_le)


plt.figure(figsize=(16,12))
for i, col in enumerate(num_cols[:9]):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df_train_clean[col], color = "black" )
    plt.title(col)
plt.tight_layout()
plt.show()



def winsorize_quantile_train(
    df,
    low_q=0.01,
    high_q=0.99
):
    df_w = df.copy()
    num_cols = df_w.select_dtypes(include=["int", "float"]).columns

    bounds = {
        col: (
            df_w[col].quantile(low_q),
            df_w[col].quantile(high_q)
        )
        for col in num_cols
    }

    for col, (low, high) in bounds.items():
        df_w[col] = df_w[col].clip(lower=low, upper=high)

    return df_w, bounds

df_train_w, win_bounds = winsorize_quantile_train(df_train_le)


plt.figure(figsize=(16,12))
for i, col in enumerate(num_cols[:9]):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df_train_w[col], color = "black" )
    plt.title(col)
plt.tight_layout()
plt.show()



plot_baseline_roc_statsmodels(df_train_clean)
print("# ------")
plot_baseline_roc_statsmodels(df_train_w)


def apply_winsor_bounds(df, bounds):
    df_w = df.copy()
    for col, (low, high) in bounds.items():
        if col in df_w.columns:
            df_w[col] = df_w[col].clip(low, high)
    return df_w

df_test_w = apply_winsor_bounds(df_test_le, win_bounds)



import statsmodels.api as sm
import pandas as pd

# 1. Variables a usar
features = [
    'age',
    'physical_activity_minutes_per_week',
    'diet_score',
    'screen_time_hours_per_day',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'gender',
    'education_level',
    'income_level',
    'smoking_status',
    'family_history_diabetes',
    'cardiovascular_history'
]

# 2. Preparar datos de entrenamiento
X_train = df_train_w[features]
y_train = df_train_w['diagnosed_diabetes']

# 3. Ajustar modelo
X_train_sm = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train_sm)
result = model.fit()

# 4. Preparar datos de test
X_test = df_test_w[features]
X_test_sm = sm.add_constant(X_test)

# 5. Predecir probabilidades
y_test_pred_proba = result.predict(X_test_sm)  # probabilidades

# 6. Crear submission probabilÃ­stica
submission = pd.DataFrame({
    'id': df_test_w.index,
    'diagnosed_diabetes': y_test_pred_proba  # probabilidades
})

submission.to_csv('submission.csv', index=False)
display(submission.head())



submission["diagnosed_diabetes"].hist(bins = 40)

