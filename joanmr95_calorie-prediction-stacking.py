import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print("Train:")
train.info()
train.describe()
print("Test:")
test.info()
test.describe()


plt.figure(figsize=(10, 6))
plt.hist(train['Calories'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribución de Calories en el Conjunto de Entrenamiento')
plt.xlabel('Calories Quemadas')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.5)
plt.show()

# 2. Estadísticas Descriptivas de 'Calories'
calories_desc = train['Calories'].describe()
print("\nEstadísticas Descriptivas de Calories:\n", calories_desc)


train['Calories_log'] = np.log1p(train['Calories'])

# 1. Distribución de la variable 'Calories_log' con Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(train['Calories_log'], bins=50, edgecolor='black', alpha=0.7, density=False)
plt.title('Distribución de log(1 + Calories) en el Conjunto de Entrenamiento')
plt.xlabel('log(1 + Calories)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.5)
plt.show()

# 2. Estadísticas Descriptivas de 'Calories_log'
calories_log_desc = train['Calories_log'].describe()
print("\nEstadísticas Descriptivas de log(1 + Calories):\n", calories_log_desc)


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Calories_log'])
plt.title('Boxplot de Calories_log en el Conjunto de Entrenamiento')
plt.xlabel('Calories Quemadas')
plt.show()


from scipy import stats

female_calories_log = train[train['Sex'] == 'female']['Calories_log']
male_calories_log = train[train['Sex'] == 'male']['Calories_log']

t_statistic, p_value = stats.ttest_ind(female_calories_log, male_calories_log)

print(f"Estadístico T: {t_statistic}")
print(f"Valor p: {p_value}")


# Codificación binaria de la variable 'Sex'
train['Sex_encoded'] = train['Sex'].map({'female': 0, 'male': 1})

# Calcular la matriz de correlación incluyendo la variable codificada
correlation_matrix_encoded = train[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories_log', 'Sex_encoded']].corr()

# Visualizar la matriz de correlación con el sexo codificado
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_encoded, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Variables Numéricas (con Calories_log y Sex Codificado)')
plt.show()


!pip install catboost
!pip install lightgbm


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================== FUNCION RMSLE ==============================

def rmsle(y_true, y_pred):
    def rmsle_inner(y_true, y_pred):
        return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))
    return rmsle_inner(np.expm1(y_true), np.expm1(y_pred))

# ============================== FEATURE ENGINEERING ==============================

for df in [train, test]:
    df['Sex_encoded'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Duration_Weight'] = df['Duration'] * df['Weight']
    df['Duration_Heart_Rate'] = df['Duration'] * df['Heart_Rate']
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR_per_Duration'] = df['Heart_Rate'] / (df['Duration'] + 1e-6)
    df['Duration_squared'] = df['Duration'] ** 2
    df['Weight_squared'] = df['Weight'] ** 2

feature_cols_final = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
                      'Sex_encoded', 'Duration_Weight', 'Duration_Heart_Rate',
                      'BMI', 'HR_per_Duration', 'Duration_squared', 'Weight_squared']

X_train = train[feature_cols_final]
y_train = train['Calories_log']
X_test = test[feature_cols_final]

# ============================== MODELOS BASE ==============================

best_hgb = HistGradientBoostingRegressor(
    l2_regularization=0.6832635188254582,
    learning_rate=0.1941992647121766,
    max_bins=255,
    max_depth=12,
    max_iter=505,
    min_samples_leaf=17,
    random_state=42
)

best_xgb = XGBRegressor(
    subsample=1.0,
    random_state=42,
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    colsample_bytree=0.8,
    verbosity=0
)

best_lgb = LGBMRegressor(
    subsample=0.7,
    random_state=42,
    num_leaves=40,
    n_estimators=700,
    max_depth=10,
    learning_rate=0.05,
    colsample_bytree=0.7
)

# ============================== ENTRENAMIENTO Y PREDICCIONES DE MODELOS ==============================

best_hgb.fit(X_train, y_train)
hgb_pred = best_hgb.predict(X_train)
hgb_rmsle = rmsle(y_train, hgb_pred)

best_xgb.fit(X_train, y_train)
xgb_pred = best_xgb.predict(X_train)
xgb_rmsle = rmsle(y_train, xgb_pred)

best_lgb.fit(X_train, y_train)
lgb_pred = best_lgb.predict(X_train)
lgb_rmsle = rmsle(y_train, lgb_pred)

# ============================== STACKING REGRESSOR ==============================

stacking_model = StackingRegressor(
    estimators=[
        ('hgb', best_hgb),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ],
    final_estimator=LinearRegression(),
    n_jobs=-1
)

# ============================== VALIDACION CRUZADA ==============================

cv = KFold(n_splits=10, shuffle=True, random_state=42)
stacking_oof_pred = cross_val_predict(stacking_model, X_train, y_train, cv=cv, method='predict')
stacking_rmsle = rmsle(y_train, stacking_oof_pred)

# ============================== ENTRENAMIENTO FINAL Y PREDICCION TEST ==============================

stacking_model.fit(X_train, y_train)
predictions_log_stacking = stacking_model.predict(X_test)
predictions_calories = np.expm1(predictions_log_stacking)

submission_df = pd.DataFrame({'id': test['id'], 'Calories': predictions_calories})
submission_df.to_csv('submission.csv', index=False)
print("Archivo 'submission.csv' generado correctamente.")

# ============================== RESUMEN FINAL ==============================

print("\n======================= RESUMEN FINAL =======================")
print(f"\U0001F539 RMSLE - HGB:      {hgb_rmsle:.5f}")
print(f"\U0001F539 RMSLE - XGBoost:  {xgb_rmsle:.5f}")
print(f"\U0001F539 RMSLE - LightGBM: {lgb_rmsle:.5f}")
print(f"\n\U0001F539 RMSLE - Stacking (CV): {stacking_rmsle:.5f}")
print("=============================================================")

