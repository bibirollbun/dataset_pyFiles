import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import itertools
import random
import warnings
warnings.simplefilter('ignore')


def feature_engineering(df):
    # Se eliminan columnas no relevantes
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'User_ID' in df.columns:
        df.drop(columns=['User_ID'], inplace=True)
    # Se codifican variables categóricas
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    if 'Gender' in df.columns:
        df['Sex'] = df['Gender'].map({'female': 1, 'male': 2})
        df.drop(columns=['Gender'], inplace=True)
    # Se crea una variable AgeSex que combina Age y Sex
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
    for col in ['Sex', 'Age', 'AgeSex']:
        df['CAT_' + col] = df[col].astype('category')
    # Índice de Masa Corporal
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    # Frecuencia cardiaca relativa a la edad
    df['Heart_Rate_per_Age'] = df['Heart_Rate'] / df['Age']
    # Interacción temperatura y ritmo cardiaco
    df['Temp_Heart_Interaction'] = df['Body_Temp'] * df['Heart_Rate']
    # Media de Heart_Rate por Sex
    group_stats = df.groupby('Sex')['Heart_Rate'].agg(['mean', 'std']).rename(columns={'mean': 'Mean_HR_by_Sex', 'std': 'Std_HR_by_Sex'})
    df = df.merge(group_stats, on='Sex', how='left')
    # Diferencia entre el valor individual y la media grupal
    df['HR_above_group_mean'] = df['Heart_Rate'] - df['Mean_HR_by_Sex']
    # Logaritmos y raíces (útil para distribuciones sesgadas)
    df['Log_Weight'] = np.log1p(df['Weight'])  
    df['Sqrt_Height'] = np.sqrt(df['Height'])
    # Potencias y términos polinómicos
    df['Age_squared'] = df['Age'] ** 2  
    df['Weight_cubed'] = df['Weight'] ** 3
    # Potenciación de las interacciones más importantes
    df['Body_Temp_Duration_Squared'] = df['Body_Temp'] * (df['Duration'] ** 2)
    df['Heart_Rate_Duration_Squared'] = df['Heart_Rate'] * (df['Duration'] ** 2)
    df['Height_Duration_Squared'] = df['Height'] * (df['Duration'] ** 2)
    
    # Características basadas en Duration (ya que aparece en 3 de las top features)
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Sqrt_Duration'] = np.sqrt(df['Duration'])
    df['Duration_per_Weight'] = df['Duration'] / df['Weight']
    df['Duration_per_Age'] = df['Duration'] / df['Age']
    
    # Interacciones de segundo orden con las variables importantes
    df['HR_Duration_Temp'] = df['Heart_Rate'] * df['Duration'] * df['Body_Temp']
    df['HR_Duration_Height'] = df['Heart_Rate'] * df['Duration'] * df['Height']
    df['Temp_Duration_Height'] = df['Body_Temp'] * df['Duration'] * df['Height']
    
    # Características basadas en HR_above_group_mean (segunda en weight)
    df['HR_above_mean_squared'] = df['HR_above_group_mean'] ** 2
    df['HR_above_mean_Duration'] = df['HR_above_group_mean'] * df['Duration']
    df['HR_above_mean_Temp'] = df['HR_above_group_mean'] * df['Body_Temp']
    
    # Interacciones con AgeSex (evitando usar la versión categórica)
    df['AgeSex_Duration'] = df['AgeSex'] * df['Duration']
    df['AgeSex_HR'] = df['AgeSex'] * df['Heart_Rate']
    df['AgeSex_Temp'] = df['AgeSex'] * df['Body_Temp']

    features = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']
    # Interacciones entre variables
    for comb in itertools.combinations(features, 2):
        df[" * ".join(comb)] = df[list(comb)].prod(axis=1)
    
    return df


import kagglehub

# Download latest version
path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")

print("Path to dataset files:", path)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_orginal = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

df_train = feature_engineering(df_train)
df_orginal = feature_engineering(df_orginal)
df_test = feature_engineering(df_test)

print('Train data: ', df_train.shape)
print('Test data: ', df_test.shape)
print('Original data: ', df_orginal.shape)


seed = 11
FOLD = 60
cv = KFold(FOLD, random_state=seed, shuffle=True)
pred_test = np.zeros((250000,))

# Listas para almacenar métricas de todos los folds
all_train_rmse = []
all_valid_rmse = []
all_best_iterations = []
all_best_scores = []
all_fold_numbers = []

# Contador para llevar el seguimiento del fold actual
fold_counter = 0

for idx_train, idx_valid in cv.split(df_train):
    fold_counter += 1
    print(f"\n=== Iniciando entrenamiento del fold {fold_counter}/{FOLD} ===\n")
    
    start_time = pd.Timestamp.now()

    X_train = df_train.iloc[idx_train]
    X_train = pd.concat([X_train, df_orginal], axis=0, ignore_index=True).sample(frac=1, random_state=seed)
    X_valid = df_train.iloc[idx_valid]
    
    print(f"Tamaño conjunto de entrenamiento: {X_train.shape[0]} filas, {X_train.shape[1]} columnas")
    print(f"Tamaño conjunto de validación: {X_valid.shape[0]} filas, {X_valid.shape[1]} columnas")

    # Fix: Ensure categorical columns are properly typed
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_valid[col] = X_valid[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    y_train = np.log1p(X_train.pop('Calories'))
    y_valid = np.log1p(X_valid.pop('Calories'))

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(df_test, enable_categorical=True)

    params = {
        'eval_metric': 'rmse',
        'seed': seed,
        'max_depth': 20,
        'learning_rate': 0.003,
        'reg_alpha': 2,
        'reg_lambda': 1,
        'max_delta_step': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.55,
        'enable_categorical': True,
        'device': "cuda"
    }

    # Diccionario para almacenar las métricas de evaluación
    evals_result = {}
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000000, 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=30,
        verbose_eval=1000,
        evals_result=evals_result
    )
    
    # Guardar las métricas de este fold para el gráfico final
    all_train_rmse.append(evals_result['train']['rmse'])
    all_valid_rmse.append(evals_result['validation']['rmse'])
    all_best_iterations.append(model.best_iteration)
    all_best_scores.append(model.best_score)
    all_fold_numbers.append(fold_counter)

    predictions = model.predict(dval)
    pred_test += model.predict(dtest)
    
    # Calcular tiempo transcurrido y mostrar métricas finales
    end_time = pd.Timestamp.now()
    elapsed_time = (end_time - start_time).total_seconds() / 60.0
    
    print(f"\n=== Fold {fold_counter}/{FOLD} completado ===")
    print(f"Tiempo de entrenamiento: {elapsed_time:.2f} minutos")
    print(f"Mejor RMSE en validación: {model.best_score:.6f}")
    print(f"Mejor iteración: {model.best_iteration}")
    print(f"=== Fin del fold {fold_counter}/{FOLD} ===\n")

pred_test /= FOLD


plt.figure(figsize=(10, 6))
for i, (train_rmse, valid_rmse, fold_num) in enumerate(zip(all_train_rmse, all_valid_rmse, all_fold_numbers)):
    # Tomar solo los primeros puntos para mayor claridad (cada 100 iteraciones)
    x_points = list(range(0, len(train_rmse), 100))
    train_points = [train_rmse[j] for j in x_points if j < len(train_rmse)]
    valid_points = [valid_rmse[j] for j in x_points if j < len(valid_rmse)]
    x_points = [x for x in x_points if x < len(train_rmse)]
    
    plt.plot(x_points, valid_points, label=f'Fold {fold_num} - Validación')

plt.title('Evolución del RMSE de Validación por Fold')
plt.xlabel('Iteraciones (cada 100)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.bar(all_fold_numbers, all_best_scores)
plt.title('Mejor RMSE por Fold')
plt.xlabel('Número de Fold')
plt.ylabel('RMSE')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.bar(all_fold_numbers, all_best_iterations)
plt.title('Mejor Iteración por Fold')
plt.xlabel('Número de Fold')
plt.ylabel('Iteración')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(all_train_rmse[-1], label='Train RMSE')
plt.plot(all_valid_rmse[-1], label='Validation RMSE')
plt.axvline(x=all_best_iterations[-1], color='r', linestyle='--', label=f'Mejor iteración: {all_best_iterations[-1]}')
plt.title(f'Train vs Validation RMSE - Fold {FOLD}')
plt.xlabel('Iteraciones')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


metrics_df = pd.DataFrame({
    'Fold': all_fold_numbers,
    'Best_RMSE': all_best_scores,
    'Best_Iteration': all_best_iterations
})
metrics_df.to_csv('/kaggle/working/fold_metrics_summary.csv', index=False)


df_subm = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
df_subm['Calories'] = np.expm1(pred_test)
df_subm.to_csv('/kaggle/working/submission.csv', index=False)


fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figure size if needed
xgb.plot_importance(
    model,
    ax=ax,
    importance_type="gain",
)
plt.title("XGB Ganancia")
plt.show()


fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figure size if needed
xgb.plot_importance(
    model,
    ax=ax,
    importance_type="weight",
)
plt.title("XGB Frecuencia")
plt.show()

