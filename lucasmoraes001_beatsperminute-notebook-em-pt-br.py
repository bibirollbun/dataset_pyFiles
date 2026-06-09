import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Machine Learning libs

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedKFold
from scipy.stats import shapiro, ttest_ind, levene, bartlett, f_oneway, ttest_rel




df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


df.head()


df.shape


df.duplicated().sum()


df1 = df.copy()


df1.drop(['id'], inplace=True, axis=1)


df1.isna().sum()


df1.dtypes


df1.describe()


variaveis = [
    "RhythmScore",
    "AudioLoudness",
    "VocalContent",
    "AcousticQuality",
    "InstrumentalScore",
    "LivePerformanceLikelihood",
    "MoodScore",
    "TrackDurationMs",
    "Energy",
    "BeatsPerMinute"
]

# Criando histogramas para cada variável
for var in variaveis:
    print(f'Estatisticas da variável {var}:')
    print(df[var].describe())
    plt.figure(figsize=(10,8))
    plt.boxplot(df[var])
   # plt.hist(df[var])
    plt.title(f"Distribuição de {var}")
    plt.xlabel(var)
    plt.ylabel("Frequência")
    plt.show()



Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).sum()


outliers


outliers_summary = pd.DataFrame({'Outliers': outliers, 'Percentual': (outliers / len(df1)) * 100})


outliers_summary[outliers_summary['Outliers'] > 0]


Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1

limit_inferior = Q1 - 1.5 * IQR
limit_superior = Q3 + 1.5 * IQR

df2 = df1[~ ((df1 < limit_inferior) | (df1 > limit_superior)).any(axis=1)]


df2 


df2.describe()


df_train = df2.copy()


poly = PolynomialFeatures(interaction_only=True)
new_features = poly.fit_transform(df_train.drop('BeatsPerMinute', axis=1 ))


new_features = pd.DataFrame(new_features, columns=poly.get_feature_names_out())
new_features.head()


# Separando dados treino, teste e validação

from sklearn.model_selection import RepeatedKFold


X = new_features.drop('1', axis=1)
y = df_train['BeatsPerMinute']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12, shuffle=True)

kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=35)

# Separação para os dados SEM novas variáveis

Xw = df_train.drop('BeatsPerMinute', axis=1)
yw = df_train['BeatsPerMinute']

Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.25, random_state=12, shuffle=True)


N_SPLITS = 5
N_REPEATS = 2
TOTAL_SPLITS = N_SPLITS * N_REPEATS

print('Iniciando treinamento com Optuna para otimização de hiperparametros...\n')

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "device": "cuda",
        # Alterado para suggest_float ou suggest_int quando apropriado,
        # mas mantendo suggest_categorical conforme o original para aderência,
        # exceto onde a lista de categorias é grande, o que sugere um float/int.
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.2]),
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800, 1200]),
        "max_depth": trial.suggest_categorical("max_depth", [1, 5, 7, 9]),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.01, 0.1, 1.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0]),
        "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
        "n_jobs": -1,
    }

    rmse_scores = []

    # Loop de Validação Cruzada (usa kf definido globalmente)
    for train_index, val_index in kf.split(X_train, y_train):
        
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
    
        model = XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold)
    
        y_predict = model.predict(X_val_fold)
    
        mse = mean_squared_error(y_val_fold, y_predict)
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)

    # O Optuna tenta MINIMIZAR o valor retornado.
    mean_rmse = sum(rmse_scores) / TOTAL_SPLITS
    return mean_rmse


# ======== Executar a otimização ========
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n--- Otimização Concluída ---")
print("Best CV RMSE:", study.best_value)
print("Best params:", study.best_params)


# Melhores parametros achados pelo hyperparameter search

with_kfold_xgbparams = {'learning_rate': 0.01, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 1.0, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'tree_method': 'hist'}


from sklearn.pipeline import Pipeline

xgb_model2 = Pipeline([('polynomial', PolynomialFeatures(interaction_only=True)), ('xgbmodel', XGBRegressor(**with_kfold_xgbparams))])



xgb_model2_scores = cross_val_score(xgb_model2, Xw, yw, cv=kf, scoring="neg_root_mean_squared_error") # Vamos utilizar os dados originais, já que o pipeline adiciona as novas features

print(np.mean(-xgb_model2_scores))


xgb_model2.fit(Xw_train, yw_train)

rmse = np.sqrt(mean_squared_error(yw_test, xgb_model2.predict(Xw_test)))

print(f'RMSE do modelo: {rmse}')

#RMSE do modelo: 25.9821325086476


importances = xgb_model2.named_steps["xgbmodel"].feature_importances_
feature_names = xgb_model2.named_steps["polynomial"].get_feature_names_out()
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.barh(
    range(len(feature_names)), 
    importances[sorted_idx], 
    align='center'
)
plt.yticks(
    range(len(feature_names)), 
    feature_names[sorted_idx]
)
plt.xlabel("Importância da Feature (Normalizada - Gain)")
plt.title("Feature Importance com Sklearn API")
plt.tight_layout()
plt.show()



N_SPLITS = 5
N_REPEATS = 2
TOTAL_SPLITS = N_SPLITS * N_REPEATS

print('Iniciando treinamento com Optuna para otimização de hiperparametros...\n')

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "device": "cuda",
        # Alterado para suggest_float ou suggest_int quando apropriado,
        # mas mantendo suggest_categorical conforme o original para aderência,
        # exceto onde a lista de categorias é grande, o que sugere um float/int.
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.2]),
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800, 1200]),
        "max_depth": trial.suggest_categorical("max_depth", [1, 5, 7, 9]),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.01, 0.1, 1.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0]),
        "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
        "n_jobs": -1,
    }

    rmse_scores = []

    # Loop de Validação Cruzada (usa kf definido globalmente)
    for train_index, val_index in kf.split(Xw_train, yw_train):
        
        Xw_train_fold = Xw_train.iloc[train_index]
        Xw_val_fold = Xw_train.iloc[val_index]
        yw_train_fold = yw_train.iloc[train_index]
        yw_val_fold = yw_train.iloc[val_index]
    
        model = XGBRegressor(**params)
        model.fit(Xw_train_fold, yw_train_fold)
    
        y_predict = model.predict(Xw_val_fold)
    
        mse = mean_squared_error(yw_val_fold, y_predict)
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)

    # O Optuna tenta MINIMIZAR o valor retornado.
    mean_rmse = sum(rmse_scores) / TOTAL_SPLITS
    return mean_rmse


# ======== Executar a otimização ========
study3 = optuna.create_study(direction="minimize", sampler=TPESampler(seed=423))
study3.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n--- Otimização Concluída ---")
print("Best CV RMSE:", study3.best_value)
print("Best params:", study3.best_params)


kfold_withoutfeature_xgbparams = {'learning_rate': 0.01, 'n_estimators': 1200, 'max_depth': 1, 'min_child_weight': 7, 'subsample': 0.6, 'colsample_bytree': 1.0, 'reg_alpha': 0, 'reg_lambda': 2.0, 'tree_method': 'hist'}


xgb_model3 = XGBRegressor(**kfold_withoutfeature_xgbparams)

xgb_model3_scores = cross_val_score(xgb_model3, Xw, yw, cv=kf, scoring="neg_root_mean_squared_error")

print(np.mean(-xgb_model3_scores))


xgb_model3.fit(Xw_train, yw_train)

rmse = np.sqrt(mean_squared_error(yw_test, xgb_model3.predict(Xw_test)))

print(f'RMSE do modelo: {rmse}')


importances = xgb_model3.feature_importances_
feature_names = Xw.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.barh(
    range(len(feature_names)), 
    importances[sorted_idx], 
    align='center'
)
plt.yticks(
    range(len(feature_names)), 
    feature_names[sorted_idx]
)
plt.xlabel("Importância da Feature (Normalizada - Gain)")
plt.title("Feature Importance com Sklearn API")
plt.tight_layout()
plt.show()



# Teste de Shapiro-Wilk para verificar a normalidade
stat_A, p_A = shapiro(xgb_model2_scores)
stat_B, p_B = shapiro(xgb_model3_scores)

# Interpretando os resultados

nivel_significancia = 0.05

if p_A > nivel_significancia:
    print("Não há evidências suficientes para rejeitar a hipótese de normalidade para o modelo com features polinomiais.")
else:
    print("Há evidências suficientes para rejeitar a hipótese de normalidade para o modelo com features polinomiais")

if p_B > nivel_significancia:
    print("Não há evidências suficientes para rejeitar a hipótese de normalidade para o modelo sem features polinomiais")
else:
    print("Há evidências suficientes para rejeitar a hipótese de normalidade para o modelo sem features polinomiais")


t_stat, p_valor = ttest_rel(xgb_model2_scores, xgb_model3_scores)

# H0 -> Não há diferença significativa entre os modelos
# H1 -> Existe uma diferença significativa entre os modelos

# Interpretando o resultado

nivel_significancia = 0.05

if p_valor <= nivel_significancia:
    print("Há evidências suficientes para rejeitar a hipótese nula. Existe uma diferença significativa entre os modelos.")
else:
    print("Não há evidências suficientes para rejeitar a hipótese nula. Não existe uma diferença significativa entre os modelos")

