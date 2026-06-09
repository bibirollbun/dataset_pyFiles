import numpy as np
import pandas as pd; pd.set_option('display.max_columns', 100)
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, plot_importance
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression

from ydf import RandomForestLearner, GradientBoostedTreesLearner
import ydf


# Lendo os arquivos CSV
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Visualizando os dados
train.head()


print("Shape of traing data:",train.shape)
print("Shape of testing data:", test.shape)


# Criando uma tabela de valores nulos, unicos e tipos de dados 

train_nulos = pd.DataFrame({'Feature': train.columns,
                            '[Train] Valores nulos': train.isnull().sum().values,
                            '[Train] % de valores nulos': ((train.isnull().sum().values)/len(train)*100)})

test_nulos = pd.DataFrame({'Feature': test.columns,
                           '[Test] Valore nulos': test.isnull().sum().values,
                           '[Test] % de valores nulos': ((test.isnull().sum().values)/len(train)*100)})



val_unicos = pd.DataFrame({'Feature': train.columns,
                           'Train valores Ãºnicos': train.nunique().values})

val_unicos_1 = pd.DataFrame({'Feature': test.columns,
                          'Test valores Ãºnicos': test.nunique().values})


tipos = pd.DataFrame({'Feature': train.columns,'DataType': train.dtypes})

junto = pd.merge(val_unicos, val_unicos_1, on = 'Feature', how='left' )
merge_df = pd.merge(train_nulos, test_nulos, on = 'Feature', how='left' )
merge_df = pd.merge(merge_df, junto, on='Feature', how='left')
merge_df = pd.merge(merge_df, tipos, on='Feature', how='left')

merge_df


#Contagem das linhas com nulos e porcentagem
linhas_com_nulos_train = train.isnull().any(axis=1).sum()
linhas_com_nulos_test = test.isnull().any(axis=1).sum()

# GrÃ¡fico de pizza
fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# Plot 1
plt.subplot(1, 2, 1)
labels = [' ', 'Valores nulos']
valores = [len(train) - linhas_com_nulos_train, linhas_com_nulos_train]
plt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90,
        colors =["#2171b5",'#c6dbef'] )
plt.title('Valores nulos de Train')

# Plot 2 
plt.subplot(1, 2, 2)
labels = [' ', 'Valores nulos']
valores = [len(train) - linhas_com_nulos_test, linhas_com_nulos_test]
plt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90,
        colors =["#2171b5",'#c6dbef'] )
plt.title('Valores nulos de Test')


plt.show()


%%time
skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
scores, test_preds_gbt = [], []
feature_importances = []

for i, (train_index, test_index) in enumerate(skf.split(train)):
    print(f"ðŸ”¹ Treinando Fold {i + 1}")
    
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    ydf_md = GradientBoostedTreesLearner(label='Price', 
                                         task=ydf.Task.REGRESSION,
                                         num_threads=10, 
                                         num_trees=1000).train(X_train)
    ydf_pred = ydf_md.predict(X_test)

    score = mean_squared_error(X_test['Price'], ydf_pred, squared=False)
    print('Fold:', i+1, 'Score:', score)
    scores.append(score)

    test_preds_gbt.append(ydf_md.predict(test))

ydf_gb_oof_score = np.mean(scores)  
ydf_gb_std = np.std(scores)
print(f"A mÃ©dia do GradientBoostedTreesLearner Ã© de {ydf_gb_oof_score}")



# Criando uma saida com a mÃ©dia dos valores previsto 
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv').drop("Price", axis = 1)
sample_submission["Price_gbt"] = np.mean(test_preds_gbt, axis=0)
sample_submission.head()


%%time
skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

ydf.verbose(-1)
scores, test_preds_rfl = [], []
for i, (train_index, test_index) in enumerate(skf.split(train)):

    print(f"ðŸ”¹ Treinando Fold {i + 1}")
            
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    
    ydf_md = RandomForestLearner(label='Price', 
                                 task=ydf.Task.REGRESSION, 
                                 num_threads=10, 
                                 num_trees=1000).train(X_train)
    ydf_pred = ydf_md.predict(X_test)

    score = mean_squared_error(X_test['Price'], ydf_pred, squared=False)
    print('Fold:', i, 'RMSE:', score)
    scores.append(score)

    test_preds_rfl.append(ydf_md.predict(test))

    print('-'*50)

ydf_gb_oof_score = np.mean(scores)  
ydf_gb_std = np.std(scores)
print(f"The 5-fold average oof RMSE score of the RandomForestLearner model is {ydf_gb_oof_score}")
print(f"The 5-fold std oof RMSE score of the RandomForestLearner model is {ydf_gb_std}")


sample_submission["Price_rfl"] = np.mean(test_preds_rfl, axis=0)
sample_submission.head()


#Fazendo uma cÃ³pia do conjunto de dados

df = train.copy().drop("id", axis = 1)
df_test = test.copy().drop("id", axis = 1)


# Selecionando os dados tipo objeto de Train
colunas_objeto = train.select_dtypes(include=['object']).columns.tolist()
colunas_objeto.append('Compartments')

# Usando um loop for para subistituir o valores pela moda
for col in colunas_objeto:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Subistituindo os valores nulos pela mdiana
df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].median())


# Selecionando os dados tipo objeto de Test
colunas_objeto = test.select_dtypes(include=['object']).columns.tolist()
colunas_objeto.append('Compartments')

# Usando um loop for para subistituir o valores pela moda
for col in colunas_objeto:
    df_test[col].fillna(df_test[col].mode()[0], inplace=True)

# Subistituindo os valores nulos pela mdiana
df_test["Weight Capacity (kg)"] = df_test["Weight Capacity (kg)"].fillna(df_test["Weight Capacity (kg)"].median())


# Separando dados em X e y
X = df.drop(columns=["Price"], axis=1)
y = df["Price"]
X_test = df_test.copy()


X['Compartments'] = X["Compartments"].astype("int32")
X['Weight Capacity (kg)'] = X["Weight Capacity (kg)"].astype("int32")

X_test['Compartments'] = X_test["Compartments"].astype("int32")
X_test['Weight Capacity (kg)'] = X_test["Weight Capacity (kg)"].astype("int32")


colunas_float  = X.select_dtypes(include=['float64',"object" ]).columns
X[colunas_float] = X[colunas_float].astype("string")

X_test[colunas_float] = X_test[colunas_float].astype("string")


categorical_features = X.select_dtypes(include=['string']).columns.tolist()


cat_best_params = {
    "learning_rate": 0.17107745771014451,
    "l2_leaf_reg": 6.823218626969235,
    "depth": 7,
}


kfold_cv = KFold(5, shuffle=True, random_state=42)
scores = []
test_preds_cat = []
test_pool = Pool(X_test, cat_features=categorical_features)
feature_importances = []

for fold, (train_idx, val_idx) in enumerate(kfold_cv.split(X, y)):
    print("-"*40)
    print(f"ðŸ”¹ Treinando Fold {fold + 1}")
    
    # Separar os dados de treino e teste para este fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    # Criar Pools (CatBoost reconhece categorias automaticamente)
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

    # Criar e treinar o modelo
    model = CatBoostRegressor(
        **cat_best_params,
        iterations=1000,
        loss_function="RMSE",
        eval_metric="RMSE", 
        verbose=100)
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    feature_importances.append(model.get_feature_importance())

    val_pred = model.predict(val_pool)
    score = mean_squared_error(y_valid, val_pred, squared = False)
    scores.append(score)

    test_pred = model.predict(test_pool)
    test_preds_cat.append(test_pred)


print(f"Mean RMSE score: {np.mean(scores):.3f}")


avg_importance = np.mean(feature_importances, axis=0)
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(10)
sns.set_style("whitegrid")
palette = sns.color_palette("Blues_r", len(top_features))
plt.figure(figsize=(9, 7))
ax = sns.barplot(x="Importance", y="Feature", data=top_features, palette=palette)
plt.title("Most Important Features - CatBoost Model", fontsize=15, fontweight="bold")
plt.xlabel("Importance Score", fontsize=10)
plt.ylabel("Features", fontsize=10)
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)


plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()


sample_submission["Price_cat"] = np.mean(test_preds_cat, axis=0)
sample_submission.head()


# Selecionando dados de train categÃ³ricos e substituindo por valores numÃ©ricos 
for colname in df.select_dtypes(["object"]):
  df[colname], _ = df[colname].factorize()

# Selecionando dados de test categÃ³ricos e substituindo por valores numÃ©ricos 
for colname in df_test.select_dtypes(["object"]):
  df_test[colname], _ = df_test[colname].factorize()


# Separando dados em X e y
X = df.drop(columns=["Price"], axis=1)
y = df["Price"]


# Dividindo os dados entre treino e validaÃ§Ã£o 
X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)


%%time
# HiperparÃ¢metros de XGBRegressor
def objective_xg(trial):

    params = {
        "n_estimators": 100,
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 0.01, 1),
        "subsample": trial.suggest_loguniform("subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1)
    }

    # Iniciando e ajustando o modelo
    model_xgb =  XGBRegressor(**params,
                             enable_categorical = True)
    
    model_xgb.fit(X_t, y_t)

    # Predict
    y_pred = model_xgb.predict(X_v)
    
    return mean_squared_error(y_v, y_pred, squared = False)


%%time
# Criar estudo e executar a otimizaÃ§Ã£o
study_xgb = optuna.create_study(direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=5, show_progress_bar=True)


#Criando o numero de folds e listas vazias
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = []
test_preds_xgbr = []


# Loop sobre os folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"ðŸ”¹ Treinando Fold {fold + 1}")    

    # Separar os dados de treino e teste para este fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    ## Criar e treinar o modelo
    model = XGBRegressor(
        **study_xgb.best_params,
        n_estimators = 100,
        eval_metric = "rmse",
        enable_categorical = True)
    
    model.fit(X_train, y_train)

     # Fazer previsÃµes nos dados de teste
    predictions = model.predict(X_valid)
    
    score = mean_squared_error(y_valid, predictions, squared = False)
    scores.append(score)

    test_pred = model.predict(df_test)
    test_preds_xgbr.append(test_pred)


    
print(f"Mean RMSE score: {np.mean(scores):.3f}")


xgb = XGBRegressor(**study_xgb.best_params,
                   n_estimators = 100,
                   eval_metric = "rmse",
                   enable_categorical = True)

xgb.fit(X_t, y_t)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax = plot_importance(
    xgb,
    show_values=False,
    title= "Feature importance | XGBoost Model",
    ax=ax,
    xlabel="",
    height=0.7,
    color="#7a1549",
)
ax.bar_label(ax.containers[0], fmt="{:,.01f}", fontsize = 8, )
ax.grid(False)

plt.show()


sample_submission["Price_xgbr"] = np.mean(test_preds_xgbr, axis=0)
sample_submission.drop('id', axis = 1, inplace = True)


sample_submission


#Criando o numero de folds e listas vazias
kf = KFold(n_splits=15, shuffle=True, random_state=42)
scores = []
test_preds_rl = []

# Loop sobre os folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"ðŸ”¹ Treinando Fold {fold + 1}")

    # Separar os dados de treino e teste para este fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    modelo =LinearRegression()

    modelo.fit(X_train,y_train)
    
    # Fazer previsÃµes nos dados de teste
    predictions = model.predict(X_valid)
    
    score = mean_squared_error(y_valid, predictions, squared = False)
    scores.append(score)
    print(f'o MSE Ã© de {score}')

    test_pred = model.predict(df_test)
    test_preds_rl.append(test_pred)


    
print(f"Mean RMSE score: {np.mean(scores):.3f}")


sample_submission["Price_rl"] = np.mean(test_preds_rl, axis=0)
sample_submission


pesos = [0.18, 0.12, 0.2, 0.2,0.3]  # Exemplo: o primeiro modelo tem mais influÃªncia


sample_submission["Previsao_Ponderada"] = np.average(sample_submission, axis=1, weights=pesos)


sample_submission


submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


submission["Price"] = sample_submission['Previsao_Ponderada']
submission.to_csv("submission.csv", index=False)
submission

