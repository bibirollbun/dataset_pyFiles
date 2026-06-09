import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from sklearn.model_selection import KFold, train_test_split

from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from ydf import RandomForestLearner, GradientBoostedTreesLearner
import ydf
import optuna

import warnings
warnings.filterwarnings('ignore')


# Lendo os arquivos CSV
df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


def cont(variable):
  plt.figure(figsize=(8, 6))
  ax = sns.countplot(
        data=train.dropna(), x=variable,
        color='#08519c',  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
        )
  for p in ax.patches:
      ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

  plt.xlabel(variable)
  plt.ylabel("Contagem")
  plt.title(f"GrÃ¡fico de barras de {variable} ")


def hist(variable):
  plt.figure(figsize=(8, 6))
  sns.histplot(data=train, x= variable, kde=True, color= '#08519c')
  plt.title(f"DistribuiÃ§Ã£o de {variable}")
  plt.xlabel(variable)
  plt.ylabel("FrequÃªncia")
  plt.show()

def media_preÃ§o(variable):
  # Agrupando os dados
  var_med = train.groupby(variable)['Price'].median()

  #Criando o grÃ¡fico
  plt.figure(figsize=(8, 6))

  ax = sns.barplot(x=var_med.index, y=var_med.values, palette='Blues')
  plt.title(f"Valor mÃ©dio de {variable} ")
  plt.xlabel(variable)
  plt.ylabel("Price")

  # Adicionando legendas
  for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

def reg_plot(variable):
  # Criando o grÃ¡fico
  plt.figure(figsize=(8, 6))
  sns.scatterplot(x=train[variable], y=train['Price'])
  plt.title(f'GrÃ¡fico de dispersÃ£o de {variable}')

  plt.show()


pie_chart_palette =[
    "#f7fbff",  # Azul muito claro (quase branco)
    "#deebf7",  # Azul claro
    "#c6dbef",  # Azul mÃ©dio-claro
    "#9ecae1",  # Azul mÃ©dio
    "#6baed6",  # Azul
    "#4292c6",  # Azul mÃ©dio-escuro
    "#2171b5",  # Azul escuro
    "#08519c",  # Azul mais escuro
    "#08306b"   # Azul muito escuro (quase preto)
]

countplot_color = '#2171b5'

# FunÃ§Ã£o para criar grÃ¡ficos de pizza e contagem
def plot_obj(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # GrÃ¡fico pizza
    plt.subplot(1, 2, 1)
    train[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable}")

    # Grafico de barr
    plt.subplot(1, 2, 2)
    ax = sns.countplot(
        data=train.dropna(), x=variable,
        color=countplot_color,  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
        )
    for p in ax.patches:
      ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} ")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


# Concatenando os arquivos de treinamento 
train = pd.concat([df_train, train_extra], axis=0).reset_index(drop=True)


# Visualizando os dados
train.head()


# Vendo o formato dos dados 
print("Shape of traing data:",train.shape)
print("Shape of testing data:", test.shape)


train.describe().round(2).T


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



hist('Price')


hist('Weight Capacity (kg)')


cont('Compartments')


colunas_objeto = train.select_dtypes(include=['object']).columns

# Perform univariate analysis for each categorical variable
for variable in colunas_objeto:
    plot_obj(variable)


colunas_objeto = train.select_dtypes(include=['object']).columns

# Perform univariate analysis for each categorical variable
for variable in colunas_objeto:
    media_preÃ§o(variable)


var = ['Compartments', 'Weight Capacity (kg)']

for variable in var:
  reg_plot(variable)


#Criando grÃ¡fico de correlaÃ§Ã£o
#Fazendo uma cÃ³pia dos dados
X = train.copy()

#Apagando dados alguns dados
X.drop(["id"], axis=1, inplace=True)

#Transformando todos os dados  para formato numÃ©ricos
for colname in X.select_dtypes(["object"]):
  X[colname], _ = X[colname].factorize()

#Removendo as colunas que nÃ£o serÃ£o usadas
# Calculando a matriz de correlaÃ§Ã£o
corr = X.corr()

# Criando o heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)

# Adicionando tÃ­tulos
plt.title('Heatmap de CorrelaÃ§Ã£o dos Dados')

# Mostrando o grÃ¡fico
plt.show()


# Selecionando os dados tipo objeto
colunas_objeto = train.select_dtypes(include=['object']).columns.tolist()
colunas_objeto.append('Compartments')

# Usando um loop for para subistituir o valores pela moda
for col in colunas_objeto:
    train[col].fillna(train[col].mode()[0], inplace=True)

# Subistituindo os valores nulos pela mdiana
train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())


# Selecionando os dados tipo objeto
colunas_objeto = test.select_dtypes(include=['object']).columns.tolist()
colunas_objeto.append('Compartments')

# Usando um loop for para subistituir o valores pela moda
for col in colunas_objeto:
    test[col].fillna(train[col].mode()[0], inplace=True)

# Subistituindo os valores nulos pela mdiana
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())


# Novas Fetures
train['Brand_Material'] = train['Brand'] + '_' + train['Material']
train['Brand_Size'] = train['Brand'] + '_' + train['Size']
train['Has_Laptop_Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})
train['Is_Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Compartments_Category'] = pd.cut(train['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Poucos', 'Moderado', 'Alguns', 'Muitos']).astype(str)
train['peso_Category'] = pd.cut(train['Weight Capacity (kg)'], bins=[0, 10, 15, 20, 25, np.inf], labels=['Leve', 'Moderado', 'Robusto', 'Bem Robusta', "Forte"]).astype(str)
train['Weight_Capacity_Ratio'] = train['Weight Capacity (kg)'] / train['Weight Capacity (kg)'].max()
train['Weight_to_Compartments'] = train['Weight Capacity (kg)'] / (train['Compartments'] + 1)
train['Style_Size'] = train['Style'] + '_' + train['Size']

train["Style_Peso"] = train['Style'] + '_' + train['peso_Category']
train["Color_Peso"] = train['Color'] + '_' + train['peso_Category']
train["Brand_Material_Peso"] = train['Brand_Material'] + '_' + train['peso_Category']
train["Brand_Size_Peso"] = train['Brand_Size'] + '_' + train['peso_Category']
train["CompCat_Peso"] = train['Compartments_Category'] + '_' + train['peso_Category']


# Novas Fetures
test['Brand_Material'] = test['Brand'] + '_' + test['Material']
test['Brand_Size'] = test['Brand'] + '_' + test['Size']
test['Has_Laptop_Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})
test['Is_Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Compartments_Category'] = pd.cut(test['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many']).astype(str)
test['peso_Category'] = pd.cut(test['Weight Capacity (kg)'], bins=[0, 10, 15, 20, 25, np.inf], labels=['Leve', 'Moderado', 'Robusta', 'Bem Robusta', "Forte"]).astype(str)
test['Weight_Capacity_Ratio'] = test['Weight Capacity (kg)'] / test['Weight Capacity (kg)'].max()
test['Weight_to_Compartments'] = test['Weight Capacity (kg)'] / (test['Compartments'] + 1)
test['Style_Size'] = test['Style'] + '_' + test['Size']
test["Style_Peso"] = test['Style'] + '_' + test['peso_Category']
test["Color_Peso"] = test['Color'] + '_' + test['peso_Category']
test["Brand_Material_Peso"] = test['Brand_Material'] + '_' + test['peso_Category']
test["Brand_Size_Peso"] = test['Brand_Size'] + '_' + test['peso_Category']
test["CompCat_Peso"] = test['Compartments_Category'] + '_' + test['peso_Category']


# Importando as bibliotecas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Fazendo a cÃ³pia dos dados
X = train.copy()

# Apagando od iD
X.drop(["id"], axis=1, inplace=True)

#Transformando todos os dados  para formato numÃ©ricos
for colname in X.select_dtypes(["object"]):
  X[colname], _ = X[colname].factorize()

# Normalizando os dados
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(X))

# Criando o cluster como feature
kmeans = KMeans(n_clusters=5)
train["Cluster"] = kmeans.fit_predict(data_normalized)

# Visualizando os dados
train.head()


# Importando as bibliotecas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Fazendo a cÃ³pia dos dados
X_test = test.copy()

# Apagando od iD
X_test.drop(["id"], axis=1, inplace=True)

#Transformando todos os dados  para formato numÃ©ricos
for colname in X_test.select_dtypes(["object"]):
  X_test[colname], _ = X_test[colname].factorize()

# Normalizando os dados
scaler = MinMaxScaler()
data_normalized_test = pd.DataFrame(scaler.fit_transform(X_test))

# Criando o cluster como feature
kmeans = KMeans(n_clusters=5)
test["Cluster"] = kmeans.fit_predict(data_normalized_test)

# Visualizando os dados
test.head()


# Separando uma amostra
df_amostra = train.sample(frac=0.1, random_state=42)

# Separando os dados em X e y
X = df_amostra.drop(['Price','id'], axis=1)
y = df_amostra['Price']
X_test = test.drop("id", axis = 1)


df_amostra.shape, y.shape


for colname in X.select_dtypes(["object"]):
  X[colname], _ = X[colname].factorize()

for colname in X_test.select_dtypes(["object"]):
  X_test[colname], _ = X_test[colname].factorize()


# Normalizando os dados de treino

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X.columns = scaler.get_feature_names_out()

# Normalizando os dados de teste
X_test = pd.DataFrame(scaler.fit_transform(X_test))
X_test.columns = scaler.get_feature_names_out()


# Selecionando os dados categÃ³ricos
colunas_objeto = X.select_dtypes(include=['object']).columns

# Uando get dummies
X = pd.get_dummies(X, columns= colunas_objeto, drop_first=True)

# Normalizando os dados
scaler = MinMaxScaler()
X_scaler = pd.DataFrame(scaler.fit_transform(X))
X_scaler.columns = scaler.get_feature_names_out()


# Selecionando os dados categÃ³ricos
colunas_objeto = X_test.select_dtypes(include=['object']).columns

# Uando get dummies
X_test = pd.get_dummies(X_test, columns= colunas_objeto, drop_first=True)

# Normalizando os dados
scaler = MinMaxScaler()
X_test_scaler = pd.DataFrame(scaler.fit_transform(X_test))
X_test_scaler.columns = scaler.get_feature_names_out()


from sklearn.decomposition import PCA

pca = PCA(n_components=19)  # Reduzindo para 190 ao usar o get dummies
X_pca = pd.DataFrame(pca.fit_transform(X))

print("VariÃ¢ncia Explicada pelos Componentes:", sum(pca.explained_variance_ratio_))
print("VariÃ¢ncia Acumulada:", sum(np.cumsum(pca.explained_variance_ratio_)))


pca = PCA(n_components=19)  
X_pca_test = pd.DataFrame(pca.fit_transform(X_test))

print("VariÃ¢ncia Explicada pelos Componentes:", sum(pca.explained_variance_ratio_))
print("VariÃ¢ncia Acumulada:", sum(np.cumsum(pca.explained_variance_ratio_)))


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error
import optuna
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
SEED = 1
import catboost as cb


X_t, X_v, y_t, y_v = train_test_split(X_pca, y, test_size=0.2, random_state=42)


%%time
def objective_xg(trial):

    params = {
        "eval_metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 100,1000,step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_loguniform("subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10),
        "gamma": trial.suggest_float("gamma", 0, 10),
        
    }

    # Initialize and fit the model
    model_xgb =  XGBRegressor(**params,
                             enable_categorical = True)
    
    model_xgb.fit(X_t, y_t)

    # Predict
    y_pred = model_xgb.predict(X_v)
    
    return mean_squared_error(y_v, y_pred, squared = False)


%%time
study_xgb = optuna.create_study(direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=5, show_progress_bar=True)


print("Melhores hiperprametros:", study_xgb.best_params)


xgb = XGBRegressor(**study_xgb.best_params,
                   eval_metric = "rmse",
                   enable_categorical = True)

xgb.fit(X_t, y_t)



# Fazendo previsÃµes
y_pred = xgb.predict(X_v)

# Avaliando o modelo
mae = mean_absolute_error(y_v, y_pred)
print(f"MAE: {mae}")


saida = xgb.predict(X_test)
sub1 = pd.DataFrame({"id": test["id"],
                    "Price": saida})
sub1.to_csv("XGB_simples.csv", index = False)


# Resultado de MAE outro modelos 

#LinearRegression() -33.653082140080436 
#DecisionTreeRegressor() -45.18324581498228
#XGBRegressor 33.632214118188614
#XGBRegressor: 33.66241802150213


# Definindo o "scorer" como MAE (quanto menor, melhor)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Avaliando o modelo com 5-fold cross-validation
scores = cross_val_score(xgb, X_pca, y, cv=5, scoring=mae_scorer)
# Exibindo os resultados
print(f"MAE MÃ©dio: {np.mean(scores)}")
print(f"MAE por fold: {scores}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_preds = []


# Loop sobre os folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"ðŸ”¹ Treinando Fold {fold + 1}")    

    # Separar os dados de treino e teste para este fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    ## Criar e treinar o modelo
    model = XGBRegressor(
        **study_xgb.best_params,
        eval_metric = "rmse",
        enable_categorical = True)
    
    model.fit(X_train, y_train)

     # Fazer previsÃµes nos dados de teste
    predictions = model.predict(X_valid)
    
    score = mean_absolute_error(y_valid, predictions)
    scores.append(score)

    test_pred = model.predict(X_test)
    test_preds.append(test_pred)


    
print(f"Mean MAE score: {np.mean(scores):.3f}")


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


sample_submission["Price"] = np.mean(test_preds, axis=0)
sample_submission.to_csv("XGBRegressor_mean.csv", index=False)
sample_submission


# Separando uma amostra
df_amostra = train.sample(frac=0.1, random_state=42)

# Separando os dados em X e y
X = df_amostra.drop(['Price','id'], axis=1)
y = df_amostra['Price']
X_test = test.drop("id", axis = 1)


colunas_float  = X.select_dtypes(include=['float64',"object" ]).columns
X[colunas_float] = X[colunas_float].astype("string")
X_test[colunas_float] = X_test[colunas_float].astype("string")


cat_best_params = {
    "learning_rate": 0.17107745771014451,
    "l2_leaf_reg": 6.823218626969235,
    "depth": 7,
}





kfold_cv = KFold(5, shuffle=True, random_state=SEED)
scores = []
test_preds = []
feature_importances = []
X_test_pool = Pool(X_test, cat_features=X.columns.values)

for fold, (train_idx, val_idx) in enumerate(kfold_cv.split(X, y)):
    print(f"ðŸ”¹ Treinando Fold {fold + 1}")
    model = CatBoostRegressor(
        **cat_best_params,
        iterations=1000,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
    )
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=X.columns.values)
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=X.columns.values)

    model.fit(
        X_train_pool, eval_set=X_valid_pool, early_stopping_rounds=200, verbose=False
    )
    feature_importances.append(model.get_feature_importance())

    val_pred = model.predict(X_valid_pool)
    score = mean_squared_error(y_val_fold, val_pred, squared = False)
    scores.append(score)

    test_pred = model.predict(X_test_pool)
    test_preds.append(test_pred)

print(f"Mean RMSE score: {np.mean(scores):.3f}")


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


sample_submission["Price"] = np.mean(test_preds, axis=0)
sample_submission.to_csv("submission.csv", index=False)
sample_submission

