#basics
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import shap

#preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, OneHotEncoder, QuantileTransformer, quantile_transform
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.compose import TransformedTargetRegressor

#statistics
from scipy import stats
from scipy.stats import skew, randint
from scipy.special import boxcox1p

#feature engineering
from sklearn.feature_selection import mutual_info_regression

#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config

#model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV, ShuffleSplit
from sklearn.metrics import mean_absolute_error, make_scorer
import optuna

%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv(dirname+"/train.csv", index_col = "id")
test = pd.read_csv(dirname+"/test.csv", index_col = "id")


#numerical feature descriptive statistics
train.describe().T


#categorical feature descriptive statistics

train.describe(include='object').T.sort_values(by=['unique'], ascending=False)


train.nunique()


def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(train).T# train


import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest


def norm_test(df):
    
    # Filtrar solo las columnas numéricas
    numerical_features = df.select_dtypes(include=[np.number])
    
    # Realizar tests de normalidad para cada característica numérica
    results = []
    
    for column in numerical_features.columns:
        data = df[column]
        
        # Shapiro-Wilk Test
        stat_shapiro, p_shapiro = shapiro(data)
        
        # D'Agostino's K-squared Test
        stat_dagostino, p_dagostino = normaltest(data)
        
        results.append({
            "Feature": column,
            "Shapiro-Wilk p-value": p_shapiro,
            "D'Agostino p-value": p_dagostino,
            "Shapiro Result": "Normal" if p_shapiro > 0.05 else "Not Normal",
            "D'Agostino Result": "Normal" if p_dagostino > 0.05 else "Not Normal"
        })
    
    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame(results)
    
    # Mostrar los resultados
    display(results_df)

norm_test(train)



display(train["Weight Capacity (kg)"].hist(bins = 40, color = "orange"))
plt.show()
train["Compartments"].hist(bins = 40, color = "red")


def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlib and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart and setting the figsize.
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## Creating a grid with 3 columns and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1)

    ## Customizing the QQ plot.
    ax2 = fig.add_subplot(grid[2, :2])
    ax2.set_title('QQ Plot')
    stats.probplot(df.loc[:, feature], plot=ax2)

    ## Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[1, :2])  # Move this to the last row
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:, feature], ax=ax3)

# Call the function
plotting_3_chart(train, 'Price')


print("log1p(train.Price)): \n\n")
plt.hist(np.log1p(train.Price))


from scipy.stats import spearmanr

def spearman_corr(df: pd.DataFrame, target: str) -> pd.DataFrame:

    test_num_list = []
    test_num_cols = []
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target)
    
  
    valid_data = df[[target] + list(num_cols)].dropna()
    
    for num_col in num_cols:
   
        correlation, p_value = spearmanr(valid_data[num_col], valid_data[target])
        
        test_num_list.append([correlation, np.round(p_value, 5)])
        test_num_cols.append(num_col)
    

    return pd.DataFrame(test_num_list, index=test_num_cols, columns=['correlation', 'p_value']).sort_values(by="correlation", ascending=False)

spearman_corr(train, "Price").T



def customized_scatterplot(y, x):
        ## Sizing the plot. 
    plt.subplots(figsize = (12,8))
    sns.scatterplot(y = y, x = x)

customized_scatterplot(train.Price, train["Weight Capacity (kg)"]) 


sns.violinplot(data = train, y = "Price", x = "Size")


from scipy.stats import f_oneway, kruskal
import pandas as pd

def cat_kruskal(train_df: pd.DataFrame, target: str) -> pd.DataFrame:
    test_cat_list = []
    test_cat_cols = []

    # Identificar columnas categóricas
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        # Agrupar los valores del target por las categorías de la columna
        test_group = train_df.groupby(col)[target].apply(list)
        
        # Pruebas estadísticas
        f_oneway_result = f_oneway(*test_group)
        kruskal_result = kruskal(*test_group)
    
        # Evaluar conclusiones según los p-valores
        anova_conclusion = (
            "Significant differences (p < 0.05)"
            if f_oneway_result.pvalue < 0.05
            else "No significant differences (p ≥ 0.05)"
        )
        kruskal_conclusion = (
            "Significant differences (p < 0.05)"
            if kruskal_result.pvalue < 0.05
            else "No significant differences (p ≥ 0.05)"
        )
        
        # Agregar resultados y conclusiones
        test_cat_list.append([
            f_oneway_result.statistic, f_oneway_result.pvalue, anova_conclusion,
            kruskal_result.statistic, kruskal_result.pvalue, kruskal_conclusion
        ])
        test_cat_cols.append(col)
    
    # Crear DataFrame con resultados
    return pd.DataFrame(
        test_cat_list,
        index=test_cat_cols,
        columns=[
            'anova_statistic', 'anova_pvalue', 'anova_conclusion',
            'kruskal_statistic', 'kruskal_pvalue', 'kruskal_conclusion'
        ]
    )


# Llamar a la función
results = cat_kruskal(train, "Price")
display(results)



num_cols = [col for col in train.columns if train[col].dtype != "O"]
cat_cols = [col for col in train.columns if train[col].dtype == "O"]

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))

# original code for this plot : # https://www.kaggle.com/code/igorvolianiuk/only-catboost-eda
for ax, col in zip(axes.flatten(), cat_cols):
    train[col].value_counts().plot(
        kind='barh', 
        color='black',
        fill = True,
        title=f'Backpacks {col}',
        ax=ax 
    )

plt.tight_layout() 
plt.show()




train[num_cols].isnull().sum()


X = train.copy()
y = X.pop("Price")

full = pd.concat([X,test], axis = 0)

num_cols = [col for col in full.columns if full[col].dtype != "O"]
cat_cols = [col for col in full.columns if full[col].dtype == "O"]

#---------------------------------------------------------------------------------------------

# fillna categoricals
full[cat_cols] = full[cat_cols].fillna('None').astype('string').astype('category')

#impute numericals
train_median_weight = train['Weight Capacity (kg)'].median()
full['Weight Capacity (kg)'] = full['Weight Capacity (kg)'].fillna(train_median_weight).astype('string')
full["Compartments"] = full['Compartments'].astype('string')
#---------------------------------------------------------------------------------------------



print("# check miss. : \n\n")

full.isnull().sum()


full.info()


X_train = full[:train.shape[0]]
X_test = full[train.shape[0]:]

# catboost_params = {
#         'loss_function': 'RMSE',
#         'eval_metric': 'RMSE',
#         'learning_rate': 0.05550266178302702,
#         'iterations': 2000,
#         'depth': 4,
#         'random_strength': 0,
#         'l2_leaf_reg': 5.189087598805998,
#         'task_type':'GPU',
#         'random_seed': 42,
#         'verbose': False    
#     }

catboost_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.05550266178302702,
    'iterations': 2100,
    'depth': 4,
    'random_strength': 0,
    'l2_leaf_reg': 5.189087598805998,
    'task_type': 'GPU',  # Usa GPU
    'devices': '0',  # Especifica la GPU (0 es la predeterminada)
    'bootstrap_type': 'Poisson',  # Optimiza para GPU
    'max_ctr_complexity': 2,  # Reduce carga de memoria en GPU
    'gpu_ram_part': 0.95,  # Usa el 95% de la memoria de la GPU
    'thread_count': -1,  # Usa todos los núcleos disponibles de la CPU
    'random_seed': 42,
    'verbose': 100  # Muestra progreso en la consola
}


cv = KFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X_train, y)
scores = []
test_preds = []
X_test_pool = Pool(X_test, cat_features=X_train.columns.values)
for train_idx, val_idx in cv_splits:
    model = CatBoostRegressor(**catboost_params)
    X_train_fold, X_val_fold = X_train.loc[train_idx], X_train.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=X_train.columns.values)
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=X_train.columns.values)
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=100, early_stopping_rounds=200)
    val_pred = model.predict(X_valid_pool)
    score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    scores.append(score)
    test_pred = model.predict(X_test_pool)
    test_preds.append(test_pred)
print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')



sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample_submission['Price'] = np.mean(test_preds, axis=0)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(20)


