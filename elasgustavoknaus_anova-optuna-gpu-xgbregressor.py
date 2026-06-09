
#basics
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


#preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, quantile_transform


#statistics
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import randint

#feature engineering
from sklearn.feature_selection import mutual_info_regression


#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config


#algorithms
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR


#model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import optuna
# from optuna.samplers import TPESampler
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_slice


#stacking
from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingCVRegressor

%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# import warnings

# # Ignorar todas las advertencias
# warnings.filterwarnings("ignore")




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



X = train.copy()
y = X.pop("Price")

full = pd.concat([X,test], axis = 0)

X = full.copy()

num_cols = [col for col in full.columns if full[col].dtype != "O"]
cat_cols = [col for col in full.columns if full[col].dtype == "O"]

display(X.isnull().sum())

def maping(X):
    
    # Mapas de valores
    Size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    Laptop_Compartment_map = {'No': 0, 'Yes': 1}
    Waterproof_map = {'No': 0, 'Yes': 1}
    
    # Crear un diccionario para las columnas
    mapping_dict = {
        "Size": Size_map,
        "Laptop Compartment": Laptop_Compartment_map,
        "Waterproof": Waterproof_map
    }
    
    ordinal_binary_features = ["Size", "Laptop Compartment", "Waterproof"]
    
    # Reemplazar valores en las columnas seleccionadas
    X[ordinal_binary_features] = X[ordinal_binary_features].replace(mapping_dict)
    
    display(X)
    return X






# score_dataset(X,y)

def encode(df, cat_cols, handle_missing='ignore'):
    """
    Encodes categorical columns in a Pandas DataFrame.

    Args:
        df: The Pandas DataFrame.
        cat_cols: A list of categorical column names.
        handle_missing: How to handle missing values. Options are:
            'add': Adds a "None" category for missing values (default).
            'impute': Imputes missing values with the most frequent category.
            'ignore':  Does nothing with missing values. They will be encoded as NaN.

    Returns:
        The DataFrame with encoded categorical columns.  Modifies the DataFrame in place.
    """

    for name in cat_cols:
        df[name] = df[name].astype("category")

        if handle_missing == 'add':
            if "None" not in df[name].cat.categories:
                df[name] = df[name].cat.add_categories("None")
                df[name] = df[name].fillna("None") # Fill NaN values with "None"
        elif handle_missing == 'impute':
            most_frequent = df[name].mode()[0] #Get the most frequent value. Handles multiple modes correctly.
            df[name] = df[name].fillna(most_frequent)
        elif handle_missing == 'ignore':
            pass #Do nothing, NaNs remain.
        else:
            raise ValueError("Invalid value for 'handle_missing'.  Must be 'add', 'impute', or 'ignore'.")
        df[name] = df[name].cat.codes  # Convert categories to numerical codes
    return df

def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df

X = maping(X)
X = encode(X,cat_cols)
X = impute(X)
X["Compartments"] = X["Compartments"].astype("int32")
display(X.isnull().sum())
X.tail(4)


new_train = X[:train.shape[0]]
new_test = X[train.shape[0]:]

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# score_dataset(new_train,y) # r/ 0.5942874207194525


# import optuna
# from xgboost import XGBRegressor

# def objective(trial):
#     xgb_params = dict(
#         max_depth=trial.suggest_int("max_depth", 2, 10),
#         learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
#         min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
#         colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
#         subsample=trial.suggest_float("subsample", 0.2, 1.0),
#         reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
#         reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
#         tree_method='gpu_hist',  # Usar GPU
#         gpu_id=0,  # Cambiar el índice si tienes múltiples GPUs
#         n_jobs=-1  # Usar todos los núcleos de CPU disponibles
#     )
#     xgb = XGBRegressor(**xgb_params)
#     return score_dataset(new_train, y, xgb)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=23)
# xgb_params = study.best_params



xgb_params = dict({'max_depth': 5,
 'learning_rate': 0.0017138152308176283,
 'n_estimators': 3312,
 'min_child_weight': 6,
 'colsample_bytree': 0.9685854078301197,
 'subsample': 0.747071429599332,
 'reg_alpha': 0.41194470339798916,
 'reg_lambda': 1.3749587223051007})


xgb = XGBRegressor(**xgb_params)
score_dataset(new_train, y, xgb)

xgb.fit(new_train, np.log(y))
predictions = np.exp(xgb.predict(new_test))

output = pd.DataFrame({'id': new_test.index, 'Price': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


