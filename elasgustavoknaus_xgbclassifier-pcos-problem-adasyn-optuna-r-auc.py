# Basics
import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and evaluation
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, 
                                      cross_val_score, cross_val_predict, GridSearchCV, 
                                      RandomizedSearchCV, ShuffleSplit)
from sklearn.metrics import (mean_squared_error, mean_squared_log_error, 
                              mean_absolute_error, mean_absolute_percentage_error, r2_score, 
                              make_scorer)
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, 
                                    OneHotEncoder, OrdinalEncoder, FunctionTransformer, QuantileTransformer, 
                                    quantile_transform)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import boxcox1p
from scipy import stats
import optuna
import shap
from catboost import CatBoostRegressor, Pool

# Configure warnings and plotting styles
warnings.filterwarnings("ignore")
plt.style.use("seaborn-darkgrid")
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

# Load dataset paths if needed
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv(dirname + "/train.csv", index_col = "ID")
test = pd.read_csv(dirname + "/test.csv", index_col = "ID")
display(train.head())
display(train.shape)
display(train.describe())
print("missings ? ")
train.isnull().sum()


train.PCOS.value_counts()


# def missing(df):
#     df.isnull()
def m(train):
    print("%")
    print(train.isnull().mean()*100)

m(train)
print("test \n")
m(test)


print(train.duplicated().sum())
print(test.duplicated().sum())


cat_cols = [col for col in train.columns if train[col].dtype == "O"]
num_cols = [col for col in train.columns if train[col].dtype != "O"]
num_cols


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
train["Weight_kg"].hist(bins = 35, color = "orange")


fig = plt.figure(figsize=(10,5))
plt.title("Swarmplot of Weight With PCOS Diagnosis")
sns.swarmplot(x=train["PCOS"], y=train["Weight_kg"], color = "orange", marker = "v")
plt.show()



def target_dist(df = train,target = "Messi papáaaa"):
    """
    IN
    pandas df with "y"
    target: string type
    """
    
    # Analyze the class distribution
    class_distribution = train[target].value_counts()
    print("Class distribution:\n", class_distribution)
    
    # Identify the majority class
    majority_class = class_distribution.idxmax()
    majority_class_count = class_distribution.max()
    
    # Calculate naive baseline accuracy
    naive_baseline_accuracy = majority_class_count / len(train)
    print(f"Majority class: {majority_class}")
    print(f"Naive baseline accuracy: {naive_baseline_accuracy*100:.2f}%")
    
    # Pie Plot
    
    # Define the labels and sizes for the pie chart
    labels = ['NO', 'YES']
    sizes = class_distribution.values
    explode = (0.1, 0)  # Highlight the first slice (class 0: Did not survive)
    
    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    
    # Add a title and show the plot
    ax.set_title('Class Distribution: PCOS')
    plt.show()

print("train")
target_dist(target = "PCOS")



import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(10, 5))
ax = sns.histplot(x="PCOS", hue="PCOS", data=train, multiple="stack")

# Añadir valores en la punta de las barras
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',  # Texto con el valor
        (p.get_x() + p.get_width() / 2, p.get_height()),  # Posición en la punta
        ha='center', va='bottom', fontsize=10, color='black', fontweight='bold'
    )

plt.xlabel("PCOS")
plt.show()

# Para cada columna categórica
for i, col in enumerate(cat_cols):
    fig = plt.figure(figsize=(10, 3))
    ax = sns.histplot(x=col, hue="PCOS", data=train, multiple="stack")

    # Añadir valores en la punta de cada barra
    for p in ax.patches:
        if p.get_height() > 0:  # Evitar etiquetas en barras invisibles
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold'
            )

    plt.xlabel(col)
    plt.gcf().autofmt_xdate()
    plt.show()



target = "PCOS"
features =  list(test.columns)
# X = train[features]
# y = train[target].map({'No': 0, 'Yes': 1})


def num_fillna(df, columns=None):
    """
    Imputa valores faltantes en columnas numéricas usando la mediana.

    Args:
        df: El DataFrame de Pandas.
        columns: Una lista de nombres de columnas numéricas a las que se aplicará la imputación.
            Si es None, se aplicará a todas las columnas numéricas.

    Returns:
        Un nuevo DataFrame con las columnas numéricas especificadas transformadas.
    """

    df_filled = df.copy()  # Crea una copia para no modificar el DataFrame original

    if columns is None:
        columns = df_filled.select_dtypes(include=['number']).columns

    for column in columns:
        median = df_filled[column].median()
        df_filled[column].fillna(median, inplace=True)

    return df_filled

def cat_fillna(df, columns=None):
    """
    Imputa valores faltantes en columnas categóricas usando la moda.

    Args:
        df: El DataFrame de Pandas.
        columns: Una lista de nombres de columnas categóricas a las que se aplicará la imputación.
            Si es None, se aplicará a todas las columnas categóricas.

    Returns:
        Un nuevo DataFrame con las columnas categóricas especificadas transformadas.
    """

    df_filled = df.copy()  # Crea una copia para no modificar el DataFrame original

    if columns is None:
        columns = df_filled.select_dtypes(exclude=['number']).columns

    for column in columns:
        mode = df_filled[column].mode()[0]  # Obtiene la moda (maneja múltiples modas)
        df_filled[column].fillna(mode, inplace=True)

    return df_filled

# Ejemplo de uso:
data = {'col1': [1, 2, None, 4, 5], 
        'col2': ['A', 'B', None, 'A', 'B'], 
        'col3': [10, 20, 30, None, 50]}
df = pd.DataFrame(data)

df_num_filled = num_fillna(df, columns=['col1', 'col3'])  # Imputa col1 y col3 con la mediana
print("DataFrame con valores numéricos imputados:\n", df_num_filled)

df_cat_filled = cat_fillna(df, columns=['col2'])  # Imputa col2 con la moda
print("\nDataFrame con valores categóricos imputados:\n", df_cat_filled)

df_all_filled = num_fillna(cat_fillna(df))  # Imputa todo el DataFrame
print("\nDataFrame con todos los valores imputados:\n", df_all_filled)



def FE(df):
    full = df
    # check categories of Age in train set
    display(full.Age.value_counts())
    
    # simplify age structure - training data
    full['Age_Group'] = 'MISSING'
    # 20-25
    full.loc[full.Age=='20-25', 'Age_Group'] = '20t25'
    # translate all that are < 20 in level "lt20"
    full.loc[full.Age=='15-20', 'Age_Group'] = 'lt20'
    full.loc[full.Age=='Less than 20', 'Age_Group'] = 'lt20'
    full.loc[full.Age=='Less than 20-25', 'Age_Group'] = 'lt20'
    # translate all that are > 25 in level "gt25"
    full.loc[full.Age=='35-44', 'Age_Group'] = 'gt25'
    full.loc[full.Age=='25-30', 'Age_Group'] = 'gt25'
    full.loc[full.Age=='45 and above', 'Age_Group'] = 'gt25'
    full.loc[full.Age=='30-35', 'Age_Group'] = 'gt25'
    full.loc[full.Age=='30-25', 'Age_Group'] = 'gt25'
    full.loc[full.Age=='30-40', 'Age_Group'] = 'gt25'
    # check results
    display(full['Age_Group'].value_counts())
    
    # simplify Exercise_Type structure - test data
    full['Exercise_Type_Clean'] = 'MISSING'
    
    # replace values
    full.loc[full.Exercise_Type=='Cardio (e.g.', 'Exercise_Type_Clean'] = 'Cardio'
    full.loc[full.Exercise_Type=='No Exercise', 'Exercise_Type_Clean'] = 'No Exercise'
    full.loc[full.Exercise_Type=='Flexibility and balance (e.g.', 'Exercise_Type_Clean'] = 'Flexibility'
    full.loc[full.Exercise_Type=='Strength training (e.g.', 'Exercise_Type_Clean'] = 'Strength'
    full.loc[full.Exercise_Type=='Strength training', 'Exercise_Type_Clean'] = 'Strength'
    full.loc[full.Exercise_Type=='Yes Significantly', 'Exercise_Type_Clean'] = 'Other'
    full.loc[full.Exercise_Type=='No', 'Exercise_Type_Clean'] = 'No Exercise'
    full.loc[full.Exercise_Type=='Sleep_Benefit', 'Exercise_Type_Clean'] = 'MISSING'
    full.loc[full.Exercise_Type=='Not Applicable', 'Exercise_Type_Clean'] = 'MISSING'
    full.loc[full.Exercise_Type=='Somewhat', 'Exercise_Type_Clean'] = 'Somewhat'
    full.loc[full.Exercise_Type=='Strength (e.g.', 'Exercise_Type_Clean'] = 'Strength'

    cat_cols = [col for col in full.columns if full[col].dtype == "O"]
    num_cols = [col for col in full.columns if full[col].dtype != "O"]

    full = num_fillna(full, num_cols)
    full = cat_fillna(full, cat_cols)
    
    full = full.drop(['Age'], axis=1)
    full = full.drop(['Exercise_Type'], axis=1)
    
    full["Weight_kg"] = full["Weight_kg"].astype("int32") 
    
    return full

train = FE(train) 
test = FE(test)


from sklearn.preprocessing import LabelEncoder

def label_encoder(df, columns):
    """
    Aplica Label Encoding a múltiples columnas de un DataFrame.

    Args:
        df: El DataFrame de Pandas.
        columns: Una lista de nombres de columnas a las que se aplicará Label Encoding.

    Returns:
        Un nuevo DataFrame con las columnas especificadas transformadas.
    """

    df_encoded = df.copy()  # Crea una copia para no modificar el DataFrame original
    encoder = LabelEncoder()

    for column in columns:
        if column in df_encoded.columns:  # Verifica si la columna existe
            if df_encoded[column].dtype == 'object' or df_encoded[column].dtype == 'category' :  # Aplica Label Encoding solo a columnas de tipo object o category
                df_encoded[column] = encoder.fit_transform(df_encoded[column])
            else:
                print(f"La columna '{column}' no es de tipo 'object' o 'category'. No se aplicó Label Encoding.")
        else:
            print(f"La columna '{column}' no existe en el DataFrame.")

    return df_encoded

train=label_encoder(train,train.columns)
test=label_encoder(test,test.columns)



train


from imblearn.over_sampling import ADASYN

X = train.copy()
y = X.pop("PCOS")

adasyn = ADASYN(random_state=42)  # Puedes ajustar parámetros como `sampling_strategy`
X_train_resampled, y_train_resampled = adasyn.fit_resample(X, y)



test


y_train_resampled.value_counts()



# def score_dataset(X, y, model=XGBRegressor()):
#     # Label encoding for categoricals
#     #
#     # Label encoding is good for XGBoost and RandomForest, but one-hot
#     # would be better for models like Lasso or Ridge. The `cat.codes`
#     # attribute holds the category levels.
#     for colname in X.select_dtypes(["category"]):
#         X[colname] = X[colname].cat.codes
#     # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
#     log_y = np.log(y)
#     score = cross_val_score(
#         model, X, log_y, cv=5, scoring="neg_mean_squared_error",
#     )
#     score = -1 * score.mean()
#     score = np.sqrt(score)
#     return score


# import optuna

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
#     )
#     xgb = XGBRegressor(**xgb_params)
#     return score_dataset(X_train, y_train, xgb)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20)
# xgb_params = study.best_params


# X_train = create_features(df_train)
# y_train = df_train.loc[:, "SalePrice"]

# xgb_params = dict(
#     max_depth=6,           # maximum depth of each tree - try 2 to 10
#     learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
#     n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
#     min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
#     colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
#     subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
#     reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
#     reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
#     num_parallel_tree=1,   # set > 1 for boosted random forests
# )

# xgb = XGBRegressor(**xgb_params)
# score_dataset(X_train, y_train, xgb)



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier  # Use XGBClassifier for classification
import optuna

def score_dataset(X, y, model=XGBClassifier()):  # Updated for classification
    # Label encoding for categoricals (if needed)
    # for colname in X.select_dtypes(["category"]):
        # X[colname] = X[colname].cat.codes

    # Use StratifiedKFold for classification to maintain class balance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Added random_state

    # Use appropriate scoring metric for classification (e.g., 'roc_auc', 'f1')
    score = cross_val_score(
        model, X, y, cv=cv, scoring="roc_auc"  # Changed scoring metric
    )
    score = score.mean() # No need to multiply by -1 or sqrt for AUC
    return score

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
#         gamma=trial.suggest_float("gamma", 1e-4, 1e2, log=True), # Add gamma for regularization
#         scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.1, 10.0), #for imbalanced classes
#     )

# # Acá entran los datos  # here, load data
#     xgb = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss') # use_label_encoder=False and eval_metric added
#     return score_dataset(X_train_resampled, y_train_resampled, xgb)


# study = optuna.create_study(direction="maximize") # Changed direction to maximize for AUC
# study.optimize(objective, n_trials=133) # You can increase n_trials for more thorough search




# xgb_params = study.best_params

xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.001360613457007491,
    'n_estimators': 5177,
    'min_child_weight': 1,
    'colsample_bytree': 0.6211894236873056,
    'subsample': 0.8608162511883563,
    'reg_alpha': 0.04562800073217687,
    'reg_lambda': 0.1953979908910888,
    'gamma': 0.030300113279955323,
    'scale_pos_weight': 9.3489190030136
}



# # load model
# import joblib

# # . Guarda el modelo en disco
# filename = 'mi_modelo_muchachoszzz.joblib'  # Nombre del archivo
# joblib.dump(model, filename)  # Guarda el modelo en formato joblib

# # . Carga el modelo desde disco
# loaded_model = joblib.load(filename)  # Carga el modelo desde el archivo




# . Train the final model with the best parameters
xgb = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_resampled, y_train_resampled)


predictions = xgb.predict_proba(test)[:, 1]
# Create submission
submission = pd.DataFrame({
    'ID': test.index,
    'PCOS': predictions
})

submission.to_csv('submission.csv', index=False)

print("Submission Preview:")
print(submission.head())
print("\nPrediction Statistics:")
print(f"Number of predictions: {len(predictions)}")
print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
print(f"Mean prediction: {predictions.mean():.3f}")


plt.hist(predictions)

