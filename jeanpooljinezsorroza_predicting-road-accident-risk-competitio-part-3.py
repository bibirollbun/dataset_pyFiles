"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("ianktoo/simulated-roads-accident-data")

print("Path to dataset files:", path)"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col="id")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col="id")

original = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
original_1 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
original_2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')

orignal_df = pd.concat([original, original_1, original_2])
df_full = pd.concat([df_train, orignal_df]) 


# notebook: House Prices--advance_preprocesor_search_parameters.ipynb
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    # Categóricas (objetos, categorías o booleanas)
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    
    # Numéricas pero categóricas (discretas con pocas categorías)
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    
    # Categóricas cardinales (muchas categorías únicas)
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    
    # Unir categóricas verdaderas + numéricas discretas
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # Numéricas reales (sin incluir num_but_cat)
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    # Excluir target si está en alguna lista
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
                
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    
    # Resumen
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('Cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)
    
    return cat_cols, num_cols, cat_but_car, num_but_cat
#-----------------------------------------------------------------------------------------------
from sklearn.feature_selection import mutual_info_regression

# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()
#-----------------------------------------------------------------------------------------------
# notebook: Feature Selection Explained: RFECV / From-Kaggle: AvanishMeed
def plot_feature_importances(feat, feat_import):
    """Plot feature importance scores vs features"""
    df = pd.DataFrame({"Features": feat, "Importances": feat_import})
    plt.figure(figsize=(15, 6))
    sns.barplot(x="Importances", y="Features", data = df.sort_values("Importances", ascending=False))
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    plt.title(f"Ranking of the {len(feat)} best features", size=15)
    plt.show()
#-----------------------------------------------------------------------------------------------
def rfecv_func(pipeline, X, y, cv):
    # --- Paso 1: separar preprocesador y modelo ---
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]

    # --- Paso 2: aplicar el preprocesador al X ---
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # --- Paso 3: RFECV con el modelo final ---
    rfecv = RFECV(
        estimator=model,
        step=1,
        min_features_to_select=1,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    rfecv.fit(X_processed, y)

    # --- Paso 4: mostrar resultados ---
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Optimal number of features: {rfecv.n_features_}")
    selected_features = [f for f, s in zip(feature_names, rfecv.support_) if s]
    print(f"Optimal features: {selected_features}")

    return rfecv
#-----------------------------------------------------------------------------------------------
def pipe(preprocessor, model, X, y):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe = pipeline.fit(X, y)
    print(f"Pipe del Modelo: {model.__class__.__name__}")
    return pipe
#-----------------------------------------------------------------------------------------------
def get_transformed_columns(preprocessor):
    columns = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                cols_names = transformer.get_feature_names_out(cols)
            except:
                cols_names = cols
        else:
            cols_names = cols
        columns.extend(cols_names)
    return columns
#-----------------------------------------------------------------------------------------------
def Optimal_features_model(pipeline, rfecv, X, y):
    # Obtener nombres de features transformadas
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    # Features seleccionadas por RFECV
    selected_features = [f for f, s in zip(feature_names, rfecv.support_) if s]
    print(f"\nFeatures seleccionadas ({len(selected_features)}):")
    print(selected_features)
    
    # Crear un selector que pase solo las features elegidas
    selector = VarianceThreshold()  # dummy selector
    selector.get_support = lambda: rfecv.support_  # usamos las máscaras del RFECV

    # Clonar el modelo final
    final_estimator = pipeline.steps[-1][1]

    # Crear un nuevo pipeline con preprocesador + selector + modelo
    reduced_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', selector),
        (pipeline.steps[-1][0], final_estimator)
    ])

    # Entrenar todo el pipeline de forma normal
    reduced_pipe.fit(X, y)
    
    print("\nModelo reentrenado con features óptimas.")
    return reduced_pipe
#-----------------------------------------------------------------------------------------------



# Data visualization
print(20*'****')
print('DIMENSION OF THE DATA IMPORTED: \n')
print('df_train = ',df_train.shape)
print('df_test = ',df_test.shape)
print('orignal_df = ',orignal_df.shape)
print('df_full = ',df_full.shape)
print(20*'****')


df_full.isnull().sum()


print('Duplicated values in df_full',df_full.duplicated().sum())
df_full = df_full.drop_duplicates()
print('Duplicated values in df_full',df_full.duplicated().sum())


bool_cols = df_full.select_dtypes(include='bool').columns.tolist()
for col in bool_cols:
    df_full[col] = df_full[col].astype(int)
    
X = df_full.drop(['accident_risk'],axis=1)
y = df_full['accident_risk']

scores = make_mi_scores(X, y)
plot_mi_scores(scores)


"""#Columns Preprocessor
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scaler", StandardScaler(), num_cols),
        ("passthrough", "passthrough", num_but_cat)
    ]
)

#Fitting the models
cat_model_pipe = pipe(preprocessor, cat_model, X, y)
xgb_model_pipe = pipe(preprocessor, xgb_model, X, y)
lgbm_model_pipe = pipe(preprocessor, lgbm_model, X, y)
gbr_model_pipe = pipe(preprocessor, gbr_model, X, y)"""


"""#Plotting features importance by models
models = {
    "CatBoost": cat_model_pipe,
    "XGBoost": xgb_model_pipe,
    "LightGBM": lgbm_model_pipe,
    "GradientBoosting": gbr_model_pipe
}
for name, model in models.items():
    print(f"Feature importances for {name}")
    final_estimator = model.steps[-1][1]
    if name == "CatBoost":
        importances = final_estimator.get_feature_importance()
    else:
        importances = final_estimator.feature_importances_
    transformed_cols = get_transformed_columns(model.named_steps['preprocessor'])
    plot_feature_importances(transformed_cols, importances)"""


cv_split = KFold(5, shuffle=True, random_state=0)


"""catboost_rfecv = rfecv_func(cat_model_pipe, X, y, cv_split)"""


"""xgb_rfecv = rfecv_func(xgb_model_pipe, X, y, cv_split)"""


"""lgbm_rfecv = rfecv_func(lgbm_model_pipe, X, y, cv_split)"""


"""gbr_rfecv = rfecv_func(gbr_model_pipe, X, y, cv_split)"""


"""catboost_opt_model = Optimal_features_model(
    cat_model_pipe,
    catboost_rfecv,
    X,
    y
)"""


"""xgb_opt_model = Optimal_features_model(
    xgb_model_pipe,
    xgb_rfecv,
    X,
    y
)"""


"""lgbm_opt_model = Optimal_features_model(
    lgbm_model_pipe,
    lgbm_rfecv,
    X,
    y
)"""


"""grb_opt_model = Optimal_features_model(
    gbr_model_pipe,
    gbr_rfecv,
    X,
    y
)"""


"""# Dividimos nuevamente el dataset base para evitar overfitting en el meta-modelo
X_train_blend, X_holdout, y_train_blend, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# Predicciones de cada modelo base sobre el set de blending
pred_cat_train = catboost_opt_model.predict(X_train_blend)
pred_xgb_train = xgb_opt_model.predict(X_train_blend)
pred_lgbm_train = lgbm_opt_model.predict(X_train_blend)
pred_gbr_train = grb_opt_model.predict(X_train_blend)

# Crear el dataset de meta-features
blend_train = np.column_stack([
    pred_cat_train,
    pred_xgb_train,
    pred_lgbm_train,
    pred_gbr_train
])

# Entrenar el meta-modelo (puedes usar Ridge, Lasso, CatBoost, etc.)
meta_model = LinearRegression()
meta_model.fit(blend_train, y_train_blend)

# ---- Etapa de predicción ----

# Predicciones de los modelos base sobre el holdout
pred_cat_hold = catboost_opt_model.predict(X_holdout)
pred_xgb_hold = xgb_opt_model.predict(X_holdout)
pred_lgbm_hold = lgbm_opt_model.predict(X_holdout)
pred_gbr_hold = grb_opt_model.predict(X_holdout)

# Dataset de meta-features para test
blend_hold = np.column_stack([
    pred_cat_hold,
    pred_xgb_hold,
    pred_lgbm_hold,
    pred_gbr_hold
])

# Predicción final combinada
final_preds = meta_model.predict(blend_hold)

rmse = np.sqrt(mean_squared_error(y_holdout, final_preds))
print(f"RMSE del modelo blend: {rmse:.4f}")"""


"""# Predicciones de modelos base sobre df_test
pred_cat_test = catboost_opt_model.predict(df_test)
pred_xgb_test = xgb_opt_model.predict(df_test)
pred_lgbm_test = lgbm_opt_model.predict(df_test)
pred_gbr_test = grb_opt_model.predict(df_test)

# Dataset de meta-features para df_test
blend_test = np.column_stack([pred_cat_test, pred_xgb_test, pred_lgbm_test, pred_gbr_test])

# Predicción final con el meta-modelo
final_predictions = meta_model.predict(blend_test)"""


"""# Crear el DataFrame de submission
submission = pd.DataFrame({
    'id': df_test.index,
    'accident_risk': final_predictions
})

# Mostrar las primeras filas
print(submission.head())

# Guardar a CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created!")
"""

