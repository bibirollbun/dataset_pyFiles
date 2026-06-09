import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline
#import cupy as cp
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch

import optuna
import logging


# Configurar el logger para guardar en un archivo 'my_logs.log'
logging.basicConfig(filename='my_logs.log',
                    level=logging.DEBUG,  # Definir el nivel de logging
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Formato de los logs

# Crear algunos logs
logging.debug("Este es un mensaje de depuración.")
logging.info("Este es un mensaje de información.")
logging.warning("Este es un mensaje de advertencia.")
logging.error("Este es un mensaje de error.")
logging.critical("Este es un mensaje crítico.")


DATASET_DIR = "/kaggle/input/ml-zoomcamp-2024-competition"
SALES_FILE = f"{DATASET_DIR}/sales.csv"
STORES_FILE = f"{DATASET_DIR}/stores.csv"
CATALOG_FILE = f"{DATASET_DIR}/catalog.csv"
TEST_FILE = f"{DATASET_DIR}/test.csv"


PREPRO_DATA_DIR = "/kaggle/working/outputs"
os.makedirs(PREPRO_DATA_DIR, exist_ok=True)
FULL_RAW_TRAIN_FILE = f"{PREPRO_DATA_DIR}/raw_train.csv"
FULL_CLEAN_TRAIN_FILE = f"{PREPRO_DATA_DIR}/full_clean_train.csv"
TRAIN_FILE = f"{PREPRO_DATA_DIR}/train.csv"
VALID_FILE = f"{PREPRO_DATA_DIR}/valid.csv"

RAW_TEST_FILE = f"{PREPRO_DATA_DIR}/raw_test.csv"
CLEAN_TEST_FILE = f"{PREPRO_DATA_DIR}/clean_test.csv"

MODELS_DIR = "/kaggle/working/models"
os.makedirs(MODELS_DIR, exist_ok=True)
BASELINE_MODEL_SAVE_PATH = f"{MODELS_DIR}/baseline-xbg-" + datetime.now().strftime("%Y-%m-%d") + ".pkl"
OPTIMIZED_MODEL_SAVE_PATH = f"{MODELS_DIR}/opt-xbg-" + datetime.now().strftime("%Y-%m-%d") + ".pkl"


AVAILABLE_GPUS = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
AVAILABLE_GPUS


!ls -lh "{DATASET_DIR}"
!ls -lh "{SALES_FILE}"
!ls -lh "{STORES_FILE}"
!ls -lh "{CATALOG_FILE}"
!ls -lh "{TEST_FILE}"


%%time
# Load the data
sales = pd.read_csv(SALES_FILE, index_col=0)
#sales = pd.read_csv(SALES_FILE)
stores = pd.read_csv(STORES_FILE, index_col=0)
#stores = pd.read_csv(STORES_FILE)
catalog = pd.read_csv(CATALOG_FILE, index_col=0)
#catalog = pd.read_csv(CATALOG_FILE)
test = pd.read_csv(TEST_FILE, index_col=0)
#catalog = pd.read_csv(CATALOG_FILE)


print(f"SALES  : {sales.shape}")
print(f"STORE  : {stores.shape}")
print(f"CATALOG: {catalog.shape}")
print(f"TEST   : {test.shape}")


sales.head(5)


sales.dtypes


stores.head(5)


stores.dtypes


print(f"Store_id : ={stores.store_id.unique()}")
print(f"Divisions: ={stores.division.unique()}")
print(f"Format   : ={stores.format.unique()}")
print(f"Cities   : ={stores.city.unique()}")
print(f"Area     : ={stores.area.unique()}")


catalog.head(5)


catalog.dtypes


print(f"item_id      : {len(catalog.item_id.unique())}")
print(f"Dept_name    : {len(catalog.dept_name.unique())}")
print(f"Class_name   : {len(catalog.class_name.unique())}")
print(f"Subclass_name: {len(catalog.subclass_name.unique())}")


sales.isnull().sum()


stores.isnull().sum()


catalog_missing_values = catalog.isnull().sum()/len(catalog)
catalog_missing_values


catalog_missing_values[catalog_missing_values > 0.5]


def read_data(data_path, stores_path, catalog_path):
    # Load the data
    sales = pd.read_csv(data_path, index_col=0)
    sales['store_id'] = sales['store_id'].astype(str)
    sales['item_id']  = sales['item_id'].astype(str)
    
    stores = pd.read_csv(stores_path, index_col=0)
    stores['store_id'] = stores['store_id'].astype(str)
    
    catalog = pd.read_csv(catalog_path, index_col=0)
    catalog['item_id'] = catalog['item_id'].astype(str)

    # Merge info
    data = pd.merge(sales, stores, on='store_id', how='left')
    data = pd.merge(data, catalog, on='item_id', how='left')
    
    return data


def read_test_data(test_path, stores_path, catalog_path):
    # Load the data
    test = pd.read_csv(test_path, index_col=0, sep=';')
    test['store_id'] = test['store_id'].astype(str)
    test['item_id']  = test['item_id'].astype(str)
    
    stores = pd.read_csv(stores_path, index_col=0)
    stores['store_id'] = stores['store_id'].astype(str)
    
    catalog = pd.read_csv(catalog_path, index_col=0)
    catalog['item_id'] = catalog['item_id'].astype(str)

    # Merge info
    data = pd.merge(test, stores, on='store_id', how='left')
    data = pd.merge(data, catalog, on='item_id', how='left')
    
    return data


def run_data_wrangling(data):
    # Base: "row_id", "item_id", "store_id", "date"
    # [SALES] Date: str to datetime
    data['date'] = pd.to_datetime(data["date"])
    # [SALES] Removing aggregate features: price_base y sum_total
    filter_agg_feats = ~data.columns.isin(["price_base", "sum_total"])
    data = data[data.columns[filter_agg_feats]]

    # [Store] Create location feature
    #stores['location'] = stores[["city", "area"]].apply("-".join, axis=1)
    #data.loc[:, 'location']  = data["city"] + "-" + data["area"].astype(str)
    # [Store] Remove city and area
    filter_store_feats = ~data.columns.isin([
        'divison', 'format',
        #'city', 'area'
    ])
    data = data[data.columns[filter_store_feats]]
    
    # [Catalog] Remove missig values: weight_volume, weight_netto, fatness
    filter_catalog_feats = ~data.columns.isin([
        'item_type',
        'weight_volume',
        'weight_netto',
        'fatness'
    ])
    data = data[data.columns[filter_catalog_feats]]

    # Change int to str
    #data['item_id'] = data['item_id'].astype(str)
    data['store_id'] = data['store_id'].astype(str)
    
    # Creta new features
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["weekday"] = data["date"].dt.dayofweek
    data['week_num'] = data['date'].dt.isocalendar().week

    data['weekday_sin'] = np.sin((data.weekday-1)*(2.*np.pi/7))
    data['weekday_cos'] = np.cos((data.weekday-1)*(2.*np.pi/7))
    data['month_sin'] = np.sin((data.month-1)*(2.*np.pi/12))
    data['month_cos'] = np.cos((data.month-1)*(2.*np.pi/12))
    data['week_sin'] = np.sin(2 * np.pi * (data.week_num-1) / 52)
    data['week_cos'] = np.cos(2 * np.pi * (data.week_num-1) / 52)

    ''' '''
    filter_dates_feats = ~data.columns.isin([
        'date'
        #'year',
        #'month',
        #'day',
        #'weekday',
        #'week_num'
    ])
   
    #data = data.drop("date", axis=1)
    data = data[data.columns[filter_dates_feats]]

    # Get categorical column names (object/string)
    #categoricas = data.select_dtypes(include=['object']).columns.tolist()
    # df[col] = df[col].astype('category')
    #data = data.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)

    return data
    


%%time
if not os.path.exists(FULL_RAW_TRAIN_FILE):
    print("Create raw data")
    raw_data = read_data(
        SALES_FILE,
        STORES_FILE,
        CATALOG_FILE
    )
    # Save raw data
    print(f"Saving raw data: {FULL_RAW_TRAIN_FILE}")
    raw_data.to_csv(FULL_RAW_TRAIN_FILE, index=False)


%%time
if not os.path.exists(FULL_CLEAN_TRAIN_FILE):
    print(f"Read raw train: {FULL_RAW_TRAIN_FILE}")
    raw_data = pd.read_csv(FULL_RAW_TRAIN_FILE)
    # Preprocessing
    data = run_data_wrangling(raw_data)
    # Save data
    print(f'Read clean train: {FULL_CLEAN_TRAIN_FILE}')
    data.to_csv(FULL_CLEAN_TRAIN_FILE, index=False)


#%%time
#data = run_data_wrangling(raw_data)


data.head(5)


data.dtypes


data.isnull().sum()/len(data)


def split_dataset(data, test_size=0.2, random_state=42):
    """Splits a dataset into training and validation sets."""
    #df = pd.read_csv(file_path)
    #train_df, valid_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, valid_df = train_test_split(data, test_size=test_size, random_state=random_state)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    return train_df, valid_df

# Example usage
# train, valid = split_dataset('dataset.csv')


%%time
train, valid = split_dataset(data)


train.head(5)


valid.head(5)


print(f"train: {train.shape}, valid: {valid.shape}")


train.dtypes


train.columns


train.isnull().sum()


%%time
#data.columns[~(data.columns == target_column)]

# Target variable
target_column = 'quantity'

# Get X features 
x_features = train.columns[~(train.columns == target_column)]

# Get numeric column names
numerical_features = train[x_features].select_dtypes(include=['number']).columns.tolist()

# Get categorical column names (object/string)
categorical_features = train[x_features].select_dtypes(include=['object']).columns.tolist()


print("Numerical features  :", numerical_features)
print("Categorical features:", categorical_features)


for col in categorical_features:
    print(f"Unique categories in {col} in train: {len(train[col].unique())}")
    print(f"Unique categories in {col} in valid: {len(train[col].unique())}")


def build_pipeline_feat_imp(model, categorical_features, numerical_features):
    """Builds a preprocessing and modeling pipeline."""
    #categorical_features = ['item_type', 'class_name', 'subclass_name', 'store_id', 'division', 'dept_name', 'location', 'weekday']
    #numerical_features = ['year', 'month', 'day']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            #('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ('cat', TargetEncoder(), categorical_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        #('model', model)
        ('cls', model)
    ])
    
    return pipeline


# Detecta la GPU disponible
def detect_gpu():
    try:
        # Verifica si hay GPU disponible
        #if cp.cuda.runtime.getDeviceCount() > 0:
        if len(AVAILABLE_GPUS) > 0:
            return 0  # Retorna el ID de la primera GPU (usualmente 0)
        else:
            return -1  # No hay GPU disponible
    except Exception as e:
        print(f"Error al verificar la GPU: {e}")
        return -1 


def feature_importance_rf(
    train_df, target_column,
    categorical_features, numerical_features
):
    """Calculates feature importance using Random Forest."""
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    #model.fit(X, y)
    pipeline = build_pipeline_feat_imp(model, categorical_features, numerical_features)
    pipeline.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plot_feature_importance(importance, "Random Forest Feature Importance")
    return importance


def feature_importance_xgb(
    train_df, 
    target_column,
    num_feats, 
    ordinal_cat_feats, 
    target_cat_feats,
    params=None
):
    """Calculates feature importance using XGBoost."""
    #X = train_df.drop(columns=[target_column])
    print(f"[TRAIN] Ordinal_cat_feats: {ordinal_cat_feats}")
    print(f"[TRAIN] Target_cat_feats : {target_cat_feats}")
    print(f"[TRAIN] num_feats        : {num_feats}")
    feats = num_feats + target_cat_feats + ordinal_cat_feats
    X_train = train_df[feats]
    y_train = train_df[target_column]

    if not params:
        params = {
            'objective': 'reg:squarederror',
            #'tree_method': 'hist',
            '_estimators': 100,
            'random_state': 42
        }

    enc = make_column_transformer(
        #(StandardScaler(), numerical_features),
        ('passthrough', num_feats),
        (TargetEncoder(), target_cat_feats),
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
         ordinal_cat_feats),
        remainder="passthrough",
        verbose_feature_names_out=True,
    )
    feature_types = ["c" if fn in ordinal_cat_feats else "q" 
                     for fn in X_train.columns]

    print(f"[TRAIN] Params: {params}")
    print(f"[TRAIN] feature_types: {feature_types}")
    # Detecta la GPU
    gpu_id = detect_gpu()
    if gpu_id >= 0:
        print(f"Detected GPU, using GPU {gpu_id}.")
        params['device'] = 'cuda'
    else:
        print("No GPU detected, using CPU.")
        params['device'] = 'cpu'
        
    reg = xgb.XGBRegressor(
        **params, 
        feature_types=feature_types,
        enable_categorical=True
    )
    #pipeline = make_pipeline(enc, reg)
    pipeline = make_pipeline(enc, reg)    
    pipeline.fit(X_train, y_train)
    #model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    #pipeline = build_pipeline(model, categorical_features, numerical_features)
    #pipeline.fit(X, y)
    #importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    # check XGBoost is using the feature type correctly.
    model_types = reg.get_booster().feature_types
    assert model_types is not None
    for a, b in zip(model_types, feature_types):
        assert a == b
    '''
    xgb_model = pipeline.named_steps['xgbregressor'] 
    preprocessor = pipeline.named_steps['preprocessor']]
    importance = pd.Series(
        xgb_model.feature_importances_, 
        index=preprocessor.get_feature_names_out()
    ).sort_values(ascending=False)
    #plot_feature_importance(importance, "XGBoost Feature Importance")
    return importance
    '''
    return pipeline



def plot_feature_importance(importance, title):
    """Plots feature importance as a bar chart."""
    plt.figure(figsize=(10, 6))
    importance.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(importance):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    plt.show()


def select_features(importance, method='threshold', value=0.01):
    """Selects features based on different criteria."""
    if method == 'threshold':
        selected_features = importance[importance >= value].index.tolist()
    elif method == 'top_k':
        selected_features = importance.nlargest(value).index.tolist()
    elif method == 'cumulative':
        cumulative_importance = importance.cumsum() / importance.sum()
        selected_features = cumulative_importance[cumulative_importance <= value].index.tolist()
    else:
        raise ValueError("Invalid method. Choose 'threshold', 'top_k', or 'cumulative'.")
    
    return selected_features


def compute_correlation(train_df):
    """Computes correlation between numerical and categorical features."""
    categorical_features = ['item_type', 'class_name', 'subclass_name', 'store_id', 'division', 'dept_name', 'location', 'weekday']
    numerical_features = ['year', 'month', 'day', 'quantity']
    
    # Encode categorical features
    encoder = TargetEncoder()
    encoded_cats = encoder.fit_transform(train_df[categorical_features], train_df['quantity'])
    
    # Concatenate numerical and encoded categorical features
    df_encoded = pd.concat([train_df[numerical_features], encoded_cats], axis=1)
    
    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()
    
    return correlation_matrix


%%time
'''
rf_importance = feature_importance_rf(
    train, target,
    categorical_features, numerical_features
)'''


numerical_features, categorical_features


# Target variable
target_column = 'quantity'
num_feats = [
    'weekday_sin', 'weekday_cos',
    'week_sin', 'week_cos',
    'month_sin', 'month_cos',
    #'year'
]
target_cat_feats = [
    'item_id', 
    'dept_name', 
    'class_name', 'subclass_name'
]
#ordinal_cat_feats = ['store_id', 'division', 'format', 'location']
ordinal_cat_feats = [
    'store_id', 
    'city', 'area', #worst
    'month', 'day', 'weekday','week_num'
]
all_feats = num_feats + ordinal_cat_feats + target_cat_feats


%%time

xgb_importance_results = feature_importance_xgb(
    train, target_column,
    num_feats, 
    ordinal_cat_feats, 
    target_cat_feats,
)


#print(xgb_importance.named_steps)


xgb_model = xgb_importance_results.named_steps['xgbregressor'] 
preprocessor = xgb_importance_results.named_steps['columntransformer']
xgb_importance = pd.Series(
    xgb_model.feature_importances_, 
    index=preprocessor.get_feature_names_out()
).sort_values(ascending=False)
#plot_feature_importance(importance, "XGBoost Feature Importance")


xgb_importance


plot_feature_importance(xgb_importance, "XGBoost Feature Importance")


xgb_importance[xgb_importance > 0.015]


# Target variable
target_column = 'quantity'
num_feats = [
    #'weekday_sin', 
    'weekday_cos',
    #'week_sin', 
    'week_cos',
    'month_sin', 
    #'month_cos',
    #'year'
]
target_cat_feats = [
    'item_id', 
    'dept_name', 
    'class_name', 
    'subclass_name'
]
ordinal_cat_feats = [
    'store_id', 
    #'city', 'area', #worst
    #'month', 'day', 'weekday','week_num'
]
all_feats = num_feats + ordinal_cat_feats + target_cat_feats



%%time

xgb_importance_results = feature_importance_xgb(
    train, target_column,
    num_feats, 
    ordinal_cat_feats, 
    target_cat_feats,
)


xgb_model = xgb_importance_results.named_steps['xgbregressor'] 
preprocessor = xgb_importance_results.named_steps['columntransformer']
xgb_importance = pd.Series(
    xgb_model.feature_importances_, 
    index=preprocessor.get_feature_names_out()
).sort_values(ascending=False)
#plot_feature_importance(importance, "XGBoost Feature Importance")


xgb_importance


plot_feature_importance(xgb_importance, "XGBoost Feature Importance")


xgb_importance


# Target variable
target_column = 'quantity'
num_feats = [
    #'weekday_sin', 
    'weekday_cos',
    #'week_sin', 
    'week_cos',
    'month_sin', 
    #'month_cos',
    #'year'
]
target_cat_feats = [
    'item_id', 
    'dept_name', 
    'class_name', 
    'subclass_name'
]
ordinal_cat_feats = [
    'store_id', 
    #'city', 'area', #worst
    #'month', 'day', 'weekday','week_num'
]
all_feats = num_feats + ordinal_cat_feats + target_cat_feats


# Detecta la GPU disponible
def detect_gpu():
    try:
        # Verifica si hay GPU disponible
        #if cp.cuda.runtime.getDeviceCount() > 0:
        if len(AVAILABLE_GPUS) > 0:
            return 0  # Retorna el ID de la primera GPU (usualmente 0)
        else:
            return -1  # No hay GPU disponible
    except Exception as e:
        print(f"Error al verificar la GPU: {e}")
        return -1 


def train_xgb(
    train, 
    valid,
    target_column,
    num_feats, 
    ordinal_cat_feats, 
    target_cat_feats,
    params
):
    """Trains a baseline model using XGBoost with GPU acceleration."""
    #X_train = train_df.drop(columns=[target_column])
    print(f"[TRAIN] Ordinal_cat_feats: {ordinal_cat_feats}")
    print(f"[TRAIN] Target_cat_feats : {target_cat_feats}")
    print(f"[TRAIN] num_feats        : {num_feats}")
    feats = num_feats + target_cat_feats + ordinal_cat_feats
    X_train = train[feats]
    y_train = train[target_column]
    print(f"[TRAIN] Train Columns: {X_train.shape}")

    #X_valid = valid_df.drop(columns=[target_column])
    X_valid = valid[feats]
    y_valid = valid[target_column]
    print(f"[TRAIN] Valid Columns: {X_valid.shape}")
    
    print(f"[TRAIN] Transformaton setting")
    enc = make_column_transformer(
        #(StandardScaler(), numerical_features),
        ('passthrough', num_feats),
        (TargetEncoder(), target_cat_feats),
        #(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        #ordinal_cat_feats),
        (Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ordinal_cat_feats),
        remainder="passthrough",
        verbose_feature_names_out=True,
    )
    feature_types = ["c" if fn in ordinal_cat_feats else "q" 
                     for fn in X_train.columns]
    #for col, ft in zip(X_train.columns, feature_types):
    #    print(f"[TRAIN] Feature Types: {col} {ft}")
   
    print(f"[TRAIN] Params: {params}")
    print(f"[TRAIN] feature_types: {feature_types}")
    # Detecta la GPU
    gpu_id = detect_gpu()
    if gpu_id >= 0:
        print(f"Detected GPU, using GPU {gpu_id}.")
        params['device'] = 'cuda'
        steps = [enc] #[enc, FunctionTransformer(move_to_gpu, validate=False)]
    else:
        print("No GPU detected, using CPU.")
        params['device'] = 'cpu'
        steps = [enc]
        
    reg = xgb.XGBRegressor(
        **params, 
        feature_types=feature_types,
        enable_categorical=True
    )
    #pipeline = make_pipeline(enc, reg)
    steps.append(reg)
    pipeline = make_pipeline(*steps)
    print(f"Train....")
    pipeline.fit(
        X_train, y_train,             
        #cls__eval_set=[(X_valid, y_valid)], 
        #xgbregressor__early_stopping_rounds=50, 
        #xgbregressor__verbose=False
    )
    # check XGBoost is using the feature type correctly.
    model_types = reg.get_booster().feature_types
    assert model_types is not None
    for a, b in zip(model_types, feature_types):
        assert a == b
    
    y_pred = pipeline.predict(X_valid)
    val_rmse = mean_squared_error(y_pred, y_valid, squared=False)
    val_mse  = mean_squared_error(y_pred, y_valid)
    val_mape = mean_absolute_percentage_error(y_pred, y_valid)
    
    return {
        'model': pipeline,
        'metrics': {
            'rmse': val_rmse,
            'mse': val_mse,
            'mape': val_mape
        }
    }


# 
seed=42
base_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_estimators': 100,
    'random_state': seed
}
base_params


%%time
base_xgb = train_xgb(
    train, 
    valid,
    target_column,
    num_feats=num_feats, 
    ordinal_cat_feats=ordinal_cat_feats, 
    target_cat_feats=target_cat_feats,
    params=base_params
)


base_xgb_metrics = base_xgb['metrics']
base_xgb_model = base_xgb['model']
base_xgb_metrics


valid_xgb_preds = base_xgb_model.predict(valid[all_feats])
valid['base_preds'] = valid_xgb_preds


valid[[target_column, 'base_preds']][valid.class_name.isnull()]


valid[[target_column, 'base_preds']][valid.class_name.isnull()]


# Saving
# Guardar el modelo usando pickle
print(f"Saving model with pickle. {BASELINE_MODEL_SAVE_PATH}")
with open(BASELINE_MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(base_xgb_model, f)


def optimize_xgb_hyperparameters(
    train_df, valid_df,
    target_column, 
    num_feats, 
    ordinal_cat_feats, 
    target_cat_feats,
    base_params=None,
    n_trials=50
):
    """Optimizes XGBoost hyperparameters using Optuna."""
    #X_train = train_df.drop(columns=[target_column])
    print(f"[HPO] Trails           : {n_trials}")
    print(f"[HPO] Ordinal_cat_feats: {ordinal_cat_feats}")
    print(f"[HPO] Target_cat_feats : {target_cat_feats}")
    print(f"[HPO] num_feats        : {num_feats}")
    feats = num_feats + target_cat_feats + ordinal_cat_feats
    X_train = train_df[feats]
    y_train = train_df[target_column]
    print(f"[HPO] Train Columns: {X_train.shape}")

    #X_valid = valid_df.drop(columns=[target_column])
    X_valid = valid_df[feats]
    y_valid = valid_df[target_column]
    print(f"Valid Columns: {X_valid.shape}")

    #for col in categorical_features:
    #    X_train[col] = X_train[col].astype('category')
    #    X_valid[col] = X_valid[col].astype('category')

    #print(f"X_train: {X_train.dtypes}")
    #print(f"X_valid: {X_valid.dtypes}")
    
    if base_params is None:
        base_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'enable_categorical': True,
            'random_state': 42
        }
        
    
    def objective(trial):
        params = base_params.copy()
        params.update({
            #'objective': 'reg:squarederror',
            #'tree_method': 'hist',
            #'device': device,
            #'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            #'lambda': trial.suggest_loguniform('lambda', 7.0, 17.0),
            #'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
            
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'lambda': trial.suggest_float('lambda', 7.0, 17.0, log=True),
            'alpha': trial.suggest_float('alpha', 7.0, 17.0, log=True),
            'eta': trial.suggest_categorical('eta', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'gamma': trial.suggest_categorical('gamma', [18, 19, 20, 21, 22, 23, 24, 25]),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        })

        print(f"[HPO] Params: {params}, Enable_categorical")
        enc = make_column_transformer(
            #(StandardScaler(), numerical_features),
            ('passthrough', num_feats),
            (TargetEncoder(), target_cat_feats),
            (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
             ordinal_cat_feats),
            remainder="passthrough",
            verbose_feature_names_out=True,
        )
        feature_types = ["c" if fn in ordinal_cat_feats else "q" for fn in X_train.columns]
        print(f"[HPO] Feature Types: {ordinal_cat_feats}")
        # Detecta la GPU
        gpu_id = detect_gpu()
        if gpu_id >= 0:
            print(f"[HPO] Detected GPU, using GPU {gpu_id}.")
            params['device'] = 'cuda'
            steps = [enc] #[enc, FunctionTransformer(move_to_gpu, validate=False)]
        else:
            print("[HPO] No GPU detected, using CPU.")
            params['device'] = 'cpu'
            steps = [enc]
        
        reg = xgb.XGBRegressor(
            **params, 
            feature_types=feature_types,
            enable_categorical=True
        )
        #pipeline = make_pipeline(enc, reg)
        steps.append(reg)
        pipeline = make_pipeline(*steps)
        pipeline.fit(
            X_train, y_train,             
            #cls__eval_set=[(X_valid, y_valid)], 
            #cls__early_stopping_rounds=50, 
            #cls__enable_categorical=True,
            #cls__verbose=False
        )
        # check XGBoost is using the feature type correctly.
        model_types = reg.get_booster().feature_types
        assert model_types is not None
        for a, b in zip(model_types, feature_types):
            assert a == b
        
        y_pred = pipeline.predict(X_valid)
        val_rmse = mean_squared_error(y_pred, y_valid, squared=False)
        return val_rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    #return study.best_params
    return {
        'best_params': study.best_params,
        'results': study.trials
    }


seed=42
max_n_trials=50
hpo_base_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': seed
}
hpo_base_params


%%time
hpo_xgb_results = optimize_xgb_hyperparameters(
    train, valid,
    target_column, 
    num_feats=num_feats, 
    ordinal_cat_feats=ordinal_cat_feats, 
    target_cat_feats=target_cat_feats,
    base_params=hpo_base_params,
    #n_trials=50
    n_trials=max_n_trials
)


hpo_xgb_results['best_params']


hpo_xgb_results['results'][0]


hpo_opt_params = hpo_xgb_results['best_params']
hpo_opt_params


opt_params = {**hpo_base_params, **hpo_opt_params}
opt_params


%%time
opt_xgb_results = train_xgb(
    train, 
    valid,
    target_column,
    num_feats=num_feats, 
    ordinal_cat_feats=ordinal_cat_feats, 
    target_cat_feats=target_cat_feats,
    params=opt_params
)


base_xgb['metrics']


opt_xgb_results['metrics']


opt_xgb_metrics = opt_xgb_results['metrics']
opt_xgb_model = opt_xgb_results['model']


opt_valid_preds = opt_xgb_model.predict(valid[all_feats])
valid['opt_preds'] = opt_valid_preds


valid[[target_column, 'base_preds', 'opt_preds']]


valid[[target_column, 'base_preds', 'opt_preds']]


valid[[target_column, 'base_preds', 'opt_preds']][valid.class_name.isnull()]


# Saving
# Guardar el modelo usando pickle
print(f"Saving model with pickle. {OPTIMIZED_MODEL_SAVE_PATH}")
with open(OPTIMIZED_MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(opt_xgb_model, f)


train[all_feats + [target_column]].to_csv('train.csv', index=False)


valid[all_feats + [target_column]].to_csv('valid.csv', index=False)


!ls -lh models


raw_test = read_test_data(
    TEST_FILE,
    STORES_FILE,
    CATALOG_FILE
)


ini_test = pd.read_csv(TEST_FILE, index_col=0, sep=';')
#catalog = pd.read_csv(CATALOG_FILE)


raw_test.shape, ini_test.shape, raw_test.shape[0] == ini_test.shape[0]


ini_test.head(5)


raw_test.head(5)


raw_test.isnull().sum()/len(raw_test)


test = run_data_wrangling(raw_test)


test.head(5)


test[test.class_name.isnull()].store_id.unique()


def make_submission(
    model,
    test,
):
    preds = model.predict(test)
    submission = pd.DataFrame({
        'row_id':test.index.values,
        'quantity':preds
    })
    return submission


test[all_feats].to_csv('test.csv', index=False)


submission = make_submission(
    opt_xgb_model,
    test[all_feats],
)


print(f"submission: {submission.shape}")
submission.head(5)


ini_test.shape, test.shape ,submission.shape


submission.to_csv("submission.csv", index=False)


!ls -lh outputs




