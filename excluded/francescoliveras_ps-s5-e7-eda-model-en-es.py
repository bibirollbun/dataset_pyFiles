import importlib.util
import subprocess
import sys

package_name = "autogluon"

# Verifica si el paquete ya estÃ¡ instalado
if importlib.util.find_spec(package_name) is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
else:
    print(f"'{package_name}' ya estÃ¡ instalado.")


import os 
import re
import gc
import sys
import math
import time
import random
import warnings
import datetime
import numpy as np 
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import missingno as msno
import plotly.express as px
import category_encoders as ce
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from itertools import combinations

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from autogluon.tabular import TabularPredictor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Put theme of notebook 
from colorama import Fore, Style

# Colors
red = Fore.RED + Style.BRIGHT
mgta = Fore.MAGENTA + Style.BRIGHT
yllw = Fore.YELLOW + Style.BRIGHT
cyn = Fore.CYAN + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT

# Reset
res = Style.RESET_ALL
plt.style.use({"figure.facecolor": "#282a36"})


# Colors
YELLOW = "#F7C53E"

CYAN_G = "#0CF7AF"
CYAB_DARK = "#11AB7C"

PURPLE = "#D826F8"
PURPLE_DARJ = "#9309AB"
PURPLE_L = "#b683d6"

BLUE = "#0C97FA"
RED = "#FA1D19"
ORANGE = "#FA9F19"
GREEN = "#0CFA58"
LIGTH_BLUE = "#01FADC"
S_BLUE = "#81c9e6"
DARK_BLUE = "#394be6"
# Palettes
PALETTE_2 = [CYAN_G, PURPLE]
PALETTE_3 = [YELLOW, CYAN_G, PURPLE]
PALETTE_4 = [YELLOW, ORANGE, PURPLE, LIGTH_BLUE]
PALETTE_5 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE]
PALETTE_6 = [BLUE, RED, ORANGE, GREEN, LIGTH_BLUE, PURPLE]

# Vaporwave palette by Francesc Oliveras
PALETTE_7 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE, DARK_BLUE, S_BLUE]
PALETTE_7_C = [PURPLE_DARJ, BLUE, PURPLE, LIGTH_BLUE, PURPLE_L, S_BLUE, DARK_BLUE]
sns.palplot(sns.color_palette(PALETTE_7))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

cmap = mcolors.LinearSegmentedColormap.from_list("", PALETTE_2)
cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])

font_family = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=10), width=1000, height=500))

warnings.filterwarnings('ignore')


PATH = "/kaggle/input/playground-series-s5e7"
SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

TARGET = "Personality"

SUBMISSION_DIR = os.path.join(PATH, SUBMISSION_FILENAME)
TRAIN_DIR = os.path.join(PATH, TRAIN_FILENAME) 
TEST_DIR = os.path.join(PATH, TEST_FILENAME)
ORIGINAL_DIR = '/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv'

SEED = 180


def show_corr_heatmap(df, title):
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize = (15, 10))
    plt.title(title)
    # sns.heatmap(corr, annot = False, linewidths=.5, fmt=".2f", square=True, mask = mask, cmap=cmap_2)
    if df.shape[1] < 25:
        sns.heatmap(corr, annot=True, linewidths=.5, fmt=".2f", square=True, mask=mask, cmap=cmap_2)
    else:
        sns.heatmap(corr, annot=False, linewidths=.5, square=True, mask=mask, cmap=cmap_2)

    plt.show()


def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


def plot_cont(col, ax, color=PALETTE_7[0]):
    sns.histplot(data=comb_df, x=col,
                hue="set",ax=ax, hue_order=labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_cont_dot(
        comb_df.query('set=="train"'),
        col, TARGET, ax_2,
        color=color
    )
    
    ax_2 = plot_cont_dot(
        comb_df, col,
        TARGET, ax_2,
        color=color
    )


def show_pie_mult(dataframe, target = TARGET):
    target_counts = dataframe[target].sum()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target, autopct='%1.1f%%', startangle=140, colors=PALETTE_7_C)

    # Agregando un cÃ­rculo blanco en el centro para hacer un agujero
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_pie_categorical(dataframe, target=TARGET):
    target_counts = dataframe[target].value_counts()

    # Creando el grÃ¡fico de pastel con un agujero en el centro
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=[PALETTE_7_C[0],PALETTE_7_C[1],PALETTE_7_C[2],
                                                                                                                            PALETTE_7_C[3],PALETTE_7_C[4],PALETTE_7_C[5]])

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Ajustando el aspecto para que sea un cÃ­rculo y mostrando el grÃ¡fico
    plt.title('DistribuciÃ³n de los Targets')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def show_box_plot(dataframe):
    # numerical_features_for_boxplot = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('id')
    numerical_features_for_boxplot = dataframe.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(20, 15))

    for i, feature in enumerate(numerical_features_for_boxplot, 1):
        plt.subplot(7, 5, i)
        sns.boxplot(y=train_df[feature], color=PALETTE_7_C[i % len(PALETTE_7_C)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


def show_hist(dataframe):
    # Filtrando las columnas numÃ©ricas para sus histogramas
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns

    # Configurando el tamaÃ±o de la figura
    plt.figure(figsize=(20, 15))

    # Creando un histograma para cada caracterÃ­stica numÃ©rica
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(7, 5, i) # Ajustar segÃºn el nÃºmero de caracterÃ­sticas numÃ©ricas
        dataframe[feature].hist(bins=20, color=PALETTE_7_C[int(i%7)])
        plt.title(feature)

    plt.tight_layout()
    plt.show()


def apply_one_hot_encoding(df, threshold=10):
    """
    Apply One-Hot Encoding to categorical columns with unique values below a given threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    threshold (int): The maximum number of unique values for a column to be eligible for OHE.

    Returns:
    pd.DataFrame: The dataframe with One-Hot Encoding applied to selected columns.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Determine columns eligible for OHE
    ohe_columns = [col for col in categorical_columns if df[col].nunique() < threshold]
    
    # Apply One-Hot Encoding
    df_ohe = pd.get_dummies(df, columns=ohe_columns, drop_first=True)
    
    return df_ohe


def apply_label_encoding(df, columns):
    """
    Apply Label Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Label Encoding.

    Returns:
    pd.DataFrame: DataFrame with Label Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        label_encoder = LabelEncoder()
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    return df_encoded


def apply_frequency_encoding(df, columns):
    """
    Apply Frequency Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Frequency Encoding.

    Returns:
    pd.DataFrame: DataFrame with Frequency Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df_encoded[col] = df[col].map(freq_map)
    return df_encoded


def apply_hashing_encoding(df, columns, n_features=8):
    """
    Apply Hashing Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Hashing Encoding.
    n_features (int): Number of features to generate for each column.

    Returns:
    pd.DataFrame: DataFrame with Hashing Encoding applied.
    """
    df_encoded = df.copy()
    for col in columns:
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed_features = hasher.transform(df_encoded[col].astype(str))
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{col}_hash_{i}" for i in range(n_features)])
        df_encoded = df_encoded.drop(columns=[col]).join(hashed_df)
    return df_encoded


def apply_binary_encoding(df, columns):
    """
    Apply Binary Encoding to specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to apply Binary Encoding.

    Returns:
    pd.DataFrame: DataFrame with Binary Encoding applied.
    """
    binary_encoder = ce.BinaryEncoder(cols=columns)
    df_encoded = binary_encoder.fit_transform(df)
    return df_encoded


def apply_direct_binary(df, columns, positive_value="Yes", negative_value="No"):
    """
    Convierte columnas binarias (p. ej. "Yes"/"No") a 1/0.
    """
    mapping = {positive_value: 1, negative_value: 0}
    for col in columns:
        # Mapear, filtrar los NaN resultantes y luego convertir
        s = df[col].map(mapping)
        df = df[s.notna()]                # eliminar filas con NaN en esta columna
        df[col] = s[s.notna()].astype(int)
    return df


def encode_binary_column(df, column, mapping, nullable=True):
    """
    Mapea los valores de `column` segÃºn `mapping` (p.e. {"Extrovert":1, "Introvert":0})
    y convierte al tipo entero. Si nullable=True usa Int64 para admitir pd.NA.
    """
    s = df[column].map(mapping)
    if nullable:
        df[column] = s.astype("Int64")    # admite valores nulos
    else:
        df[column] = s.fillna(0).astype(int)
    return df


train_df = pd.read_csv(TRAIN_DIR, index_col="id")
test_df = pd.read_csv(TEST_DIR, index_col = "id")
original_df = pd.read_csv(ORIGINAL_DIR)
submission_df = pd.read_csv(SUBMISSION_DIR, index_col = "id")


original_df.head()


train_df.head()


data_description(train_df)
data_description(test_df)
data_description(original_df)


num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]
cat_cols = ["Stage_fear", "Drained_after_socializing"]


display(show_pie_categorical(train_df, TARGET))
display(show_pie_categorical(train_df, "Stage_fear"))
display(show_pie_categorical(train_df, "Drained_after_socializing"))


X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_original = original_df.drop(columns=[TARGET])
y_original = original_df[TARGET]


from sklearn.experimental import enable_iterative_imputer  # si usas IterativeImputer
from sklearn.impute       import SimpleImputer, IterativeImputer
from sklearn.pipeline     import Pipeline
from sklearn.compose      import ColumnTransformer

# 2.1. Detecta quÃ© columnas son numÃ©ricas / categÃ³ricas en X_train
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

# 2.2. Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing'))
])

preprocessor = ColumnTransformer([
    ('nums', num_pipeline, num_cols),
    ('cats', cat_pipeline, cat_cols),
], remainder='drop')

# 2.3. Fit sÃ³lo en train y transforma ambos
preprocessor.fit(X_train)
Xt_train = preprocessor.transform(X_train)
Xt_test  = preprocessor.transform(test_df)
Xt_original = preprocessor.transform(original_df)


# 3.1. Extrae los nombres completos
feature_names = preprocessor.get_feature_names_out()
# suelen venir como "nums__Going_outside" o "cats__SomeCat"
# 3.2. Limpia los prefijos
clean_names = [name.split('__', 1)[1] for name in feature_names]

# 3.3. Crea DataFrames alineados con los Ã­ndices originales  
X_train_pre = pd.DataFrame(Xt_train, columns=clean_names, index=X_train.index)
X_test_pre  = pd.DataFrame(Xt_test,  columns=clean_names, index=test_df.index)
X_original_pre  = pd.DataFrame(Xt_original,  columns=clean_names, index=X_original.index)


# 4.1. Train: aÃ±ades la columna real
train_prepared = X_train_pre.copy()
train_prepared[TARGET] = y_train

original_prepared = X_train_pre.copy()
original_prepared[TARGET] = y_original

# 4.2. Test: la rellenas con NaN (porque no la conoces)
test_prepared = X_test_pre.copy()
test_prepared[TARGET] = np.nan

# Ahora train_prepared y test_prepared tienen IDENTICAMENTE
# las mismas columnas en el mismo orden.



train_prepared


original_prepared


original_prepared = original_prepared.dropna(subset=[TARGET])


train_prepared = apply_direct_binary(train_prepared, cat_cols)
train_prepared = encode_binary_column(
    train_prepared,
    TARGET,
    mapping={"Extrovert": 1, "Introvert": 0},
    nullable=False  
)

original_prepared = apply_direct_binary(original_prepared, cat_cols)
original_prepared = encode_binary_column(
    original_prepared,
    TARGET,
    mapping={"Extrovert": 1, "Introvert": 0},
    nullable=False  
)

test_prepared = apply_direct_binary(test_prepared, cat_cols)


train_prepared.head()


print(train_prepared.isna().sum())


train_prepared.to_csv("train_df.csv")


show_corr_heatmap(train_prepared, "Train heatmap")
show_corr_heatmap(original_prepared, "Original heatmap")
show_corr_heatmap(test_prepared, "Test heatmap")


def get_top_correlations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Devuelve los n pares de variables (distintas) con mayor correlaciÃ³n absoluta.
    
    ParÃ¡metros:
        df: DataFrame con las variables numÃ©ricas.
        n: nÃºmero de parejas a extraer (por defecto 5).
        
    Retorna:
        DataFrame con columnas ['var1','var2','corr'] ordenado por |corr| descendente.
    """
    corr_mat = df.corr().abs()
    # Enmascarar diagonal y duplicados
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    pairs = (
        corr_mat.where(mask)
                .stack()
                .reset_index()
                .rename(columns={'level_0':'var1', 'level_1':'var2', 0:'corr'})
                .sort_values('corr', ascending=False)
    )
    return pairs.head(n)

def plot_correlation_heatmap(df: pd.DataFrame, vars: list = None, figsize=(8,6)):
    """
    Dibuja un heatmap de correlaciones, pudiendo limitarlo a un subconjunto de variables.
    
    ParÃ¡metros:
        df: DataFrame de sÃ³lo numÃ©ricas.
        vars: lista de nombres de columnas a incluir. Por defecto: todas.
        figsize: tupla con tamaÃ±o de figura.
    """
    if vars is not None:
        corr = df[vars].corr()
    else:
        corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                linewidths=0.5, cbar_kws={"shrink":.8})
    plt.title("Heatmap de correlaciones")
    plt.tight_layout()
    plt.show()

def plot_top_pair_scatter(df: pd.DataFrame, top_pairs: pd.DataFrame):
    """
    Para cada par listado en top_pairs (DataFrame con columnas var1,var2),
    dibuja un scatter plot con lÃ­nea de regresiÃ³n.
    """
    for _, row in top_pairs.iterrows():
        x, y, c = row['var1'], row['var2'], row['corr']
        plt.figure(figsize=(5,4))
        sns.regplot(data=df, x=x, y=y, scatter_kws={"s":10, "alpha":.6})
        plt.title(f"{x} vs {y} (corr={c:.2f})")
        plt.tight_layout()
        plt.show()


top5_train = get_top_correlations(train_prepared, n=5)
top5_train


top5_original = get_top_correlations(original_prepared, n=5)
top5_original



def show_stats(df,
               palette: list = PALETTE_7_C,
               boxplot_figsize: tuple = (5,4),
               barplot_figsize: tuple = (5,4)):
    """
    Muestra:
      1) Un boxplot para cada variable numÃ©rica.
      2) Un diagrama de barras para cada variable categÃ³rica.
      
    ParÃ¡metros:
    -----------
    df : pd.DataFrame
        DataFrame a analizar.
    palette : list
        Lista de colores para los diagramas de barras.
    boxplot_figsize : tuple
        TamaÃ±o de figura para los boxplots.
    barplot_figsize : tuple
        TamaÃ±o de figura para los barplots.
    """
    # 1) Detectar variables numÃ©ricas y categÃ³ricas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # AÃ±adir numÃ©ricas discretas con pocos niveles (<10) a categÃ³ricas
    for col in list(numeric_cols):
        if df[col].nunique() < 10:
            categorical_cols.append(col)
    
    # Excluir categÃ³ricas de la lista de numÃ©ricas
    numeric_cols = [c for c in numeric_cols if c not in categorical_cols]
    
    # 2) Boxplots para numÃ©ricas
    for col in numeric_cols:
        plt.figure(figsize=boxplot_figsize)
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f'Boxplot de {col}', fontweight='bold')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    
    # 3) Barras para categÃ³ricas con paleta personalizada
    for col in categorical_cols:
        counts = df[col].value_counts().sort_index()
        n = len(counts)
        # ciclo de colores si hay mÃ¡s categorÃ­as que colores
        colors = [palette[i % len(palette)] for i in range(n)]
        
        plt.figure(figsize=barplot_figsize)
        plt.bar(counts.index.astype(str), counts.values, color=colors)
        plt.title(f'Diagrama de barras de {col}', fontweight='bold')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


show_stats(train_prepared)


show_stats(original_prepared)


# train = pd.read_csv(TRAIN_DIR)
# test = pd.read_csv(TEST_DIR)

train = train_prepared
test = test_prepared
original = pd.read_csv(ORIGINAL_DIR)


X_train = train.drop(columns=["id", TARGET], axis=1, errors="ignore")
X_test = test.drop(columns=["id"], axis=1, errors="ignore")
X_origin = original.drop(columns=["id", TARGET], axis=1, errors="ignore")

y_train = train[TARGET]
y_origin = original[TARGET]


print(f"Shape: {X_train.shape}")
print(f"Target shape: {y_train.shape if y_train is not None else 'None'}")


lab_encoders = {}


for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    X_origin[col] = le.transform(X_origin[col].astype(str))

    lab_encoders[col] = le

target_encoder = LabelEncoder()
y_train_enc = target_encoder.fit_transform(y_train)
y_origin_enc = target_encoder.fit_transform(y_origin)

print(f"{TARGET} class: {target_encoder.classes_}")


scaler = StandardScaler()
feature_names = X_train.columns.tolist()
print(f"Features: {feature_names}")


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_leaves': 25,
    'min_child_weight': np.float64(0.003440906647223279),
    'learning_rate': np.float64(0.09470087254583547),
    'n_estimators': 10000,
    'subsample': np.float64(0.8025291728808135),
    'colsample_bylevel': np.float64(0.8360122952647302),
    'colsample_bytree': np.float64(0.87329448975438),
    'reg_alpha': np.float64(0.002926163798802797),
    'reg_lambda': np.float64(27.126259438996986),
    'random_state': 42,
    'tree_method': 'hist',
    'device': "cuda"
}


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# â€” Asume que X_train, X_original, y_train_enc, y_origin_enc y xgb_params ya estÃ¡n definidos â€”

# 1) Mapeo de las columnas binarias de texto a 0/1, rellenando posibles NaN resultantes
binary_map = {"No": 0, "Yes": 1}
binary_cols = ["Stage_fear", "Drained_after_socializing"]

for col in binary_cols:
    # .map() dejarÃ¡ NaN donde el valor no estÃ© en binary_map; .fillna(0) asigna 0 en esos casos
    X_train[col]    = X_train[col].map(binary_map).fillna(0).astype(int)
    X_original[col] = X_original[col].map(binary_map).fillna(0).astype(int)

# 2) ConfiguraciÃ³n del modelo
model_params = xgb_params.copy()
# No necesitamos enable_categorical porque usamos enteros puros
# model_params["enable_categorical"] = True  # SÃ³lo si usas dtype 'category'

# 3) Stratified K-Fold con Early Stopping
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_scores = []
print(f"Performing {n_splits}-fold Stratified CV with Early Stopping...")

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc), start=1):
    print(f"\nTraining Fold {fold_num}/{n_splits}...")
    
    # Separar train/validation
    X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
    y_fold_train = y_train_enc[train_idx]
    X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True)
    y_fold_val   = y_train_enc[val_idx]
    
    # Concatenar datos originales SI LO DESEAS
    X_fold_train = pd.concat([X_fold_train,
                              X_original.reset_index(drop=True)],
                             axis=0, ignore_index=True)
    y_fold_train = np.concatenate([y_fold_train, y_origin_enc])
    
    # Crear el modelo para este fold
    fold_model = xgb.XGBClassifier(**model_params)
    
    # Entrenar con early stopping
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # PredicciÃ³n y mÃ©trica
    preds = fold_model.predict(X_fold_val)
    acc   = accuracy_score(y_fold_val, preds)
    cv_scores.append(acc)
    
    print(f"  Fold {fold_num} Accuracy : {acc:.4f}")
    print(f"  Best Iteration       : {fold_model.best_iteration}")

# Resultados finales
cv_scores = np.array(cv_scores)
print(f"\nCV scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} Â± {2 * cv_scores.std():.4f}")



best_iterations = []
fold_num = 1

print("Extracting best iterations from each CV fold...")
for train_idx, val_idx in skf.split(X_train, y_train_enc):
    # 1) Separar train/validation para este fold
    X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
    y_fold_train = y_train_enc[train_idx]
    X_fold_val   = X_train.iloc[val_idx].reset_index(drop=True)
    y_fold_val   = y_train_enc[val_idx]
    
    # 2) Concatenar con datos originales (si es necesario)
    X_fold_train = pd.concat(
        [X_fold_train, X_original.reset_index(drop=True)],
        axis=0, ignore_index=True
    )
    y_fold_train = np.concatenate([y_fold_train, y_origin_enc])
    
    # 3) Entrenar modelo temporal para obtener best_iteration
    temp_model = xgb.XGBClassifier(**model_params)
    temp_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    best_iterations.append(temp_model.best_iteration)
    print(f"Fold {fold_num} best iteration: {temp_model.best_iteration}")
    fold_num += 1

# 4) Calcular n_estimators Ã³ptimo (media de best iterations)
optimal_n_estimators = int(np.mean(best_iterations))
print(f"\nOptimal n_estimators (average): {optimal_n_estimators}")
print(f"Range: {min(best_iterations)} - {max(best_iterations)}")

# 5) Entrenar modelo final con n_estimators Ã³ptimo sobre todo el dataset
print(f"\nTraining final model on full dataset with {optimal_n_estimators} estimators...")
model_params_final = model_params.copy()
model_params_final['n_estimators'] = optimal_n_estimators

xgb_model_final = xgb.XGBClassifier(**model_params_final)
xgb_model_final.fit(X_train, y_train_enc)

print("Final model trained on 100% of training data!")


!pip install ace_tools


feature_names = X_train.columns.tolist()

# 2) Construir DataFrame de importancias
feature_importance = pd.DataFrame({
    'feature'   : feature_names,
    'importance': xgb_model_final.feature_importances_
})

# 3) Ordenar de mayor a menor y quedarnos con las Top 10
top10 = (
    feature_importance
    .sort_values('importance', ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# 4) Mostrar resultado
print("Top 10 Most Important Features:")
print(top10)


X_train.head()


test.head()


X_test.to_csv('X_test.csv', index=False)
X_train.to_csv('X_train.csv', index=False)


print("Columnas finales en X_test:", X_test.columns.tolist())
print(X_test.dtypes)
print(X_test.isna().sum())


X_test = test.copy()
for drop_col in ['id', 'Personality']:
    if drop_col in X_test.columns:
        X_test = X_test.drop(columns=[drop_col])

# 2) Asegurarnos de que dtypes son numÃ©ricos (int/float) o category
# (Si tuvieras columnas binarias de texto, mapÃ©alas antes, p.e. {"No":0,"Yes":1})
# Por ejemplo:
# binary_map = {"No":0, "Yes":1}
# for col in ['Stage_fear','Drained_after_socializing']:
#     X_test[col] = X_test[col].map(binary_map).fillna(0).astype(int)

# 3) Obtener predicciones y probabilidades
test_predictions = xgb_model_final.predict(X_test)
test_pred_proba  = xgb_model_final.predict_proba(X_test)

# 4) Recuperar etiquetas originales si existe encoder, sino mapeo manual
try:
    test_pred_labels = target_encoder.inverse_transform(test_predictions)
except NameError:
    test_pred_labels = ['Introvert' if p == 0 else 'Extrovert' for p in test_predictions]

# 5) Crear y guardar el archivo de submission
print("\n=== CREATING SUBMISSION FILE ===")
submission_df = pd.DataFrame({
    'id'          : test['id'],
    'Personality' : test_pred_labels
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

