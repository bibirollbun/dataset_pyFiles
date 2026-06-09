#!pip install autogluon.tabular -q
# from autogluon.tabular import TabularPredictor


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


PATH = "/kaggle/input/playground-series-s5e5"
SUBMISSION_FILENAME = "sample_submission.csv"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

TARGET = "Calories"

SUBMISSION_DIR = os.path.join(PATH, SUBMISSION_FILENAME)
TRAIN_DIR = os.path.join(PATH, TRAIN_FILENAME) 
TEST_DIR = os.path.join(PATH, TEST_FILENAME)

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


def synthetic_variables_engineering(df_i):
    df = df_i.copy()
    df["BMI"] = df["Weight"] / ((df["Height"]/100) ** 2)
    df["Age_Category"] = pd.cut(
        df["Age"],
        bins=[0, 20, 40, 60, 80, 120],
        labels=['<20', '20-40', '40-60', '60-80', '80+']
    )
    # df["Intensity"] = df[TARGET] / df["Duration"].replace(0, np.nan)
    df["Weight_Category"] = pd.cut(df["Weight"],
                                  bins=[0, 50, 70, 90, 110, 200],
                                  labels=['<50', '50-70', '70-90', '90-110', '110+'])
    df["Heart_Index"] = df["Heart_Rate"] / df["Age"].replace(0, np.nan)
    df["BSA"] = np.sqrt((df["Height"] * df["Weight"])/3600)
    return df


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


train_df = pd.read_csv(TRAIN_DIR, index_col = "id")
test_df = pd.read_csv(TEST_DIR, index_col = "id")
submission_df = pd.read_csv(SUBMISSION_DIR, index_col = "id")


train_df.head()


# data_description(train_df)
# data_description(test_df)


train_df


data_description(train_df)
data_description(test_df)


train_df_syn = synthetic_variables_engineering(train_df)
test_df_syn =  synthetic_variables_engineering(test_df)


train_df_male = train_df[train_df['Sex'] == 'male']
train_df_female = train_df[train_df['Sex'] == 'female']

test_df_male = test_df[test_df['Sex'] == 'male']
test_df_female = test_df[test_df['Sex'] == 'female']


train_df_male = train_df_male.drop(columns=["Sex"])
train_df_female = train_df_female.drop(columns=["Sex"])


show_corr_heatmap(train_df_male, "Train heatmap")
show_corr_heatmap(train_df_female, "Test heatmap")


numeric_features = train_df_male.columns
high_corr_features = ["Duration", "Heart_Rate", "Body_Temp"]


from sklearn.linear_model import LinearRegression

def var_correlation(df): 
    correlation_series = df.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
    
    manual_results = []
    
    for col in numeric_features:
        x = df[col]
        y = df[TARGET]
        beta = np.cov(x, y)[0, 1] / np.var(x)
        y_pred = beta * x
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        manual_results.append({'Variable': col, 'Coeficiente (Beta)': beta, 'R^2': r2})
    
    manual_df = pd.DataFrame(manual_results).sort_values(by='R^2', ascending=False)
    
    manual_df.reset_index(drop=True)
    return manual_df


var_correlation(train_df_male)


var_correlation(train_df_female)


numeric_features


# Crear subplots
fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

for i, feature in enumerate(high_corr_features):
    sns.scatterplot(
        data=train_df,
        x=feature,
        y=TARGET,
        hue='Sex',
        palette= PALETTE_7_C,
        ax=axis[i],
        s=15,
        alpha=0.6
    )
    
    # LÃ­nea de regresiÃ³n para male
    sns.regplot(
        data=train_df[train_df['Sex'] == 'male'],
        x=feature,
        y=TARGET,
        ax=axis[i],
        scatter=False,
        color=PALETTE_7_C[0],
        label='Male trend'
    )

    # LÃ­nea de regresiÃ³n para female
    sns.regplot(
        data=train_df[train_df['Sex'] == 'female'],
        x=feature,
        y=TARGET,
        ax=axis[i],
        scatter=False,
        color=PALETTE_7_C[1],
        label='Female trend'
    )

    axis[i].set_title(f'Calories vs {feature} por Sexo')
    axis[i].set_xlabel(feature)
    axis[i].set_ylabel('Calories')
    axis[i].legend()

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_variable_vs_calories(df, variables, point_color, line_color, title_prefix=""):
    """
    Genera scatterplots con lÃ­nea de regresiÃ³n para cada variable vs 'Calories'.

    ParÃ¡metros:
    - df: DataFrame (ya separado por sexo y sin la columna 'Sex')
    - variables: lista de nombres de columnas a comparar con 'Calories'
    - point_color: color de los puntos del scatterplot
    - line_color: color de la lÃ­nea de regresiÃ³n
    - title_prefix: string opcional para el tÃ­tulo (e.g., 'Mujeres', 'Hombres')
    """
    num_vars = len(variables)
    cols = min(num_vars, 3)  # mÃ¡ximo 3 por fila
    rows = math.ceil(num_vars / cols)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if num_vars > 1 else [axes]

    for i, feature in enumerate(variables):
        sns.scatterplot(
            data=df,
            x=feature,
            y='Calories',
            color=point_color,
            ax=axes[i],
            s=15,
            alpha=0.6
        )

        sns.regplot(
            data=df,
            x=feature,
            y='Calories',
            ax=axes[i],
            scatter=False,
            color=line_color,
            label='Tendencia'
        )

        axes[i].set_title(f'Calories vs {feature} ({title_prefix})')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Calories')
        axes[i].legend()

    # Eliminar subplots vacÃ­os si hay
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



extra_features = ['Age', 'Height', 'Weight']
plot_variable_vs_calories(train_df_male, extra_features, PALETTE_7_C[1], PALETTE_7_C[0], "Male regplot")


plot_variable_vs_calories(train_df_female, extra_features, PALETTE_7_C[0], PALETTE_7_C[1], "Female regplot")


all_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Variables objetivo
X_all = train_df[all_features]
X_top3 = train_df[high_corr_features]
y = train_df[TARGET]

# Ajustar modelos de regresiÃ³n lineal
def fit_manual_linear_regression(X, y):
    X = np.c_[np.ones(len(X)), X]  # agregar intercepto
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    y_pred = X @ beta
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - ss_res / ss_tot
    return beta, r2

# Modelo con todas las variables
beta_all, r2_all = fit_manual_linear_regression(X_all, y)

# Modelo con solo las tres mÃ¡s importantes
beta_top3, r2_top3 = fit_manual_linear_regression(X_top3, y)

# Preparar resultados
results = pd.DataFrame({
    'Modelo': ['All variables', 'Only the 3 main variables'],
    'R^2': [r2_all, r2_top3]
})

results


X_all_male = train_df_male[all_features]
X_top3_male = train_df_male[high_corr_features]
y_male = train_df_male[TARGET]

X_all_female = train_df_female[all_features]
X_top3_female = train_df_female[high_corr_features]
y_female = train_df_female[TARGET]
# Modelo con todas las variables
beta_all_male, r2_all_male = fit_manual_linear_regression(X_all_male, y_male)

# Modelo con solo las tres mÃ¡s importantes
beta_top3_male, r2_top3_male = fit_manual_linear_regression(X_top3_male, y_male)


beta_all_female, r2_all_female = fit_manual_linear_regression(X_all_female, y_female)

# Modelo con solo las tres mÃ¡s importantes
beta_top3_female, r2_top3_female = fit_manual_linear_regression(X_top3_female, y_female)

# Preparar resultados
results = pd.DataFrame({
    'Modelo': ['All variables (male)', 'Only the 3 main variables(male)', 'All variables(female)', 'Only the 3 main variables(female)'],
    'R^2': [r2_all_male, r2_top3_male, r2_all_female, r2_top3_female]
})

results


def model_and_predict_from_df(train_df, test_df, all_features, high_corr_features,
                              sample_n=5000, random_state=42):
    """
    Fits two OLS models (one with all features, one with top 3) on a given training set,
    and predicts Calories on a sample from the test set.

    Assumes both train_df and test_df are already filtered by sex and contain no 'Sex' column.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame (single sex), must contain 'Calories'.
    test_df : pd.DataFrame
        Test DataFrame (same structure as training), must contain 'Calories'.
    all_features : list of str
        List of predictor variable names for full model.
    high_corr_features : list of str
        List of top-3 correlated features for reduced model.
    sample_n : int, optional
        Number of test samples to draw (default=5000).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    -------
    pd.DataFrame
        DataFrame with two prediction columns:
        - 'Pred_Calories_All_Vars'
        - 'Pred_Calories_Top3_Vars'
    """
    # Extract target
    y_train = train_df['Calories'].values

    # --- Full feature model ---
    X_all = train_df[all_features].values
    X_all = np.c_[np.ones(len(X_all)), X_all]  # Add intercept
    beta_all = np.linalg.inv(X_all.T @ X_all) @ (X_all.T @ y_train)

    # --- Top 3 feature model ---
    X_top3 = train_df[high_corr_features].values
    X_top3 = np.c_[np.ones(len(X_top3)), X_top3]
    beta_top3 = np.linalg.inv(X_top3.T @ X_top3) @ (X_top3.T @ y_train)

    # --- Sample test set ---
    sample_test = test_df.sample(n=sample_n, random_state=random_state)

    Xs_all = sample_test[all_features].values
    Xs_all = np.c_[np.ones(len(Xs_all)), Xs_all]

    Xs_top3 = sample_test[high_corr_features].values
    Xs_top3 = np.c_[np.ones(len(Xs_top3)), Xs_top3]

    # --- Generate predictions ---
    y_pred_all = Xs_all @ beta_all
    y_pred_top3 = Xs_top3 @ beta_top3

    # --- Output DataFrame ---
    predictions_df = pd.DataFrame({
        'Pred_Calories_All_Vars': y_pred_all,
        'Pred_Calories_Top3_Vars': y_pred_top3
    }).reset_index(drop=True)

    return predictions_df





male_predictions = model_and_predict_from_df(
    train_df=train_df_male,
    test_df=test_df_male,
    all_features=all_features,
    high_corr_features=high_corr_features
)

female_predictions = model_and_predict_from_df(
    train_df=train_df_female,
    test_df=test_df_female,
    all_features=all_features,
    high_corr_features=high_corr_features
)


male_predictions


female_predictions


all_features_plus = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex']


train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "famale": 1})


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Dividir el conjunto de entrenamiento en entrenamiento y validaciÃ³n
X = train_df[all_features_plus]
y = train_df[TARGET]

X_male = train_df_male[all_features]
y_male = train_df_male[TARGET]

X_female = train_df_female[all_features]
y_female = train_df_female[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train_male, X_val_male, y_train_male, y_val_male = train_test_split(X_male, y_male, test_size=0.2, random_state=SEED)
X_train_female, X_val_female, y_train_female, y_val_female = train_test_split(X_female, y_female, test_size=0.2, random_state=SEED)

# Entrenar modelo de regresiÃ³n lineal mÃºltiple
model = LinearRegression()
model.fit(X_train, y_train)

model_male = LinearRegression()
model_male.fit(X_train_male, y_train_male)

model_female = LinearRegression()
model_female.fit(X_train_male, y_train_male)

# Predecir sobre el conjunto de validaciÃ³n
y_pred_val = model.predict(X_val)
y_pred_val_male = model_male.predict(X_val_male)
y_pred_val_female = model_female.predict(X_val_female)
# Calcular mÃ©tricas
r2 = r2_score(y_val, y_pred_val)
r2_male = r2_score(y_val_male, y_pred_val_male)
r2_female = r2_score(y_val_female, y_pred_val_female)

rmse = mean_squared_error(y_val, y_pred_val, squared=False)
rmse_male = mean_squared_error(y_val_male, y_pred_val_male, squared=False)
rmse_female = mean_squared_error(y_val_female, y_pred_val_female, squared=False)

# Mostrar resultados
print("Model Evaluation Metrics".center(50, "="))
print(f"{'Metric':<20}{'Overall':>10}{'Male':>10}{'Female':>10}")
print("-" * 50)
print(f"{'RÂ² Score':<20}{r2:>10.4f}{r2_male:>10.4f}{r2_female:>10.4f}")
print(f"{'RMSE':<20}{rmse:>10.2f}{rmse_male:>10.2f}{rmse_female:>10.2f}")
print("=" * 50)


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{name:<20} - RÂ²: {r2:.4f}, RMSE: {rmse:.2f}")
    return r2, rmse

# ----------------------------
# GLOBAL DATASET
# ----------------------------
X = train_df[all_features]
y = train_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

r2_rf, rmse_rf = evaluate("Random Forest (All)", y_test, rf_preds)
r2_xgb, rmse_xgb = evaluate("XGBoost (All)", y_test, xgb_preds)

# ----------------------------
# MALE DATASET
# ----------------------------
X_male = train_df_male[all_features]
y_male = train_df_male[TARGET]
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2, random_state=SEED)

rf_model_male = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf_model_male.fit(X_train_male, y_train_male)
rf_preds_male = rf_model_male.predict(X_test_male)

xgb_model_male = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED)
xgb_model_male.fit(X_train_male, y_train_male)
xgb_preds_male = xgb_model_male.predict(X_test_male)

r2_rf_male, rmse_rf_male = evaluate("Random Forest (Male)", y_test_male, rf_preds_male)
r2_xgb_male, rmse_xgb_male = evaluate("XGBoost (Male)", y_test_male, xgb_preds_male)

# ----------------------------
# FEMALE DATASET
# ----------------------------
X_female = train_df_female[all_features]
y_female = train_df_female[TARGET]
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2, random_state=SEED)

rf_model_female = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf_model_female.fit(X_train_female, y_train_female)
rf_preds_female = rf_model_female.predict(X_test_female)

xgb_model_female = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED)
xgb_model_female.fit(X_train_female, y_train_female)
xgb_preds_female = xgb_model_female.predict(X_test_female)

r2_rf_female, rmse_rf_female = evaluate("Random Forest (Female)", y_test_female, rf_preds_female)
r2_xgb_female, rmse_xgb_female = evaluate("XGBoost (Female)", y_test_female, xgb_preds_female)

# ----------------------------
# OPTIONAL: Summary print block
# ----------------------------
print("\n" + "Summary of RÂ² Scores".center(50, "-"))
print(f"{'Model':<25}{'All':>8}{'Male':>8}{'Female':>8}")
print(f"{'Random Forest':<25}{r2_rf:>8.4f}{r2_rf_male:>8.4f}{r2_rf_female:>8.4f}")
print(f"{'XGBoost':<25}{r2_xgb:>8.4f}{r2_xgb_male:>8.4f}{r2_xgb_female:>8.4f}")



numerical_cols = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'BMI', 'Heart_Index', 'BSA'
]

categorical_cols = ['Age_Category', 'Weight_Category']
for col in categorical_cols:
    train_df_syn[col] = train_df_syn[col].astype('category')

all_features = numerical_cols + categorical_cols

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


X = train_df_syn[all_features]
y = train_df_syn[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


lgbm_params = {
    'n_estimators': 1125,
    'num_leaves': 110,
    'min_child_samples': 9,
    'learning_rate': 0.0179455702408711,
    'colsample_bytree': 0.5979737441060009,
    'reg_alpha': 0.001975258376030875,
    'reg_lambda': 0.005106256873241264,
    'max_bin': 2**10,
    'random_state': SEED,
    'verbose': -1
}


model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LGBMRegressor(**lgbm_params))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
evaluate("LightGBM", y_test, preds)


test_df = pd.read_csv(TEST_DIR)
train_df = pd.read_csv(TRAIN_DIR)


for col in ['Age_Category', 'Weight_Category']:
    test_df_syn[col] = test_df_syn[col].astype('category')


X_test_final = test_df_syn[all_features]
y_pred_test = model.predict(X_test_final)


y_pred_test.head()


y_pred_test.to_csv("submission.csv", index=False)


# test_df_male = test_df[test_df['Sex'] == 'male'][['id'] + all_features].copy()
# test_df_female = test_df[test_df['Sex'] == 'female'][['id'] + all_features].copy()
train_df_syn["id"] = train_df["id"]
test_df_syn["id"] = test_df["id"]

test_df_male = train_df_syn[train_df_syn['Sex'] == 'male'][['id'] + all_features].copy()
test_df_female = test_df_syn[test_df_syn['Sex'] == 'female'][['id'] + all_features].copy()


test_df_male['Calories'] = rf_model_male.predict(test_df_male[all_features])
test_df_female['Calories'] = xgb_model_female.predict(test_df_female[all_features])

# Combinar y ordenar por 'id' como en el sample_submission
submission_df = pd.concat([test_df_male[['id', 'Calories']], test_df_female[['id', 'Calories']]])
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)

# Guardar como CSV final
output_path = "submission_.csv"
submission_df.to_csv(output_path, index=False)



submission_df


# --- 4. IMPORTANCIA DE VARIABLES ---
importances = pd.Series(rf_model.feature_importances_, index=all_features)
importances.sort_values().plot(kind='barh', title='Importancia de Variables - Random Forest')
plt.tight_layout()
plt.show()

# --- 5. PREDICCIONES EN TEST ---
X_test = test_df[all_features]
rf_test_pred = rf_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

# --- 6. GUARDAR CSV DEL MEJOR MODELO ---
best_model_preds = rf_test_pred if r2_rf > r2_xgb else xgb_test_pred
submission_df[TARGET] = best_model_preds 

