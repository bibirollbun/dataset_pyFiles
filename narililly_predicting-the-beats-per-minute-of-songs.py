# ==============================================================================
# 1. í™˜ê²½ ì„¤ì • (Environment Setup)
# ==============================================================================
!pip install -q --upgrade \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.2.2 \
    lightgbm==4.3.0 \
    optuna==3.6.1 \
    shap==0.44.1 \
    catboost==1.2.5
# í”„ë¡œì �íŠ¸ì�˜ ì�¬í˜„ì„±ì�„ ë³´ì�¥í•˜ê¸° ìœ„í•´, í•µì‹¬ ë�¼ì�´ë¸ŒëŸ¬ë¦¬ë“¤ì�˜ ë²„ì „ì�„ ëª…ì‹œì �ìœ¼ë¡œ ê³ ì •í•©ë‹ˆë‹¤.
# To guarantee the project's reproducibility, we explicitly pin the versions of the core libraries.


# ==============================================================================
# 2. ë�°ì�´í„° ì¤€ë¹„ (Data Preparation)
# ==============================================================================
# --- Core Libraries ---
import os
import time
import datetime
import warnings
import pandas as pd
import numpy as np

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Machine Learning Models ---
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge

# --- ML Utilities & Preprocessing ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# --- Experimentation & Analysis ---
import optuna
import shap
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Type Hinting ---
from typing import Optional
from pandas import DataFrame, Series
from typing import Tuple
from optuna.trial import Trial
from optuna.study import Study


# ê¸°ë³¸ ì„¤ì • (Settings)
SEED = 2025
N_SPLITS = 5

warnings.filterwarnings('ignore')

def format_time(seconds: float) -> str:
    """
    ì´ˆ(float)ë¥¼ ì�…ë ¥ë°›ì•„ 'ì‹œ:ë¶„:ì´ˆ' (HH:MM:SS) í˜•íƒœì�˜ ë¬¸ì��ì—´ë¡œ ë³€í™˜í•©ë‹ˆë‹¤.
    Converts seconds (float) into a string with the format 'HH:MM:SS'.
    """
    # timedelta ê°�ì²´ëŠ” ì‹œê°„ì�„ ë‹¤ë£¨ëŠ” ë�° ë§¤ìš° í�¸ë¦¬í•œ ë°©ë²•ì�…ë‹ˆë‹¤.
    # The timedelta object is a very convenient way to handle time durations.
    return str(datetime.timedelta(seconds=seconds))


def load_data(base_path: str = '/kaggle/input') -> Tuple[DataFrame, DataFrame, str]:
    """
    ì§€ì •ë�œ ê¸°ë³¸ ê²½ë¡œì—�ì„œ ê²½ì§„ëŒ€íšŒ ë�°ì�´í„°ë¥¼ ì°¾ì•„ ë�°ì�´í„°í”„ë ˆì�„ìœ¼ë¡œ ë¶ˆëŸ¬ì˜µë‹ˆë‹¤.
    Finds and loads the competition data from a specified base path into dataframes.
    """
    start_time = time.time()
    
    data_path = ''
    for dirname, _, filenames in os.walk(base_path):
        if 'train.csv' in filenames:
            data_path = dirname
            break
    
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    
    print("Data loaded successfully!")

    end_time = time.time()
    process_time = end_time - start_time
    print(f"â�±ï¸� Process Time : {format_time(process_time)}")
    
    return train_df, test_df, data_path


train_df, test_df, data_path = load_data()


# ==============================================================================
# 3. íƒ�ìƒ‰ì �ì�¸ ë�°ì�´í„° ë¶„ì„� (Exploratory Data Analysis)
# ==============================================================================
# ë�°ì�´í„°ì�˜ ì „ì²´ì �ì�¸ í�¬ê¸°ì™€ ê·œëª¨ë¥¼ íŒŒì•…í•˜ì—¬ ë¶„ì„� ì „ë�µì�˜ ê¸°ë°˜ì�„ ë§ˆë ¨í•©ë‹ˆë‹¤.
# Grasp the overall size and scale of the data to form a basis for the analysis strategy.
print(f"--- test_df Data Shape ---")
test_df.shape


print(f"--- train_df Data Shape ---")
train_df.shape


print(f"--- Data Head ---")
train_df.head()


print(f"--- Missing Values ---")
train_df.isnull().sum()


print(f"--- Data Info ---")
train_df.info()
# ê°� í”¼ì²˜ì�˜ ë�°ì�´í„° íƒ€ì�…ê³¼ ê²°ì¸¡ì¹˜ ìœ ë¬´ë¥¼ í™•ì�¸í•˜ì—¬ ë�°ì�´í„°ì�˜ ê±´ê°• ìƒ�íƒœë¥¼ ì§„ë‹¨í•©ë‹ˆë‹¤.
# Diagnose the health of the data by checking the data type and presence of nulls for each feature.


print(f"--- Descriptive Statistics ---")
train_df.describe().T
# ìˆ˜ì¹˜í˜• í”¼ì²˜ë“¤ì�˜ ë¶„í�¬(í�‰ê· , í�¸ì°¨, ì‚¬ë¶„ìœ„ìˆ˜ ë“±)ë¥¼ ìš”ì•½í•˜ì—¬ ë�°ì�´í„°ì�˜ íŠ¹ì„±ì�„ íŒŒì•…í•©ë‹ˆë‹¤.
# Understand the data's characteristics by summarizing the distribution (mean, std, quartiles, etc.) of the numerical features.


# ì‹œê°�í™” (Visualization)
# ë¶„ì„�ì—� ì‚¬ìš©í•  ìˆ˜ì¹˜í˜• í”¼ì²˜ë“¤ì�„ ì„ íƒ�í•©ë‹ˆë‹¤. (id ì œì™¸)
# Select the numerical features for analysis (excluding id).
numerical_features = train_df.select_dtypes(include=np.number).drop(columns=['id']).columns

# --- 3.1. ì „ì²´ í”¼ì²˜ í�ˆìŠ¤í† ê·¸ë�¨ ---
print("--- [Visualization] Histograms for all Numerical Features ---")
plt.figure(figsize=(20, 15))
# ê°� í”¼ì²˜ì—� ëŒ€í•œ í�ˆìŠ¤í† ê·¸ë�¨ì�„ ê·¸ë¦½ë‹ˆë‹¤.
# Draw a histogram for each feature.
for i, feature in enumerate(numerical_features):
    plt.subplot(4, 3, i + 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=12)
plt.tight_layout()
plt.show()
# ë¡œê·¸ ë³€í™˜ì�´ í•„ìš”í•œ ë¶€ë¶„ì�´ ì�ˆìŠµë‹ˆë‹¤.


# --- 3.2. ì „ì²´ í”¼ì²˜ ë°•ìŠ¤í”Œë¡¯ ---
print("\n--- [Visualization] Boxplots for all Numerical Features ---")
plt.figure(figsize=(20, 15))
# ê°� í”¼ì²˜ì—� ëŒ€í•œ ë°•ìŠ¤í”Œë¡¯ì�„ ê·¸ë¦½ë‹ˆë‹¤. ì�´ìƒ�ì¹˜(outlier)ë¥¼ í™•ì�¸í•˜ëŠ” ë�° ìœ ìš©í•©ë‹ˆë‹¤.
# Draw a boxplot for each feature, which is useful for identifying outliers.
for i, feature in enumerate(numerical_features):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=train_df[feature])
    plt.title(f'Boxplot of {feature}', fontsize=12)
plt.tight_layout()
plt.show()
# ì�´ ì�´ìƒ�ì¹˜ë“¤ì�€ ì˜¤ë¥˜ê°€ ì•„ë‹ˆë�¼, ì�Œì•… ì�¥ë¥´ì�˜ ë‹¤ì–‘ì„±ê³¼ ê°œì„±ì�„ ë°˜ì˜�í•˜ëŠ” ì��ì—°ìŠ¤ëŸ¬ìš´ ê²°ê³¼ì�…ë‹ˆë‹¤.


# --- 3.3. ê¸°ë³¸ íŠ¹ì„± ì¤‘ìš”ë�„ í™•ì�¸ ---
# ì°¸ê³  : íŠ¹ì„± ì¤‘ìš”ë�„ëŠ” ëª¨ë�¸ì�´ í•™ìŠµë�˜ì–´ì•¼ë§Œ ì•Œ ìˆ˜ ì�ˆìœ¼ë¯€ë¡œ, ì�´ ë‹¨ê³„ì—�ì„œëŠ” ë¹ ë¥´ê³  ê°„ë‹¨í•œ
# LightGBM ê¸°ë³¸ ëª¨ë�¸ì�„ 'ì�„ì‹œ'ë¡œ í•™ìŠµì‹œì¼œ ëŒ€ë�µì �ì�¸ ì¤‘ìš”ë�„ë¥¼ íŒŒì•…í•©ë‹ˆë‹¤.
# Note : Since feature importance can only be known after a model is trained, in this step,
# we will 'temporarily' train a quick and simple default LightGBM model to grasp the approximate importances.
print("\n--- [Analysis] Baseline Feature Importance ---")
X_eda = train_df.drop(columns=['id', 'BeatsPerMinute'])
y_eda = train_df['BeatsPerMinute']

# ì�„ì‹œ ëª¨ë�¸ í•™ìŠµ
# Train a temporary model.
temp_model = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
temp_model.fit(X_eda, y_eda)

# íŠ¹ì„± ì¤‘ìš”ë�„ë¥¼ ë�°ì�´í„°í”„ë ˆì�„ìœ¼ë¡œ ë§Œë“¤ì–´ ì‹œê°�í™”í•©ë‹ˆë‹¤.
# Create a dataframe of feature importances and visualize it.
feature_importance_df = pd.DataFrame({
    'feature': X_eda.columns,
    'importance': temp_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(5, 3))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Baseline Feature Importance (LGBM)')
plt.show()

print("\nBaseline Feature Importance Ranking:")
print(feature_importance_df)


# --- 3.4. íŠ¹ì„± ê°„ ìƒ�ê´€ê´€ê³„ ë¶„ì„� ---
print("--- Feature Correlation Heatmap ---")
corr_matrix = train_df.drop('id', axis=1).corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()


# ==============================================================================
# 4. ì „ì²˜ë¦¬ ë°� í”¼ì²˜ ì—”ì§€ë‹ˆì–´ë§� (Preprocessing & Feature Engineering)
# ==============================================================================
def create_catboost_features(X: DataFrame) -> DataFrame:
    """
    CatBoost ì „ìš© ë²”ì£¼í˜• íŒŒìƒ� ë³€ìˆ˜ë¥¼ ìƒ�ì„±í•©ë‹ˆë‹¤.
    Creates categorical features only for CatBoost model.
    """
    X_cb = X.copy()

    def categorize_rhythm(score):
        if score < 0.4:
            return 'Slow'
        elif score < 0.7:
            return 'Medium'
        else:
            return 'Fast'

    def categorize_duration(ms):
        if ms < 180000:
            return 'Short'
        elif ms < 270000:
            return 'Medium'
        else:
            return 'Long'

    def categorize_energy(e):
        if e < 0.3:
            return 'Low'
        elif e < 0.7:
            return 'Medium'
        else:
            return 'High'

    X_cb['RhythmCategory'] = X_cb['RhythmScore'].apply(categorize_rhythm).astype(str)
    X_cb['DurationCategory'] = X_cb['TrackDurationMs'].apply(categorize_duration).astype(str)
    X_cb['EnergyCategory'] = X_cb['Energy'].apply(categorize_energy).astype(str)

    return X_cb
    
def preprocess_and_feature_engineer(
    df_train: DataFrame, 
    df_test: DataFrame
) -> Tuple[DataFrame, Series, DataFrame, DataFrame, DataFrame]:
    """
    ì›�ë³¸ í›ˆë ¨ ë�°ì�´í„°ì™€ í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ë¥¼ ë°›ì•„, ìš°ë¦¬ê°€ ì •ì�˜í•œ ëª¨ë“  ì „ì²˜ë¦¬ ë°�
    í”¼ì²˜ ì—”ì§€ë‹ˆì–´ë§� ê³¼ì •ì�„ ì �ìš©í•œ ìµœì¢… ëª¨ë�¸ ì�…ë ¥ ë�°ì�´í„°ë¥¼ ìƒ�ì„±í•©ë‹ˆë‹¤.
    Performs all defined preprocessing and feature engineering steps on the raw
    training and test dataframes to generate the final model inputs.
    """
    start_time = time.time()

    print("--- Starting Preprocessing & Feature Engineering ---")
    # 1. ê¸°ë³¸ ë�°ì�´í„°ì…‹ ìƒ�ì„± (Create base datasets)
    X = train_df.drop(columns=['id', 'BeatsPerMinute'])
    y = train_df['BeatsPerMinute']
    X_test = test_df.drop(columns=['id'])

    # 2. ì�´ìƒ�ì¹˜ ì²˜ë¦¬ : AudioLoudness ìœˆì €ë�¼ì�´ì§•
    # 2. Outlier Handling : Winsorizing AudioLoudness
    # ê·¹ë‹¨ì �ì�¸ ê°’ì�„ ì œê±°í•˜ëŠ” ëŒ€ì‹ , ê²½ê³„ê°’ìœ¼ë¡œ ë³´ì •í•˜ì—¬ ì •ë³´ ì†�ì‹¤ì�„ ìµœì†Œí™”í•©ë‹ˆë‹¤.
    # Instead of removing extreme values, we cap them to minimize information loss.
    X['AudioLoudness'] = winsorize(X['AudioLoudness'], limits=[0.01, 0.01])
    X_test['AudioLoudness'] = winsorize(X_test['AudioLoudness'], limits=[0.01, 0.01])
    
    # 3. ì� ë¦¼ ì™„í™” : InstrumentalScore ë¡œê·¸ ë³€í™˜
    # 3. Skewness Mitigation: Log Transform for InstrumentalScore
    X['InstrumentalScore'] = np.log1p(X['InstrumentalScore'])
    X_test['InstrumentalScore'] = np.log1p(X_test['InstrumentalScore'])
    # EDAì—�ì„œ ë°œê²¬ë�œ ì� ë¦¼ í˜„ìƒ�ì�„ ì™„í™”í•˜ì—¬ ëª¨ë�¸ì�´ íŒ¨í„´ì�„ ë�” ì�˜ í•™ìŠµí•˜ë�„ë¡� ë�•ìŠµë‹ˆë‹¤.
    # Mitigate the skewness found in EDA to help the model learn patterns better.

    # 4. ìƒ�ê´€ê´€ê³„ ê¸°ë°˜ í”¼ì²˜ ìƒ�ì„±
    # 4. Create Correlation-Based Feature
    X['Non_Acoustic_Energy'] = X['Energy'] / (X['AcousticQuality'] + 1e-6)
    X_test['Non_Acoustic_Energy'] = X_test['Energy'] / (X_test['AcousticQuality'] + 1e-6)

    X['Loudness_per_Energy'] = X['AudioLoudness'] / (X['Energy'] + 1e-6)
    X_test['Loudness_per_Energy'] = X_test['AudioLoudness'] / (X_test['Energy'] + 1e-6)

    # 5. CatBoost ì „ìš© ë²”ì£¼í˜• íŒŒìƒ� ë³€ìˆ˜ ìƒ�ì„±
    # 5. Create categorical features exclusively for CatBoost
    X_cb = create_catboost_features(X)
    X_test_cb = create_catboost_features(X_test)
    
    end_time = time.time()
    process_time = end_time - start_time
    print(f"â�±ï¸� Process Time : {format_time(process_time)}")
    
    return X, y, X_test, X_cb, X_test_cb


X, y, X_test, X_cb, X_test_cb = preprocess_and_feature_engineer(train_df, test_df)

X.shape


print("--- Feature Correlation Heatmap ---")
corr_matrix = X.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()


# ì�´ í•¨ìˆ˜ëŠ” ë�°ì�´í„°í”„ë ˆì�„ì�„ ì�…ë ¥ë°›ì•„ ì•„ë¬´ê²ƒë�„ ë°˜í™˜(return)í•˜ì§€ ì•Šìœ¼ë¯€ë¡œ, ë°˜í™˜ íƒ€ì�…ì�€ Noneì�…ë‹ˆë‹¤.
# This function takes a DataFrame as input and returns nothing, so its return type is None.
def calculate_vif(X: pd.DataFrame) -> None:
    """
    ë�°ì�´í„°í”„ë ˆì�„ì�˜ ëª¨ë“  í”¼ì²˜ì—� ëŒ€í•œ VIF ì �ìˆ˜ë¥¼ ê³„ì‚°í•˜ê³  ì¶œë ¥í•©ë‹ˆë‹¤.
    Calculates and prints the VIF scores for all features in a dataframe.
    """
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("--- Variance Inflation Factor (VIF) ---")
    print(vif.sort_values('VIF', ascending=False))

# 4ë²ˆ ì „ì²˜ë¦¬ ë°� í”¼ì²˜ ì—”ì§€ë‹ˆì–´ë§� ë‹¨ê³„ê°€ ë��ë‚œ ìµœì¢… X ë�°ì�´í„°ì…‹ìœ¼ë¡œ VIFë¥¼ ê³„ì‚°í•©ë‹ˆë‹¤.
# Calculate VIF on the final X dataset after step 4 (Preprocessing & FE).
calculate_vif(X)


print(X_cb.columns)


cat_vars = ['RhythmCategory', 'DurationCategory', 'EnergyCategory']

df = X_cb.copy()
df['BPM'] = y

plt.figure(figsize=(7, 2))

for i, var in enumerate(cat_vars):
    plt.subplot(1, 3, i + 1)  # 3í–‰ 1ì—´ ì¤‘ ië²ˆì§¸ ìœ„ì¹˜
    sns.violinplot(x=var, y='BPM', data=df, inner='box', palette='pastel')
    plt.title(f"BPM by {var}")
    plt.tight_layout(pad=2.0)
    plt.xlabel(var)
    plt.ylabel("BPM")

plt.tight_layout()
plt.show()



# # ==============================================================================
# # 5. í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° ìµœì �í™” (Hyperparameter Optimization)
# # ==============================================================================
def tune_model(model_name: str, X: DataFrame, y: Series, X_cb: Optional[DataFrame] = None) -> Study:
    """
    ì§€ì •ë�œ ëª¨ë�¸ ì�´ë¦„ì—� ë§�ì¶° Optunaë¡œ í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° ìµœì �í™”ë¥¼ ìˆ˜í–‰í•©ë‹ˆë‹¤.
    Performs hyperparameter optimization with Optuna for the specified model name.
    """
    start_time = time.time()
    
    def objective(trial: Trial) -> float:
        # 1. model_nameì—� ë”°ë�¼ ë‹¤ë¥¸ íŒŒë�¼ë¯¸í„° ì¡°í•©ì�„ íƒ�ìƒ‰í•©ë‹ˆë‹¤.
        # 1. Search different parameter combinations based on the model_name.
        
        if model_name == 'lightgbm':
            params = {
                'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'random_state': SEED, 'n_jobs': -1, 'verbose': -1
            }
            # Optunaì�˜ trial ê°�ì²´ë¥¼ ì‚¬ìš©í•´, íƒ�ìƒ‰í•  í•˜ì�´í�¼íŒŒë�¼ë¯¸í„°ì�˜ ë²”ìœ„ì™€ ì¢…ë¥˜ë¥¼ ì •ì�˜í•©ë‹ˆë‹¤.
            # Defines the search space for hyperparameters using Optuna's trial object.             
        elif model_name == 'catboost':
            params = {
                'iterations': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': SEED, 'verbose': 0, 'loss_function': 'RMSE'
            }
            # CatBoost ëª¨ë�¸ì—� ìµœì �í™”í•  í•˜ì�´í�¼íŒŒë�¼ë¯¸í„°ì�˜ íƒ�ìƒ‰ ê³µê°„ì�„ ì •ì�˜í•©ë‹ˆë‹¤. iterationsëŠ” íŠ¸ë¦¬ì�˜ ê°œìˆ˜, depthëŠ” ê¹Šì�´ ë“±ì�„ ì�˜ë¯¸í•©ë‹ˆë‹¤.
            # Defines the hyperparameter search space to optimize for the CatBoost model, including number of trees (iterations), depth, etc.
        # 2. êµ�ì°¨ ê²€ì¦� ë¡œì§�ì�€ ë‹¨ í•œ ë²ˆë§Œ ì�‘ì„±ë�©ë‹ˆë‹¤.
        # 2. The cross-validation logic is written only once.
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        val_scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

            # 3. model_nameì—� ë§�ëŠ” ì˜¬ë°”ë¥¸ ëª¨ë�¸ì�„ ìƒ�ì„±í•˜ê³  í•™ìŠµì‹œí‚µë‹ˆë‹¤.
            # 3. Create and train the correct model based on model_name.
            if model_name == 'lightgbm':
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_val_fold, y_val_fold)],
                          callbacks=[lgb.early_stopping(10, verbose=False)])
                val_preds = model.predict(X_val_fold)
            elif model_name == 'catboost':
                # CatBoost ëª¨ë�¸ í•™ìŠµ ì‹œ
                # CatBoostìš© foldë³„ ë�°ì�´í„° ìƒ�ì„±
                X_train_fold_cb = create_catboost_features(X_train_fold)
                X_val_fold_cb = create_catboost_features(X_val_fold)
                
                # ë²”ì£¼í˜• ë³€ìˆ˜ ì��ë�™ ì¶”ì¶œ (í˜„ì�¬ fold ê¸°ì¤€)
                cat_features = [col for col in X_train_fold_cb.columns if 'Category' in col]
                model = CatBoostRegressor(**params)
                model.fit(
                    X_train_fold_cb, y_train_fold,
                    cat_features=cat_features,
                    eval_set=[(X_val_fold_cb, y_val_fold)],
                    early_stopping_rounds=10,
                    verbose=0
                    )
                val_preds = model.predict(X_val_fold_cb)
            rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
            val_scores.append(rmse)
            
        return np.mean(val_scores)

    # 4. Optuna ìŠ¤í„°ë””ë¥¼ ì‹¤í–‰í•˜ê³  ê²°ê³¼ë¥¼ ë°˜í™˜í•©ë‹ˆë‹¤.
    # 4. Run the Optuna study and return the results.
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    end_time = time.time()
    process_time = end_time - start_time
    print(f"â�±ï¸� Process Time : {format_time(process_time)}")
    return study


print("\n--- [1/2] Starting LightGBM Optimization with Optuna ---")
study_lgbm = tune_model('lightgbm', X, y)
lgbm_best_params = study_lgbm.best_params

print("\n--- [2/2] Starting CatBoost Optimization with Optuna ---")
study_catboost = tune_model('catboost', X, y, X_cb)
cb_best_params = study_catboost.best_params

# --- ìµœì¢… ê²°ê³¼ ì¶œë ¥ ---
print("\n--- Optuna Optimization Results ---")
print(f"ğŸš€ LightGBM Best RMSE : {study_lgbm.best_value:.4f}")
print(f"ğŸ�ˆ CatBoost Best RMSE : {study_catboost.best_value:.4f}")


# ==============================================================================
# 6. ìµœì¢… ëª¨ë�¸ ì‹¤í–‰ (Execute Final Model)
# ==============================================================================
def build_final_model_and_submit(
    X: DataFrame, 
    y: Series, 
    X_test: DataFrame, 
    study_lgbm: Study, 
    study_catboost: Study, 
    data_path: str
) -> Tuple[DataFrame, StackingRegressor]:
    """
    ìµœì �ì�˜ íŒŒë�¼ë¯¸í„°ë¡œ Stacking ì•™ìƒ�ë¸” ëª¨ë�¸ì�„ í•™ìŠµí•˜ê³ ,ìµœì¢… ì œì¶œ íŒŒì�¼ì�„ ìƒ�ì„±í•©ë‹ˆë‹¤.
    Trains a stacking ensemble with optimal parameters and generates the final submission file.
    """
    start_time = time.time()
    print("\n--- Starting Final Model Process: Stacking and Submission ---")
    
    # --- 1. ìµœì � íŒŒë�¼ë¯¸í„° ë°� ëª¨ë�¸ ì •ì�˜ ---
    # --- 1. Define Best Parameters and Models ---
    lgbm_best_params = study_lgbm.best_params
    lgbm_best_params.update({'random_state': SEED, 'n_jobs': -1, 'verbose': -1})
    cb_best_params = study_catboost.best_params
    cb_best_params.update({'random_seed': SEED, 'verbose': 0})

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('ridge', Ridge(random_state=SEED))
    ])
    base_models = [
        ('lgbm', lgb.LGBMRegressor(**lgbm_best_params)),
        ('catboost', CatBoostRegressor(**cb_best_params)),
        ('ridge', ridge_pipeline)
    ]
    # 1ë‹¨ê³„ ëª¨ë�¸ë“¤ì�˜ ì˜ˆì¸¡ì�€ ì�´ë¯¸ ë³µì�¡í•œ ë¹„ì„ í˜• ê´€ê³„ë¥¼ í•™ìŠµí•œ ê²°ê³¼ì�…ë‹ˆë‹¤.
    # ë”°ë�¼ì„œ ì�´ë“¤ì�„ ìµœì¢…ì �ìœ¼ë¡œ ì¡°í•©í•˜ëŠ” 2ë‹¨ê³„ ë©”íƒ€ ëª¨ë�¸ì�€, ì�´ ê´€ê³„ë¥¼ ë‹¨ìˆœí•˜ê²Œ ì¢…í•©í•´ì£¼ëŠ”
    # ì„ í˜• ëª¨ë�¸(LinearRegression)ì�´ ê³¼ì �í•©ì�„ ë°©ì§€í•˜ê³  ë�” ì•ˆì •ì �ì�¸ ê²½í–¥ì�´ ì�ˆìŠµë‹ˆë‹¤.
    # The predictions from the base models have already learned complex non-linear relationships.
    # Therefore, a simple linear model is often a good choice for the meta-model to avoid overfitting.
    
    meta_model = LinearRegression()
    
    # --- 2. Stacking ëª¨ë�¸ í•™ìŠµ ë°� ì˜ˆì¸¡ ---
    # --- 2. Train Stacking Model and Predict ---
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=N_SPLITS, n_jobs=-1)
    stacking_model.fit(X, y)
    stacking_predictions = stacking_model.predict(X_test)

    # --- 3. ìµœì¢… ì œì¶œ íŒŒì�¼ ìƒ�ì„± ---
    # --- 3. Create Final Submission File ---
    submission_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    submission_df['BeatsPerMinute'] = stacking_predictions
    submission_df.to_csv('submission.csv', index=False)
    
    end_time = time.time()
    process_time = end_time - start_time
    print(f"â�±ï¸� Full Final Process Time : {format_time(process_time)}")
    print("\nFinal submission file 'submission.csv' created successfully!")

    return submission_df, stacking_model

final_submission_df, stacking_model = build_final_model_and_submit(
    X, y, X_test, study_lgbm, study_catboost, data_path
)


# ==============================================================================
# 7. ëª¨ë�¸ ë¶„ì„� (Feature Analysis)
# ==============================================================================
print("\n--- Analyzing Final Model with Residual Plot ---")
# Residual Plot (ì�”ì°¨ í”Œë¡¯) : ì „ì²´ ìŠ¤íƒœí‚¹ ëª¨ë�¸ ê¸°ì¤€ìœ¼ë¡œ ê·¸ë ¤ì•¼ ìµœì¢… ì„±ëŠ¥ í�‰ê°€ ê°€ëŠ¥

# 1. ì˜ˆì¸¡ê°’ê³¼ ì�”ì°¨ ê³„ì‚°
y_pred = stacking_model.predict(X)
residuals = y - y_pred

# 2. ì‹œê°�í™”
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted BPM")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot of Stacking Model")
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. ì�”ì°¨ ë¶„í�¬ í�ˆìŠ¤í† ê·¸ë�¨
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, kde=True, color='darkorange')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. ì�”ì°¨ê°€ í�° ìƒ˜í”Œ ì¶”ì¶œ
threshold = 40
df_resid = df.copy()
df_resid['Residual'] = residuals
outliers = df_resid[np.abs(df_resid['Residual']) > threshold]

# 5. ì�´ìƒ� ìƒ˜í”Œ íŠ¹ì§• í™•ì�¸
print("\nğŸ”� ì�”ì°¨ê°€ í�° ìƒ˜í”Œ ê°œìˆ˜:", len(outliers))
print(outliers[['BPM', 'Residual', 'RhythmCategory', 'DurationCategory', 'EnergyCategory']].head())



print("\n--- Analyzing Final Model with SHAP ---")
# SHAP (ìƒ¤í”„) : ëª¨ë�¸ í•´ì„� ê¸°ëŠ¥

# 1. SHAP í•´ì„�
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_cb)

# 2. SHAP Summary Plot
shap.summary_plot(shap_values, X_cb)

# 3. SHAP Dependence Plot (ìƒ�ìœ„ ë³€ìˆ˜ í•˜ë‚˜ ì˜ˆì‹œ)
top_feature = X_cb.columns[np.abs(shap_values).mean(axis=0).argmax()]
shap.dependence_plot(top_feature, shap_values, X_cb)


# ==============================================================================
# 8. 2ë‹¨ê³„ ëª¨ë�¸ë§� : ì˜¤ì°¨ ê¸°ë°˜ ì�¬í•™ìŠµ (Stage 2 Modeling: Error-Based Retraining)
# ==============================================================================
print("\n--- Starting Stage 2 Modeling: Error-Based Retraining ---")
start_time_stage2 = time.time()

# --- 8.1. 1ë‹¨ê³„ ëª¨ë�¸ì�˜ OOF ì˜ˆì¸¡ ìƒ�ì„± (ì˜¤ì°¨ ë¶„ì„�ìš©) ---
# Stacking ëª¨ë�¸ì�˜ OOF ì˜ˆì¸¡ì�„ ì–»ëŠ” ê²ƒì�€ ë³µì�¡í•˜ë¯€ë¡œ, ê°€ì�¥ ì„±ëŠ¥ì�´ ì¢‹ì•˜ë�˜ CatBoost ë‹¨ì�¼ ëª¨ë�¸ì�˜
# OOF ì˜ˆì¸¡ì�„ ì‚¬ìš©í•˜ì—¬ 'ì–´ë ¤ìš´ ë�°ì�´í„°'ë¥¼ ì •ì�˜í•©ë‹ˆë‹¤.
# Since getting OOF predictions from a Stacking model is complex, we'll define 'difficult data'
# using the OOF predictions from our best single model, CatBoost.

kf_stage2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(X.shape[0])
analyzer_model_stage2 = CatBoostRegressor(**cb_best_params)

print("Generating OOF predictions to identify outliers...")
for fold, (train_idx, val_idx) in enumerate(kf_stage2.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    cat_features = ['cluster'] if 'cluster' in X.columns else [] # í�´ëŸ¬ìŠ¤í„° í”¼ì²˜ ìœ ë¬´ì—� ë”°ë�¼ ë�™ì �ìœ¼ë¡œ ì²˜ë¦¬
    
    analyzer_model_stage2.fit(X_train_fold, y_train_fold, cat_features=cat_features, verbose=0)
    oof_preds[val_idx] = analyzer_model_stage2.predict(X_val_fold)

# --- 8.2. ì˜¤ì°¨ë¥¼ ê¸°ì¤€ìœ¼ë¡œ 'ì‰¬ìš´ ë�°ì�´í„°'ë§Œ í•„í„°ë§� ---
oof_df = pd.DataFrame({'Residual': y - oof_preds})
X_with_resid = X.join(oof_df)

threshold = 40 # ì˜¤ì°¨ì�˜ ì�„ê³„ê°’. ì�´ ê°’ì�€ ì‹¤í—˜ì�„ í†µí•´ ì¡°ì ˆí•´ë³¼ ìˆ˜ ì�ˆìŠµë‹ˆë‹¤.
clean_indices = X_with_resid[np.abs(X_with_resid['Residual']) <= threshold].index

X_clean = X.loc[clean_indices]
y_clean = y.loc[clean_indices]

print(f"\nOriginal training data size : {X.shape[0]}")
print(f"Cleaned training data size (error <= {threshold}) : {X_clean.shape[0]}")
print(f"Outliers removed : {X.shape[0] - X_clean.shape[0]}")

# --- 8.3. 'í�´ë¦° ë�°ì�´í„°'ë¡œ 2ë‹¨ê³„ ëª¨ë�¸ í•™ìŠµ ë°� í�‰ê°€ ---
print("\n--- Training and evaluating a new model on the cleaned data ---")
model_clean = CatBoostRegressor(**cb_best_params)

# í�´ë¦° ë�°ì�´í„° ë‚´ì—�ì„œ ë‹¤ì‹œ êµ�ì°¨ ê²€ì¦�ì�„ ìˆ˜í–‰í•˜ì—¬ 2ë‹¨ê³„ ëª¨ë�¸ì�˜ ì„±ëŠ¥ì�„ ì¸¡ì •í•©ë‹ˆë‹¤.
# Perform cross-validation again within the clean data to measure the Stage 2 model's performance.
kf_clean = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
val_scores_clean = []
for fold, (train_idx, val_idx) in enumerate(kf_clean.split(X_clean, y_clean)):
    X_train_fold, y_train_fold = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
    X_val_fold, y_val_fold = X_clean.iloc[val_idx], y_clean.iloc[val_idx]
    
    cat_features = ['cluster'] if 'cluster' in X_clean.columns else []

    model_clean.fit(X_train_fold, y_train_fold, 
                    eval_set=[(X_val_fold, y_val_fold)],
                    cat_features=cat_features,
                    early_stopping_rounds=10, 
                    verbose=0)
    
    val_preds_clean = model_clean.predict(X_val_fold)
    rmse_clean = np.sqrt(mean_squared_error(y_val_fold, val_preds_clean))
    val_scores_clean.append(rmse_clean)

avg_rmse_clean = np.mean(val_scores_clean)
end_time_stage2 = time.time()

print(f"\n--- Stage 2 Modeling Complete ---")
print(f"ğŸ“Š Average RMSE on Cleaned Data : {avg_rmse_clean:.4f}")
print(f"â�±ï¸� Stage 2 Process Time : {format_time(end_time_stage2 - start_time_stage2)}")


# ==============================================================================
# 9. 2ë‹¨ê³„ ëª¨ë�¸ ì‹¬ì¸µ ë¶„ì„� (In-depth Analysis of Stage 2 Model)
# ==============================================================================
# ì�´ ì½”ë“œëŠ” 8ë²ˆ '2ë‹¨ê³„ ëª¨ë�¸ë§�'ì�´ ì‹¤í–‰ë�˜ì–´ X_clean, y_clean, avg_rmse_clean ë“±ì�˜ ë³€ìˆ˜ê°€
# ìƒ�ì„±ë�˜ì—ˆë‹¤ê³  ê°€ì •í•˜ê³  ì§„í–‰í•©ë‹ˆë‹¤.
# This code assumes that Step 8 'Stage 2 Modeling' has been executed and variables like
# X_clean, y_clean, and avg_rmse_clean have been created.

# --- 1. ê¸°ì¡´ ëª¨ë�¸ê³¼ RMSE ì �ìˆ˜ ë¹„êµ� ---
# --- 1. Compare RMSE score with the previous model ---
previous_best_rmse = study_catboost.best_value # 5ë²ˆ ë‹¨ê³„ì—�ì„œ ì°¾ì�€ CatBoost ìµœê³  ì �ìˆ˜
print("--- RMSE Score Comparison ---")
print(f"Previous Best CV Score (on all data) : {previous_best_rmse:.4f}")
print(f"New Model CV Score (on cleaned data) : {avg_rmse_clean:.4f}")
# ì°¸ê³ : ë‘� ì �ìˆ˜ëŠ” ì„œë¡œ ë‹¤ë¥¸ ë�°ì�´í„°ì…‹ìœ¼ë¡œ ì¸¡ì •ë�˜ì—ˆìœ¼ë¯€ë¡œ ì§�ì ‘ì �ì�¸ ì„±ëŠ¥ ë¹„êµ�ëŠ” ì–´ë µìŠµë‹ˆë‹¤.
# Note: A direct performance comparison is difficult as the two scores were measured on different datasets.

# --- 2. clean_indices ì €ì�¥ ---
# --- 2. Save clean_indices ---
# ë‚˜ì¤‘ì—� ì–´ë–¤ ë�°ì�´í„°ê°€ 'ì‰¬ìš´ ë¬¸ì œ'ì˜€ëŠ”ì§€, 'ì–´ë ¤ìš´ ë¬¸ì œ'ì˜€ëŠ”ì§€ ë¶„ì„�í•˜ê¸° ìœ„í•´ ì�¸ë�±ìŠ¤ë¥¼ ì €ì�¥í•©ë‹ˆë‹¤.
# We save the indices for later analysis to understand which data points were 'easy' or 'hard'.
outlier_indices = train_df.index.difference(clean_indices)

# ì�¸ë�±ìŠ¤ë¥¼ íŒŒì�¼ë¡œ ì €ì�¥
# Save indices to files
np.save('clean_indices.npy', clean_indices)
np.save('outlier_indices.npy', outlier_indices)
print("\n'clean_indices.npy' and 'outlier_indices.npy' saved successfully!")

# --- 3. model_clean ê¸°ì¤€ SHAP í•´ì„� ---
# --- 3. SHAP Analysis based on model_clean ---
print("\n--- Analyzing the new model trained on cleaned data ---")
# 'í�´ë¦° ë�°ì�´í„°' ì „ì²´ë¡œ ìµœì¢… ë¶„ì„�ìš© ëª¨ë�¸ì�„ ë‹¤ì‹œ í•™ìŠµí•©ë‹ˆë‹¤.
# Retrain the final analysis model on the entire 'clean data'.
model_clean_final = CatBoostRegressor(**cb_best_params)
cat_features = ['cluster'] if 'cluster' in X_clean.columns else []
model_clean_final.fit(X_clean, y_clean, cat_features=cat_features, verbose=0)

# SHAP ë¶„ì„�
# For performance, we analyze a sample of the cleaned data
X_sample_clean = X_clean.sample(min(1000, len(X_clean)), random_state=SEED)
explainer_clean = shap.TreeExplainer(model_clean_final)
shap_values_clean = explainer_clean.shap_values(X_sample_clean)

print("Displaying SHAP Summary Plot for the model on Cleaned Data...")
shap.summary_plot(shap_values_clean, X_sample_clean)
# 'ì‰¬ìš´ ë¬¸ì œ'ë§Œ í•™ìŠµí•œ ëª¨ë�¸ì�€ ì–´ë–¤ í”¼ì²˜ë¥¼ ì¤‘ìš”í•˜ê²Œ ìƒ�ê°�í•˜ëŠ”ì§€, ê·¸ ê²½í–¥ì�´ ì�´ì „ê³¼ ë‹¤ë¥¸ì§€ í™•ì�¸í•©ë‹ˆë‹¤.
# Check which features the model trained only on 'easy problems' considers important, and if the trend differs from before.

