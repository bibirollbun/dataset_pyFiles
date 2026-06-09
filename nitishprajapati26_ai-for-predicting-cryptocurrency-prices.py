# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
test_path  = '/kaggle/input/drw-crypto-market-prediction/test.parquet'

df_train = pd.read_parquet(train_path)
df_train = df_train.tail(250000)

y_train_DRW = df_train[['label']]
x_train_DRW = df_train.drop(['label'], axis=1)

df_test = pd.read_parquet(test_path)
df_test = df_test.tail(250000)

y_test_DRW = df_test[['label']]
x_test_DRW = df_test.drop(['label'], axis=1)

x_train_DRW.shape, y_train_DRW.shape, x_test_DRW.shape, y_test_DRW.shape





y_train_DRW.describe()


df_train.head()


x_train_DRW.head()


 """duplicate_cols = df.T.duplicated()

    if duplicate_cols.any():
        print("Duplicate columns found.")
        cols_to_drop = df.columns[duplicate_cols]
        print("Duplicate columns:", list(cols_to_drop))
        
        df.drop(columns=cols_to_drop, inplace=True)
        print("Duplicate columns removed.")
    else:
        print("No duplicate columns found.")"""


dataframes = {
    'x_train_DRW': x_train_DRW,
    'x_test_DRW': x_test_DRW
}

for name, df in dataframes.items():
    print("\n==========================================")
    print(f"Properties for DataFrame: '{name}'")
    print("==========================================")

    print("Shape:", df.shape)

    print("\nInfo:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nTotal missing values:", df.isna().sum().sum())

    # Duplicate row check
    if df.duplicated().any():
        print("Duplicate rows found. Removing duplicates.")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicate rows found.")

    # Duplicate column check
 

    # Infinite values check
    has_inf = df.isin([np.inf, -np.inf]).any().any()

    if has_inf:
        print("Infinite values detected.")
        cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()]
        print("Columns with infinite values:", list(cols_with_inf))

        for col in cols_with_inf:
            if df[col].value_counts().values[0] == len(df):
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped column: {col}")
    else:
        print("No infinite values found.")



duplicate_columns = [
    'X104', 'X110', 'X116', 'X122', 'X128', 'X134', 'X140',
    'X146', 'X152', 'X158', 'X164', 'X170', 'X176', 'X182',
    'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387',
    'X393', 'X399', 'X405', 'X411', 'X417', 'X423', 'X429'
]

for base_col in duplicate_columns:
    duplicates = []
    
    for col in df_train.columns:
        if col != base_col:
            if df_train[base_col].equals(df_train[col]):
                duplicates.append(col)
    
    print(f"{base_col} is duplicate with: {duplicates}")



import matplotlib.pyplot as plt
import statsmodels.api as sm

left_skewed_cols = []
right_skewed_cols = []

for name, df in dataframes.items():

    print("\nChecking skewness for:", name)

    for col in df.columns:
        if df[col].mean() < df[col].median():
            left_skewed_cols.append(col)
        elif df[col].mean() > df[col].median():
            right_skewed_cols.append(col)

    # Simple histogram for one column
    plt.hist(df['X247'], bins=100)
    plt.title(f"Histogram of X247 ({name})")
    plt.xlabel("X247")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q plot
    sm.qqplot(df['X247'], line='s')
    plt.title(f"Q-Q Plot of X247 ({name})")
    plt.show()

print("\nLeft skewed columns:")
print(left_skewed_cols)

print("\nRight skewed columns:")
print(right_skewed_cols)



from sklearn.model_selection import train_test_split
x_train_original, x_test_original, y_train_original, y_test_original = train_test_split(
    x_train_DRW,
    y_train_DRW,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

print("Train-test split created")
x_train_original.shape, x_test_original.shape, y_train_original.shape, y_test_original.shape



import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import kurtosis

# ---------------- PARAMETERS ----------------
LOW_Q = 0.05
HIGH_Q = 0.95
SKEW_TH = 1.0
KURT_TH = 1.0
STD_TH = 1e-9

scaler = StandardScaler()
pt = PowerTransformer(method="yeo-johnson", standardize=False)

# --------------------------------------------------
# TRAIN = fit, TEST = transform only
# --------------------------------------------------

# ----------- 1. IQR / PERCENTILE CLIPPING (fit on train) ----------
for df in [x_train_original, x_test_original]:
    for col in x_train_original.columns:
        lower = x_train_original[col].quantile(LOW_Q)
        upper = x_train_original[col].quantile(HIGH_Q)
        df[col] = df[col].clip(lower, upper)

# ----------- 2. FIND SKEWED COLUMNS (train only) ----------
skewed_cols = []

for col in x_train_original.columns:
    skew_val = x_train_original[col].skew()
    kurt_val = kurtosis(x_train_original[col], fisher=True)

    if abs(skew_val) > SKEW_TH or abs(kurt_val) > KURT_TH:
        skewed_cols.append(col)

# Remove zero / near-zero variance columns
final_skewed_cols = [
    col for col in skewed_cols
    if x_train_original[col].nunique() > 1
    and x_train_original[col].std() > 1e-6
]

print("Power transform columns:", final_skewed_cols)

# ----------- 3. POWER TRANSFORMATION ----------
if final_skewed_cols:
    pt.fit(x_train_original[final_skewed_cols])

    x_train_original[final_skewed_cols] = pt.transform(
        x_train_original[final_skewed_cols]
    )
    x_test_original[final_skewed_cols] = pt.transform(
        x_test_original[final_skewed_cols]
    )

# ----------- 4. HANDLE INF / NAN ----------
for df in [x_train_original, x_test_original]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)   # safe fallback

# ----------- 5. STANDARD SCALING ----------
final_scale_cols = []

for col in x_train_original.columns:
    if x_train_original[col].std() > STD_TH:
        final_scale_cols.append(col)

scaler.fit(x_train_original[final_scale_cols])

x_train_original = pd.DataFrame(
    scaler.transform(x_train_original[final_scale_cols]),
    columns=final_scale_cols,
    index=x_train_original.index
)

x_test_original = pd.DataFrame(
    scaler.transform(x_test_original[final_scale_cols]),
    columns=final_scale_cols,
    index=x_test_original.index
)

# ----------- 6. MEMORY OPTIMIZATION ----------
for df in [x_train_original, x_test_original]:
    for col in df.columns:
        df[col] = df[col].astype("float32")

    df.replace(
        [np.inf, -np.inf],
        [np.finfo(np.float32).max, np.finfo(np.float32).min],
        inplace=True
    )

# REMOVE DUPLICATE COLUMNS (CORRECT WAY)
x_train_original = x_train_original.loc[:, ~x_train_original.columns.duplicated()]
x_test_original  = x_test_original.loc[:, ~x_test_original.columns.duplicated()]

gc.collect()

# ----------- FINAL OUTPUT ----------
x_train_original_processed = x_train_original
x_test_original_processed  = x_test_original

print("âœ… Outliers handled, data transformed, scaled, and memory optimized")



train_stats = x_train_original_processed.describe().T[['mean', 'std']]
test_stats = x_test_original_processed.describe().T[['mean', 'std']]
drift_df = train_stats.join(
    test_stats,
    lsuffix='_train',
    rsuffix='_test'
)

drift_df['mean_diff'] = (
    drift_df['mean_test'] - drift_df['mean_train']
).abs()

drift_df['std_ratio'] = (
    drift_df['std_test'] / drift_df['std_train']
)

drift_df = drift_df.sort_values(by='mean_diff', ascending=False)

drift_df



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
NUM_FEATURES = 10
SAMPLE_ROWS = 50
LABEL_COL = 'label'

# ---------------- MERGE X & y ----------------
features_to_plot = x_train_original_processed.iloc[:, :NUM_FEATURES]

merged_df = pd.concat(
    [features_to_plot, y_train_original[[LABEL_COL]]],
    axis=1
)

print("Merged dataframe shape:", merged_df.shape)

# ---------------- PAIRPLOT (RELATIONSHIP CHECK) ----------------
sns.pairplot(merged_df.iloc[:SAMPLE_ROWS, :])
plt.show()

# ---------------- PEARSON CORRELATION ----------------
pearson_corr = merged_df.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

# ---------------- SPEARMAN CORRELATION ----------------
spearman_corr = merged_df.corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Heatmap")
plt.show()



import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# ---------------- SETTINGS ----------------
CORR_THRESHOLD = 0.80
ROWS_TO_INCLUDE = 50000
LABEL_COL = 'label'
CORR_METHOD = 'pearson'

# ---------------- STEP 1: CORRELATION MATRIX (X only) ----------------
corr_matrix = x_train_original_processed.tail(ROWS_TO_INCLUDE).corr(method=CORR_METHOD)

# ---------------- STEP 2: MUTUAL INFORMATION (X vs y) ----------------
mi_scores = mutual_info_regression(
    x_train_original_processed.tail(ROWS_TO_INCLUDE),
    y_train_original[LABEL_COL].tail(ROWS_TO_INCLUDE)
)

mi_df = pd.DataFrame({
    'Feature': x_train_original_processed.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

mi_dict = mi_df.set_index('Feature')['MI_Score'].to_dict()

# ---------------- STEP 3: FIND HIGHLY CORRELATED PAIRS ----------------
high_corr_pairs = []

cols = corr_matrix.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > CORR_THRESHOLD:
            high_corr_pairs.append((cols[i], cols[j], corr_val))

print(f"Highly correlated pairs (|corr| > {CORR_THRESHOLD}):")
for c1, c2, val in high_corr_pairs:
    print(f"{c1} & {c2} : {val:.3f}")

# ---------------- STEP 4: DECIDE WHICH FEATURES TO DROP (USING MI) ----------------
features_to_drop = set()

for c1, c2, _ in high_corr_pairs:
    mi_1 = mi_dict.get(c1, -np.inf)
    mi_2 = mi_dict.get(c2, -np.inf)

    # Drop the less informative feature
    if mi_1 >= mi_2:
        features_to_drop.add(c2)
    else:
        features_to_drop.add(c1)

print("\nNumber of features dropped:", len(features_to_drop))

# ---------------- STEP 5: DROP FROM TRAIN ----------------
x_train_reduced_post_pearson = x_train_original_processed.drop(
    columns=list(features_to_drop),
    errors='ignore'
)

print("Train shape after multicollinearity removal:",
      x_train_reduced_post_pearson.shape)

# ---------------- STEP 6: APPLY SAME COLUMNS TO TEST ----------------
x_test_reduced_post_pearson = x_test_original_processed[
    x_train_reduced_post_pearson.columns
]

print("Test shape after multicollinearity removal:",
      x_test_reduced_post_pearson.shape)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# ---------------- SETTINGS ----------------
ROWS_TO_INCLUDE = 50000
LABEL_COL = 'label'
MI_THRESHOLD = 0.05

# ---------------- STEP 1: COMPUTE MUTUAL INFORMATION ----------------
mi_scores = mutual_info_regression(
    x_train_reduced_post_pearson.tail(ROWS_TO_INCLUDE),
    y_train_original[LABEL_COL].tail(ROWS_TO_INCLUDE)
)

mi_df = pd.DataFrame({
    'Feature': x_train_reduced_post_pearson.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

print("Total features:", mi_df.shape)

# ---------------- STEP 2: VISUALIZE MI SCORES ----------------
plt.figure(figsize=(14, 4))
sns.barplot(
    data=mi_df.head(50),
    x='Feature',
    y='MI_Score'
)
plt.xticks(rotation=90)
plt.title("Top 50 Features by Mutual Information")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.lineplot(x=range(len(mi_df)), y=mi_df['MI_Score'])
plt.xlabel("Feature Rank")
plt.ylabel("MI Score")
plt.title("Mutual Information Scores (Descending)")
plt.grid(linestyle=':')
plt.show()

# ---------------- STEP 3: SELECT FEATURES ABOVE THRESHOLD ----------------
selected_features = mi_df[mi_df['MI_Score'] >= MI_THRESHOLD]['Feature'].tolist()
dropped_features  = mi_df[mi_df['MI_Score'] < MI_THRESHOLD]['Feature'].tolist()

print("Features kept after MI filtering:", len(selected_features))
print("Features dropped after MI filtering:", len(dropped_features))

# ---------------- STEP 4: APPLY FEATURE SELECTION ----------------
x_train_reduced_post_pearson_MI = x_train_reduced_post_pearson[selected_features]
x_test_reduced_post_pearson_MI  = x_test_reduced_post_pearson[selected_features]

print("Train shape after MI filtering:", x_train_reduced_post_pearson_MI.shape)
print("Test shape after MI filtering:", x_test_reduced_post_pearson_MI.shape)




 x_train_reduced_post_pearson_MI
x_test_reduced_post_pearson_MI
y_train_original
y_test_original


#-------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt                  # To perform data visualisation
import seaborn as sns                                 # To perform data visualisation
import plotly.express as px                           # To perform data visualisation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
%matplotlib inline
                                             
from sklearn.linear_model import LinearRegression     # To perform prediction
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import math

from scipy.stats import chi2_contingency              # To perform chi sqr test
from scipy import stats
from scipy.stats import skewtest, norm
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler      # To perform feature scaling
from sklearn.model_selection import train_test_split  # To perform train test split

from sklearn.model_selection import GridSearchCV      # To perform hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import pearsonr                       # To perform pearson correlation
from scipy.stats import loguniform
from scipy.stats import uniform
from random import randint
import copy

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import gc
#import mlflow

sns.set_theme(style="white", palette="muted")
%matplotlib inline

#-------------------------------------------------------------------------------------------
import warnings                                       # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Ensure target is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# -----------------------------
# TRAIN LINEAR REGRESSION
# -----------------------------
lr_model = LinearRegression()

lr_model.fit(x_train_reduced_post_pearson_MI, y_train)

# -----------------------------
# PREDICT
# -----------------------------
lr_pred = lr_model.predict(x_test_reduced_post_pearson_MI)

# -----------------------------
# METRICS
# -----------------------------

# Pearson Correlation
corr = pd.Series(lr_pred).corr(y_test.reset_index(drop=True))
print(f"Pearson Correlation: {corr:.6f}")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(f"RMSE: {rmse:.6f}")

# R2 Score
r2 = r2_score(y_test, lr_pred)
print(f"R2 Score: {r2:.6f}")


from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Make sure y is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# Define alpha grid
custom_alphas = np.logspace(-3, 0, 100)

# -----------------------------
# TRAIN LASSO DIRECTLY
# -----------------------------
lasso_model = LassoCV(
    cv=5,
    alphas=custom_alphas,
    max_iter=10000,
    random_state=42
)

lasso_model.fit(x_train_reduced_post_pearson_MI, y_train)

print(f"Optimal alpha: {lasso_model.alpha_}")

# -----------------------------
# PREDICT
# -----------------------------
lasso_pred = lasso_model.predict(x_test_reduced_post_pearson_MI)

# -----------------------------
# METRICS
# -----------------------------

# Pearson Correlation
corr = pd.Series(lasso_pred).corr(y_test.reset_index(drop=True))
print(f"Pearson Correlation: {corr:.6f}")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
print(f"RMSE: {rmse:.6f}")

# R2
r2 = r2_score(y_test, lasso_pred)
print(f"R2 Score: {r2:.6f}")


from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Ensure target is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# Alpha grid
alphas = np.logspace(-3, 0, 100)

# -----------------------------
# TRAIN RIDGE
# -----------------------------
ridge_model = RidgeCV(alphas=alphas, cv=5)

ridge_model.fit(x_train_reduced_post_pearson_MI, y_train)

print(f"Optimal alpha: {ridge_model.alpha_}")

# -----------------------------
# PREDICT
# -----------------------------
ridge_pred = ridge_model.predict(x_test_reduced_post_pearson_MI)

# -----------------------------
# METRICS
# -----------------------------

# Pearson Correlation
corr = pd.Series(ridge_pred).corr(y_test.reset_index(drop=True))
print(f"Pearson Correlation: {corr:.6f}")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
print(f"RMSE: {rmse:.6f}")

# R2 Score
r2 = r2_score(y_test, ridge_pred)
print(f"R2 Score: {r2:.6f}")


from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Ensure target is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# Define alpha grid
alphas = np.logspace(-3, 0, 100)

# -----------------------------
# TRAIN ELASTIC NET
# -----------------------------
elastic_model = ElasticNetCV(
    alphas=alphas,
    cv=5,
    l1_ratio=0.5,      # 0 = Ridge, 1 = Lasso, 0.5 = balanced
    max_iter=10000,
    random_state=42
)

elastic_model.fit(x_train_reduced_post_pearson_MI, y_train)

print(f"Optimal alpha: {elastic_model.alpha_}")

# -----------------------------
# PREDICT
# -----------------------------
elastic_pred = elastic_model.predict(x_test_reduced_post_pearson_MI)

# -----------------------------
# METRICS
# -----------------------------

# Pearson Correlation
corr = pd.Series(elastic_pred).corr(y_test.reset_index(drop=True))
print(f"Pearson Correlation: {corr:.6f}")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))
print(f"RMSE: {rmse:.6f}")

# R2
r2 = r2_score(y_test, elastic_pred)
print(f"R2 Score: {r2:.6f}")



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Ensure target is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,      # number of trees
    max_depth=None,        # allow full growth
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(x_train_reduced_post_pearson_MI, y_train)

# -----------------------------
# PREDICT
# -----------------------------
rf_pred = rf_model.predict(x_test_reduced_post_pearson_MI)

# -----------------------------
# METRICS
# -----------------------------

# Pearson Correlation
corr = pd.Series(rf_pred).corr(y_test.reset_index(drop=True))
print(f"Pearson Correlation: {corr:.6f}")

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"RMSE: {rmse:.6f}")

# R2 Score
r2 = r2_score(y_test, rf_pred)
print(f"R2 Score: {r2:.6f}")


Pearson Correlation for Random Forest from RF's best features - original: -0.009571749228437102
Root Mean Squared Error for Random Forest from RF's best features - original: 1.3119
R2 score for RF from RF's best features - original: -0.5359


Features Importances List - RF Regressor: (31, 2)
Above Threshold Features Importances  List - RF Regressor: (31, 2)
Pearson Correlation for Random Forest from RF's best features - original: 0.028684649250007015
Root Mean Squared Error for Random Forest from RF's best features - original: 1.4222
R2 score for RF from RF's best features - original: -3.9047


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# INPUT DATA (ALREADY PROCESSED)
# -----------------------------
X_train = x_train_reduced_post_pearson_MI
y_train = y_train_original['label']   # âœ… FIXED HERE

X_test  = x_test_reduced_post_pearson_MI

ROWS_TO_INCLUDE = 50000
THRESHOLD = 0.015

# -----------------------------
# RANDOM FOREST HYPERPARAMETERS
# -----------------------------
rf_params = {
    'n_estimators': 150,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 0.7,
    'n_jobs': 2,
    'random_state': 42
}

# -----------------------------
# TRAIN RANDOM FOREST REGRESSOR
# -----------------------------
rf_model = RandomForestRegressor(**rf_params)

rf_model.fit(
    X_train.tail(ROWS_TO_INCLUDE),
    y_train.tail(ROWS_TO_INCLUDE)
)

# -----------------------------
# FEATURE IMPORTANCES
# -----------------------------
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Total features:", importance_df.shape)

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    data=importance_df.head(50),
    x='Feature',
    y='Importance',
    ax=ax[0]
)
ax[0].set_title('Top 50 Features - Random Forest')
ax[0].tick_params(axis='x', rotation=90)

sns.lineplot(
    x=range(len(importance_df)),
    y=importance_df['Importance'],
    ax=ax[1]
)
ax[1].set_title('RF Feature Importance (Ranked)')
ax[1].set_xlabel('Feature Rank')

plt.tight_layout()
plt.show()

# -----------------------------
# SELECT FEATURES ABOVE THRESHOLD
# -----------------------------
above_threshold_features_RF = importance_df[
    importance_df['Importance'] >= THRESHOLD
]

print("Features above threshold:", above_threshold_features_RF.shape)

# -----------------------------
# REDUCE TRAIN & TEST DATA
# -----------------------------
X_train_reduced_RF = X_train[above_threshold_features_RF['Feature']]
X_test_reduced_RF  = X_test[X_train_reduced_RF.columns]

print("Reduced Train Shape:", X_train_reduced_RF.shape)
print("Reduced Test Shape :", X_test_reduced_RF.shape)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Ensure target is 1D
y_train = y_train_original['label']
y_test  = y_test_original['label']

# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
rf_model_two = RandomForestRegressor(
    tuned_hyperparameters = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2
},
    random_state=42,
    n_jobs=-1
)

rf_model_two.fit(x_train_reduced_post_pearson_MI_RF, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
RF_pred = rf_model_two.predict(x_test_reduced_post_pearson_MI_RF)

RF_pred_df = pd.DataFrame(
    RF_pred,
    columns=['pred_label from RF top features - Original']
)

m1_RF = RF_pred_df['pred_label from RF top features - Original'].round(3)
m2_RF = y_test.reset_index(drop=True).round(3)

# -----------------------------
# METRICS (UNCHANGED ORDER)
# -----------------------------

# Pearson Correlation
corr_pandas_RF = m1_RF.corr(m2_RF)
print(f"Pearson Correlation for Random Forest from RF's best features - original: {corr_pandas_RF}")

# RMSE (same order as your code)
rmse_RF = np.sqrt(mean_squared_error(m1_RF, m2_RF))
print(f"Root Mean Squared Error for Random Forest from RF's best features - original: {rmse_RF:.4f}")

# R2 (same order as your code)
r2_RF = r2_score(m1_RF, m2_RF)
print(f"R2 score for RF from RF's best features - original: {r2_RF:.4f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# -----------------------------
# INPUT DATA (ALREADY PROCESSED)
# -----------------------------
X_train = x_train_reduced_post_pearson_MI
y_train = y_train_original['label']

X_test  = x_test_reduced_post_pearson_MI

THRESHOLD = 0.015

# -----------------------------
# XGBOOST HYPERPARAMETERS
# -----------------------------
xg_model = xgb.XGBRegressor(
    random_state=42,
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    n_jobs=-1
)

# -----------------------------
# TRAIN XGBOOST REGRESSOR
# -----------------------------
xg_model.fit(X_train, y_train)

# -----------------------------
# FEATURE IMPORTANCES
# -----------------------------
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xg_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Total features:", importance_df.shape)

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    data=importance_df.head(50),
    x='Feature',
    y='Importance',
    ax=ax[0]
)
ax[0].set_title('Top 50 Features - XGBoost')
ax[0].tick_params(axis='x', rotation=90)

sns.lineplot(
    x=range(len(importance_df)),
    y=importance_df['Importance'],
    ax=ax[1]
)
ax[1].set_title('XGBoost Feature Importance (Ranked)')
ax[1].set_xlabel('Feature Rank')

plt.tight_layout()
plt.show()

# -----------------------------
# SELECT FEATURES ABOVE THRESHOLD
# -----------------------------
above_threshold_features_XG = importance_df[
    importance_df['Importance'] >= THRESHOLD
]

print("Features above threshold:", above_threshold_features_XG.shape)

# -----------------------------
# REDUCE TRAIN & TEST DATA
# -----------------------------
X_train_reduced_XG = X_train[above_threshold_features_XG['Feature']]
X_test_reduced_XG  = X_test[X_train_reduced_XG.columns]

print("Reduced Train Shape:", X_train_reduced_XG.shape)
print("Reduced Test Shape :", X_test_reduced_XG.shape)



import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# INPUT DATA (CORRECT VARIABLES)
# -----------------------------
X_train = X_train_reduced_XG        # XG-selected features
y_train = y_train_original['label']

X_test  = X_test_reduced_XG
y_test  = y_test_original['label']

# -----------------------------
# TRAIN XGBOOST REGRESSOR
# -----------------------------
model_XG = xgb.XGBRegressor(
    random_state=42,
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    n_jobs=-1
)

model_XG.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
XG_pred = model_XG.predict(X_test)

XG_pred_df = pd.DataFrame(
    XG_pred,
    columns=['pred_label from XG top features - Original']
)

m1_XG = XG_pred_df['pred_label from XG top features - Original'].round(3)
m2_XG = y_test.reset_index(drop=True).round(3)

# -----------------------------
# METRICS
# -----------------------------
corr_pandas_XG = m1_XG.corr(m2_XG)
rmse_XG = np.sqrt(mean_squared_error(m2_XG, m1_XG))
r2_XG = r2_score(m2_XG, m1_XG)

print(f"Pearson Correlation for XG Regressor from XG's best features - original: {corr_pandas_XG}")
print(f"Root Mean Squared Error for XG Regressor from XG's best features - original: {rmse_XG:.4f}")
print(f"R2 score for XG from XG's best features - original: {r2_XG:.4f}")



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# SETTINGS
# -----------------------------
TIMESTAMPS_NUM = 10000
THRESHOLD = 0          # keep all, or change later
N_REPEATS = 5
RANDOM_STATE = 42

# -----------------------------
# PREPARE DATA (TIME-AWARE)
# -----------------------------
X_train = x_train_reduced_post_pearson_MI.tail(TIMESTAMPS_NUM)
y_train = y_train_original['label'].tail(TIMESTAMPS_NUM)

X_test  = x_test_reduced_post_pearson_MI.tail(TIMESTAMPS_NUM)
y_test  = y_test_original['label'].tail(TIMESTAMPS_NUM)

# -----------------------------
# TRAIN SVR MODEL
# -----------------------------
svr_model = SVR(
    kernel='rbf',
    C=1,
    epsilon=0.1,
    cache_size=500
)

svr_model.fit(X_train, y_train)

# -----------------------------
# PREDICTION & METRICS
# -----------------------------
y_pred = pd.Series(svr_model.predict(X_test), name="Prediction from SVR").round(3)
y_true = y_test.reset_index(drop=True).round(3)

pearson_svr = y_pred.corr(y_true)
rmse_svr = np.sqrt(mean_squared_error(y_true, y_pred))
r2_svr = r2_score(y_true, y_pred)

print("\n--- SVR Model Results (RBF Kernel) ---")
print(f"Pearson Correlation: {pearson_svr:.4f}")
print(f"RMSE: {rmse_svr:.4f}")
print(f"R2 Score: {r2_svr:.4f}")

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
perm_result = permutation_importance(
    svr_model,
    X_test,
    y_test,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": perm_result.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\nTotal features:", importance_df.shape[0])

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    data=importance_df.head(50),
    x="Feature",
    y="Importance",
    ax=ax[0]
)
ax[0].set_title("Top 50 Features - SVR (Permutation Importance)")
ax[0].tick_params(axis="x", rotation=90)

sns.lineplot(
    x=range(len(importance_df)),
    y=importance_df["Importance"],
    ax=ax[1]
)
ax[1].set_title("SVR Feature Importance (Descending)")
ax[1].set_xlabel("Feature Rank")

plt.tight_layout()
plt.show()

# -----------------------------
# FEATURE SELECTION
# -----------------------------
above_threshold_features_svr = importance_df[
    importance_df["Importance"] >= THRESHOLD
]

X_train_reduced_SVR = X_train[above_threshold_features_svr["Feature"]]
X_test_reduced_SVR  = X_test[X_train_reduced_SVR.columns]

print("Selected features:", X_train_reduced_SVR.shape[1])



import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# INPUT DATA (USE AVAILABLE VARIABLES)
# -----------------------------
X_train = X_train_reduced_SVR
y_train = y_train_original['label']

X_test  = X_test_reduced_SVR
y_test  = y_test_original['label']

# -----------------------------
# TRAIN SVR REGRESSOR
# -----------------------------
model_SVR = SVR(
    kernel='rbf',
    C=1,
    epsilon=0.1,
    cache_size=500
)

model_SVR.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
SVR_pred = model_SVR.predict(X_test)

m1_SVR = pd.Series(SVR_pred).round(3)
m2_SVR = y_test.reset_index(drop=True).round(3)

# -----------------------------
# METRICS
# -----------------------------
corr_pandas_SVR = m1_SVR.corr(m2_SVR)
rmse_SVR = np.sqrt(mean_squared_error(m2_SVR, m1_SVR))
r2_SVR = r2_score(m2_SVR, m1_SVR)

print("Pearson Correlation:", corr_pandas_SVR)
print("RMSE:", rmse_SVR)
print("R2 Score:", r2_SVR)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# -----------------------------
# SETTINGS
# -----------------------------
TIMESTAMPS_NUM = 10000
THRESHOLD = -0.025

EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2

# -----------------------------
# PREPARE DATA (TIME-AWARE)
# -----------------------------
X_train = x_train_reduced_post_pearson_MI.tail(TIMESTAMPS_NUM)
y_train = y_train_original['label'].tail(TIMESTAMPS_NUM)

X_test  = x_test_reduced_post_pearson_MI
y_test  = y_test_original['label']

input_dim = X_train.shape[1]

# -----------------------------
# BUILD ANN MODEL
# -----------------------------
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

ann_model.compile(
    optimizer=Adam(),
    loss='mse',
    metrics=['mae']
)

# -----------------------------
# TRAIN ANN
# -----------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

ann_model.fit(
    X_train,
    y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=0
)

# -----------------------------
# PREDICTION & METRICS
# -----------------------------
y_pred = ann_model.predict(X_test, verbose=0).flatten()

rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred))
r2_ann = r2_score(y_test, y_pred)
pearson_ann, _ = pearsonr(y_test, y_pred)

print(f"\nANN Results:")
print(f"RMSE    : {rmse_ann:.4f}")
print(f"R2      : {r2_ann:.4f}")
print(f"Pearson : {pearson_ann:.4f}")

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
# sklearn needs a wrapper-like interface â†’ use lambda
def ann_predict(X):
    return ann_model.predict(X, verbose=0).flatten()

perm_result = permutation_importance(
    estimator=ann_model,
    X=X_test.tail(TIMESTAMPS_NUM),
    y=y_test.tail(TIMESTAMPS_NUM),
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": perm_result.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances (ANN):")
print(importance_df.head())

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    data=importance_df.head(50),
    x='Feature',
    y='Importance',
    ax=ax[0]
)
ax[0].set_title('Top 50 Features - ANN (Permutation Importance)')
ax[0].tick_params(axis='x', rotation=90)

sns.lineplot(
    x=range(len(importance_df)),
    y=importance_df['Importance'],
    ax=ax[1]
)
ax[1].set_title('ANN Feature Importance (Descending)')
ax[1].set_xlabel('Feature Rank')

plt.tight_layout()
plt.show()

# -----------------------------
# FEATURE SELECTION
# -----------------------------
above_threshold_features_ANN = importance_df[
    importance_df['Importance'] >= THRESHOLD
]

X_train_reduced_ANN = X_train[above_threshold_features_ANN['Feature']]
X_test_reduced_ANN  = X_test[X_train_reduced_ANN.columns]

print("\nSelected features:", X_train_reduced_ANN.shape[1])



import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# -----------------------------
# INPUT DATA (CORRECT VARIABLES)
# -----------------------------
X_train = x_train_reduced_post_pearson_MI_ANN
y_train = y_train_original['label']

X_test  = x_test_reduced_post_pearson_MI_ANN
y_test  = y_test_original['label']

input_dim = X_train.shape[1]

# -----------------------------
# BUILD ANN MODEL
# -----------------------------
model_ANN = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model_ANN.compile(
    optimizer=Adam(),
    loss='mse',
    metrics=['mae']
)

# -----------------------------
# TRAIN ANN
# -----------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

model_ANN.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=0
)

# -----------------------------
# PREDICTION
# -----------------------------
ANN_pred = model_ANN.predict(X_test, verbose=0).flatten()

ANN_pred_df = pd.DataFrame(
    ANN_pred,
    columns=['Predictions from ANN top features - Original']
)

m1_ANN = ANN_pred_df['Predictions from ANN top features - Original'].round(3)
m2_ANN = y_test.reset_index(drop=True).round(3)

# -----------------------------
# METRICS
# -----------------------------
corr_pandas_ANN, _ = pearsonr(m1_ANN, m2_ANN)
rmse_ANN = np.sqrt(mean_squared_error(m2_ANN, m1_ANN))
r2_ANN = r2_score(m2_ANN, m1_ANN)

print(f"Pearson Correlation from ANN's best features - Original: {corr_pandas_ANN}")
print(f"Root Mean Squared Error from ANN's best features - Original: {rmse_ANN:.4f}")
print(f"R2 score for ANN from ANN's best features - original: {r2_ANN:.4f}")




































































7. INITIAL FEATURE EXPLORATION - REDUCING DIMENSIONALITY

1.RELATIVE IMPORTANCE OF FEATURES - PEARSON'S CORRELATION AND SPEARMAN'S CORRELATION
2.RELATIVE IMPORTANCE OF FEATURES - MUTUAL INFORMATION SCORE
3.RELATIVE IMPORTANCE OF FEATURES - DECISION TREE AND RANDOM FOREST APPROACHâœ… Data Drift

More precisely, in ML / data science terms, what you are seeing is:

ğŸ”¹ Covariate Shift (a type of Data Drift)
 multicollinear features (can be tested by pearson's correlation)
#4.Even though features like X1,X2 etc. percieved to have a linear relationships with label, their correlation percentage i



import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# DataFrames dictionary
# ------------------------------
dataframes = {
    'x_train': x_train,
    'x_test': x_test
}

# ==============================
# DATASET DESCRIPTION
# ==============================
for name, df in dataframes.items():
    print("\n" + "=" * 40)
    print(f"DataFrame: {name}")
    print("=" * 40)

    # Shape
    print("Shape:", df.shape)

    # Info
    print("\nInfo:")
    df.info()

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Missing values
    print("\nTotal Missing Values:", df.isna().sum().sum())

    # Duplicates
    if df.duplicated().any():
        print("Duplicates found. Removing duplicates...")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicates found.")

    # Infinite values
    if df.isin([np.inf, -np.inf]).any().any():
        print("Infinite values found.")
        cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()]
        print("Columns with infinite values:", list(cols_with_inf))
    else:
        print("No infinite values found.")

# ==============================
# SKEWNESS CHECK
# ==============================
left_skewed = []
right_skewed = []

for name, df in dataframes.items():

    for col in df.columns:
        if df[col].mean() < df[col].median():
            left_skewed.append(col)
        elif df[col].mean() > df[col].median():
            right_skewed.append(col)

    # Histogram & Violin plot for X247
    print(f"\nSkewness visualization for X247 in {name}")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Histogram of X247", "Violin Plot of X247")
    )

    fig.add_trace(go.Histogram(x=df['X247'], nbinsx=500), row=1, col=1)
    fig.add_trace(
        go.Violin(
            y=df['X247'],
            box_visible=True,
            meanline_visible=True
        ),
        row=1, col=2
    )

    fig.update_layout(title="Distribution of X247", showlegend=False)
    fig.show()

    # Q-Q plot
    sm.qqplot(df['X247'], line='s')
    plt.title(f"Q-Q Plot of X247 ({name})")
    plt.show()

print("\nLeft skewed columns:", left_skewed)
print("Right skewed columns:", right_skewed)



from sklearn.model_selection import train_test_split

x_train_original, x_test_original, y_train_original, y_test_original = train_test_split(
    x_train,
    y_train,
    test_size=0.3,
    random_state=42,
    shuffle=False
)

print("Train-test split created")




# HANDLING OUTLIERS

import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import kurtosis


class OutlierTreatment:

    # Step 1: Handle outliers and skewness
    def handle_outliers(self, data, fit_data):

        data = data.copy()
        fit_data = fit_data.copy()

        # Clip extreme values (5% to 95%)
        for col in fit_data.columns:
            lower = fit_data[col].quantile(0.05)
            upper = fit_data[col].quantile(0.95)
            data[col] = data[col].clip(lower, upper)
            fit_data[col] = fit_data[col].clip(lower, upper)

        # Find skewed columns
        skewed_cols = []
        for col in fit_data.columns:
            if abs(fit_data[col].skew()) > 1 or abs(kurtosis(fit_data[col])) > 1:
                if fit_data[col].std() > 1e-6:
                    skewed_cols.append(col)

        # Apply power transformation
        if skewed_cols:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            pt.fit(fit_data[skewed_cols])

            data[skewed_cols] = pt.transform(data[skewed_cols])
            fit_data[skewed_cols] = pt.transform(fit_data[skewed_cols])

            # Clip again to avoid extreme values
            for col in skewed_cols:
                data[col] = data[col].clip(
                    fit_data[col].quantile(0.10),
                    fit_data[col].quantile(0.90)
                )

        return data, fit_data

    # Step 2: Standard scaling
    def scale_features(self, data, fit_data, std_threshold=1e-9):

        scaler = StandardScaler()

        valid_cols = [
            col for col in fit_data.columns
            if fit_data[col].std() > std_threshold
        ]

        scaler.fit(fit_data[valid_cols])
        data[valid_cols] = scaler.transform(data[valid_cols])

        return data[valid_cols]

    # Step 3: Memory optimization
    def optimize_memory(self, data):

        for col in data.columns:
            data[col] = data[col].astype("float32")

        data.replace(
            [np.inf, -np.inf],
            [np.finfo(np.float32).max, np.finfo(np.float32).min],
            inplace=True
        )

        gc.collect()
        return data


# ===================== USAGE =====================

processor = OutlierTreatment()

# X train
x_train_clean, x_train_fit = processor.handle_outliers(
    x_train_original, x_train_original
)
x_train_original_processed = processor.optimize_memory(
    processor.scale_features(x_train_clean, x_train_fit)
)

# X test (fit on train)
x_test_clean, _ = processor.handle_outliers(
    x_test_original, x_train_original
)
x_test_original_processed = processor.optimize_memory(
    processor.scale_features(x_test_clean, x_train_fit)
)

# y train
y_train_clean, y_train_fit = processor.handle_outliers(
    y_train_original, y_train_original
)
y_train_original_processed = processor.optimize_memory(
    processor.scale_features(y_train_clean, y_train_fit)
)

# y test
y_test_clean, _ = processor.handle_outliers(
    y_test_original, y_train_original
)
y_test_original_processed = processor.optimize_memory(
    processor.scale_features(y_test_clean, y_train_fit)
)

print("All data processed and converted to float32")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# function to check distribution of a numeric column
def check_distribution(df, col):
    data = df[col].dropna()

    # basic stats
    mean = np.mean(data)
    median = np.median(data)
    skewness = skew(data)
    kurt = kurtosis(data)

    print("Distribution for:", col)
    print("Mean    =", round(mean, 4))
    print("Median  =", round(median, 4))
    print("Skewness=", round(skewness, 4))
    print("Kurtosis=", round(kurt, 4))
    print("Note: skew > 1 means right skewed, < -1 means left skewed")

    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # histogram + kde
    sns.histplot(data, bins=40, kde=True, ax=ax[0], color="skyblue")
    ax[0].set_title("Histogram + KDE: " + col)

    # boxplot
    sns.boxplot(x=data, ax=ax[1], color="lightcoral")
    ax[1].set_title("Boxplot: " + col)

    plt.tight_layout()
    plt.show()



check_distribution(df,"label")


var=df.var().sort_values(ascending=False)
info=df.info()
null=df.isnull().sum()
duplicate_columns = df.T.duplicated()
duplicate_rows = df.duplicated()


dup=duplicate_columns.sort_values(ascending=False)
print(dup.head(29))
rowdup=duplicate_rows.sort_values(ascending=False)
print(rowdup.head(20))


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
bad_cols = ['X716', 'X711', 'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704',
            'X705', 'X706', 'X707', 'X708', 'X710', 'X709', 'X712', 'X713', 'X714', 'X715',
            'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872']
#df = df.drop(columns=bad_cols)


import pandas as pd
pd.set_option('display.max_rows', 10) 

pd.set_option('display.max_columns', 15)    
pd.set_option('display.width', 1000)        
pd.set_option('display.expand_frame_repr', False) 
missing_summary = []

for col in df.columns:
    total = len(df[col])
    missing = df[col].isnull().sum()
    percent_missing = (missing / total) * 100

    missing_summary.append({
        'column': col,
        'missing_count': missing,
        'percent_missing': percent_missing
    })

missing_df = pd.DataFrame(missing_summary).sort_values('percent_missing', ascending=True)
print(missing_df.head(20))



import pandas as pd

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    total = len(df[col])
    outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
    percent_outliers = outliers / total * 100
    
    outlier_summary.append({
        'column': col,
        'outlier_count': outliers,
        'percent_outlier': percent_outliers
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('percent_outlier', ascending=False)
print(outlier_df.head(20))



from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0099)
df_new= selector.fit_transform(df.drop('label', axis=1))



from sklearn.preprocessing import PowerTransformer, QuantileTransformer, FunctionTransformer

def auto_transform_numeric(df):
    """
    Automatically transform numeric columns based on skewness and kurtosis.
    Returns a new DataFrame with transformed numeric columns only.
    """
    transformed_df = pd.DataFrame(index=df.index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        
        # Decide transformer
        if abs(skew) < 1 and abs(kurt) < 3:
            # roughly normal â†’ no transform
            transformer = FunctionTransformer(func=None)
        elif skew > 1:
            # right-skewed
            if (df[col] <= 0).any():
                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                transformer = FunctionTransformer(func=np.log1p)
        elif skew < -1:
            # left-skewed â†’ Yeo-Johnson
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            # heavy-tailed â†’ QuantileTransformer
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        
        # Fit and transform
        transformed = transformer.fit_transform(df[[col]])
        
        # Ensure it's 1D
        if isinstance(transformed, pd.DataFrame):
            transformed = transformed.values
        transformed_df[col] = transformed.ravel()  # safer than flatten()
    
    return transformed_df


df_transformed = auto_transform_numeric(df)
df_transformed.head(5)
 


var=df_transformed.var().sort_values(ascending=False)
var.tail(20)



means=df.mean()
std=df.std()
coefficient_variation=std/means
var=df.var()
var.sort_values(ascending=False) 


corr_matrix = df.corr().abs()  
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print("Highly correlated features to consider dropping:")
print(to_drop)
df_reduced = df.drop(columns=to_drop)
print("Shape before dropping:", df.shape)
print("Shape after dropping:", df_reduced.shape)
df_reduced.head()
df_reduced.shape


corr_matrix = df.corr(numeric_only=True).abs()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features = [col for col in df.columns if col != 'target']
X = df[features]
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_scaled)

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features after PCA: {X_pca.shape[1]}")
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca['target'] = y.values
print(df_pca.head())





import pandas as pd
from sklearn.feature_selection import mutual_info_regression
X = df.drop(columns=["label"])
y = df["label"]
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "Mutual_Information": mi_scores
})
mi_df = mi_df.sort_values(by="Mutual_Information", ascending=False)

print("Top 15 features by Mutual Information with label:")
print(mi_df.head(15))



corr_feature_label = df.corrwith(df["label"])
corr_df = corr_feature_label.reset_index()
corr_df.columns = ["Feature", "Correlation_with_Label"]
corr_df = corr_df.sort_values(by="Correlation_with_Label", key=abs, ascending=False)
print("Top 15 correlated features with label:")
print(corr_df.head(15))



top_corr = corr_df.head(20).set_index("Feature")
plt.figure(figsize=(6,8))
sns.heatmap(top_corr, annot=True, cmap="coolwarm", cbar=False)
plt.title("Top 20 Features vs Label Correlation")
plt.show()


df1=df_transformed.head(5)
df2=df.head(5)
print(df1)
print(df2)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data_sample=df.head(2000)
X = data_sample.drop("label", axis=1)
y = data_sample["label"]
data_test = df.iloc[2000:2500]
X_test = data_test .drop("label", axis=1)
y_test = data_test ["label"]
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 score:", r2)
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 important features:\n", importances.sort_values(ascending=False).head(10))



import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


if 'label' in df_reduced.columns:
    X = df_reduced.drop('label', axis=1)
    y = df_reduced['label']
else:
   
    X = df_reduced
    y = df['label']



X_train = X.iloc[:split_point]
X_test  = X.iloc[split_point:]  
y_train = y.iloc[:split_point]
y_test  = y.iloc[split_point:]

print(f"Training on {len(X_train)} rows (Past)")
print(f"Testing on  {len(X_test)} rows (Future)")


print("\nTraining LightGBM...")
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"\n--- TIME SERIES RESULTS ---")
print(f"Model Error (RMSE): {rmse}")

