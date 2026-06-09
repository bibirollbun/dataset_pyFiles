# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Basic Imports
import pandas as p
import numpy as n
import matplotlib.pyplot as m
import seaborn as s
import warnings as w

# Machine Learning Core
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ignore warnings for clean output
w.filterwarnings('ignore')

# Matte Neon Hybrid Theme
s.set_theme(style='whitegrid')
m.style.use('dark_background')
m.rcParams.update({
    'axes.facecolor': '#111111',
    'figure.facecolor': '#0A0A0A',
    'axes.edgecolor': '#00FFB3',
    'axes.labelcolor': '#E0E0E0',
    'text.color': '#E0E0E0',
    'xtick.color': '#BFBFBF',
    'ytick.color': '#BFBFBF',
    'grid.color': '#333333'
    })


#Basic Statistics
def load_and_describe_data(data):

    # Basic Info
    print("Dataset Shape:", data.shape)
    print("Dataset Size:", data.size)
    print("\nData Types:\n", data.dtypes)
    print("\nMissing Values per Column:\n", data.isnull().sum())

    # Numerical Summary
    numeric_data = data.select_dtypes(include=[n.number])
    print("\nStatistical Summary:\n", numeric_data.describe())

    # Average Skewness
    avg_skewness = numeric_data.skew().mean()
    print("\nAverage Skewness:", round(avg_skewness, 3))

    # Outlier Score (based on IQR)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
    total_outliers = outliers.sum()
    outlier_score = total_outliers / numeric_data.size * 100

    print("Total Outliers:", total_outliers)
    print(f"Outlier Score: {outlier_score:.2f}%")

    return ""


df = pd.read_csv("/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv")
df['harvest_month'] = pd.to_datetime(df['harvest_date']).dt.month
df = df.drop(columns=['id', 'field_id', 'harvest_date'])


load_and_describe_data(df)


def explore_categorical_data(data, target_col='yield_tpha'):

    cat_cols = data.select_dtypes(include=['object']).columns[:4]
    fig, axes = m.subplots(1, 3, figsize=(12, 5))
    axes = axes.flatten()

    palette = ["#00FFB3", "#00BFFF", "#FF6EC7", "#FFD700"]

    for i, col in enumerate(cat_cols):
        s.boxplot(x=col, y=target_col, data=data, ax=axes[i],
                  palette=palette, linewidth=1.4, fliersize=3,
                  boxprops=dict(alpha=0.9), whiskerprops=dict(color='#00FFB3'))
        axes[i].set_title(f'{target_col} by {col}', fontsize=10, color='#E0E0E0')
        axes[i].tick_params(axis='x', rotation=25)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    m.tight_layout()
    m.show()


explore_categorical_data(df)


import matplotlib.pyplot as m
import seaborn as s
import numpy as n
import pandas as p

def explore_numeric_data_distribution(data):
    # Select numeric columns
    num_cols = data.select_dtypes(include=[n.number]).columns
    # Distribution plots in grid
    cols_to_plot = num_cols[:12]  # only first few for clarity

    fig, axes = m.subplots(4, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        s.histplot(data[col], kde=True, color="#00FFB3", ax=axes[i])
        axes[i].set_title(col, fontsize=9, color="#E0E0E0")
        axes[i].tick_params(colors="#BFBFBF")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    m.tight_layout()
    m.show()

explore_numeric_data_distribution(df)


from matplotlib.colors import LinearSegmentedColormap

neon_green = LinearSegmentedColormap.from_list(
        "neon_green", ["#000000", "#003300", "#00FF66", "#99FFCC"]
)

# Correlation heatmap
num_cols = df.select_dtypes(include=[n.number])
m.figure(figsize=(12, 6))
s.heatmap(num_cols.corr(), cmap=neon_green, annot=False, linewidths=0.5)
m.title("Correlation Heatmap")
m.tight_layout()
m.show()


s.pairplot(df, vars=['soil_ph', 'soil_moisture', 'avg_temperature', 'yield_tpha'],
           diag_kind='kde', plot_kws={'alpha': 0.2, 'color': '#00FF99'})


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ========================================
# Preprocessing Pipeline (no feature engineering)
# ========================================

# numeric pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# categorical pipeline
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# combine numeric and categorical
preprocessor = ColumnTransformer([
    ('num', num_pipe, make_column_selector(dtype_include=['int64', 'float64'])),
    ('cat', cat_pipe, make_column_selector(dtype_include=['object', 'category']))
])



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def train_super_stacker(X, y, preprocessor, cv=3, random_state=42):

    print("\nStacking Ensemble (RMSE Optimization)\n")

    # === Define a wide set of strong, diverse, and lightweight regressors ===
    base_models = {
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state
        ),
        "CatBoost": CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, verbose=0, random_state=random_state
        ),
        "Ridge": RidgeCV(alphas=(0.1, 1.0, 10.0)),
        "Lasso": Lasso(alpha=0.001, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state),
        "LinearSVM": LinearSVR(C=1.0, epsilon=0.2, random_state=random_state, max_iter=5000),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, max_depth=8, random_state=random_state
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5)
    }

    # === Cross-validation setup ===
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    # === Evaluate base models ===
    for name, model in base_models.items():
        pipeline = make_pipeline(preprocessor, model)
        scores = cross_val_score(
            pipeline, X, y, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rmse = -scores
        results.append({
            "Model": name,
            "Mean_RMSE": rmse.mean(),
            "Std_RMSE": rmse.std()
        })
        print(f"{name:<20} | RMSE: {rmse.mean():.4f} Â± {rmse.std():.4f}")

    # === Create results summary ===
    results_df = pd.DataFrame(results).sort_values(by="Mean_RMSE", ascending=True)
    print("\nBase Model Summary:")
    print(results_df.head(10))

    # === Build Stacking Ensemble ===
    estimators = [(name, model) for name, model in base_models.items()]
    meta_learner = RidgeCV(alphas=(0.1, 1.0, 10.0))

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        passthrough=True,
        n_jobs=-1
    )

    stacked_pipeline = make_pipeline(preprocessor, stacking_model)

    # === Evaluate stacked model ===
    stacked_scores = cross_val_score(
        stacked_pipeline, X, y, cv=kf,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    stacked_rmse = -stacked_scores

    print("\nStacked Ensemble RMSE: "
          f"{stacked_rmse.mean():.4f} Â± {stacked_rmse.std():.4f}")

    # === Fit final model on full data ===
    stacked_pipeline.fit(X, y)
    print("\nFinal super-stacked model trained successfully.")

    return stacked_pipeline, results_df


X =  df.drop('yield_tpha', axis=1)
y = df['yield_tpha']


stacked_model, result_df = train_super_stacker(X, y, preprocessor, cv=3, random_state=42)


X_test = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv', index_col='id')

X_test['harvest_month'] = pd.to_datetime(X_test['harvest_date']).dt.month
X_test = X_test.drop(columns=['field_id', 'harvest_date'])

preds = stacked_model.predict(X_test)

# build submission dataframe
submission = pd.DataFrame({
    'id': X_test.index,
    'yield_tpha': preds
})

# save to csv
submission.to_csv('yield_pred_submission(2).csv', index=False)

print("submission.csv created successfully.")
print(submission.head())


# ==========================================================
# Crop Yield Prediction 
# ==========================================================
# Author: Hassan Rasheed
# Date: 2025-10-25
#
# Hey everyone ğŸ‘‹
# This is my simple approach for the crop yield prediction task.
# I focused on building a clean preprocessing + stacking setup 
# that performs well without overcomplicating things.
#
# I know my variable names (p, n, s, m) look funny ğŸ˜… â€” 
# itâ€™s just a habit to code faster.
#
# Thanks to everyone sharing ideas on the forums â€” 
# I learned a lot from your notebooks too. ğŸ™Œ
#
# If this helped you in any way, feel free to upvote â€” it really means a lot. â�¤ï¸�
# ==========================================================


