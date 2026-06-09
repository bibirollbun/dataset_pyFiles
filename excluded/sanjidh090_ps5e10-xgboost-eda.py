# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
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


train = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.columns


# Distribution of Target Variable

y_train = train['accident_risk']

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.1) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_train, bins=50, kde=True, color='red', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of accident_risk (Target Variable)")
ax_hist.set_xlabel("")

sns.boxplot(x=y_train, ax=ax_box, color='yellow')
ax_box.set_xlabel("accident_risk")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

print("ðŸŽ¨ Classic Donut Chart Comparison of Categorical Variables in Train & Test Datasets ðŸŽ¨")

# Elegant, classic palette â€” warm, vintage tone
classic_palette = ["#3B82F6", "#EAB308", "#10B981", "#EF4444", "#8B5CF6", "#F59E0B"]

# Get categorical/boolean columns
obj_cols = train.select_dtypes(include=['object', 'bool']).columns

sns.set_style("white")

for variable in obj_cols:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(wspace=0.35)
    fig.patch.set_facecolor("#FDFCF8")  # soft ivory background

    # Overall title
    fig.suptitle(
        f"ðŸ“Š Donut Comparison: {variable}",
        fontsize=15,
        fontweight="bold",
        color="#1E3A8A",   # deep navy
        y=1.03,
        fontname="Georgia"
    )

    # ===== Train Donut =====
    train_counts = train[variable].value_counts()
    colors = sns.color_palette(classic_palette, len(train_counts))
    wedges, texts, autotexts = axes[0].pie(
        train_counts,
        labels=train_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.55, edgecolor='white'),
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#1F2937")  # charcoal text
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    axes[0].set_title(
        f"Train [{variable}]",
        fontsize=12,
        fontweight="bold",
        color="#334155"
    )
    axes[0].set_facecolor("#FDFCF8")

    # ===== Test Donut =====
    test_counts = test[variable].value_counts()
    colors = sns.color_palette(classic_palette, len(test_counts))
    wedges, texts, autotexts = axes[1].pie(
        test_counts,
        labels=test_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.55, edgecolor='white'),
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#1F2937")
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    axes[1].set_title(
        f"Test [{variable}]",
        fontsize=12,
        fontweight="bold",
        color="#334155"
    )
    axes[1].set_facecolor("#FDFCF8")

    # Borders off for a clean look
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()



# Converting Object Columns to Categorical Type

for col in train.select_dtypes(include='object').columns:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


train_col = [ 'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 'school_season', 'num_reported_accidents', 'accident_risk']
#drop id



for i in range(len(train_col)):
    c = train_col[i]
    print(f"{train[c].value_counts()}")
    print(f"="*50)


print(f" Train shape: {train.shape}")
print(f" Test shape : {test.shape}")


target = 'accident_risk'


train.drop_duplicates()


num_cols =  train.select_dtypes(include="number").columns.tolist()
cat_cols = train.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")


# Define color palette
palette = sns.color_palette("tab10" ,len(num_cols))

#to show Distribution
plt.figure(figsize=(25, 15))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[col], kde=True, color=palette[i-1], bins=10)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()



plt.figure(figsize = (8,6))
correlation_matrix = train[num_cols +[target] ].corr()
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm',linewidths=2)
plt.show()




#Categorical Columns Unique
for i in cat_cols :
   print(f" {i} (Uniques): {train[i].unique()}")


from scipy import stats
#target visualization
target_col = "accident_risk"
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram with KDE
axes[0].hist(train[target_col], bins=50, density=True, alpha=0.7, 
             color='blue', edgecolor='black')
axes[0].set_xlabel('Accident Risk', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Distribution of Accident Risk (Training Data)', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Q-Q plot for normality check
stats.probplot(train[target_col], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Normality Assessment', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical tests
shapiro_stat, shapiro_p = stats.shapiro(train[target_col].sample(min(5000, len(train))))
print("\nShapiro-Wilk Test for Normality:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  Interpretation: Data is approximately Normal distribution")
else:
    print("  Interpretation: Data is NOT Normal distribution")




!pip install --upgrade xgboost scikit-learn

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()

def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.

    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test, new_num

"""# Data Processing"""

# Identify feature
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test, new_num = create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)

print(df.columns.tolist())

df.head()

"""# CV score of the model"""

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(df.drop(columns=target), label=df[target], enable_categorical=True)

# Define XGBoost parameters
xgb_params  = {
    'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'rmse',
    'random_state': 42,'max_bin': 512, 'min_child_weight': 3,
    'max_delta_step': 1, 'max_depth': 11, 'learning_rate': 0.010453775390437146,
    'subsample': 0.8162196077561874,'colsample_bytree': 0.8057453252225478,
    'gamma': 0.011515371568909936,'reg_alpha': 0.1153674139991063,
    'reg_lambda': 0.4029264986439234,'colsample_bylevel': 0.8675078626084138,
    'colsample_bynode': 0.8804930677965951,'scale_pos_weight': 0.3615894752587659,
}

# Run cross-validation
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=7,
    num_boost_round=2000,
    metrics='rmse',
    verbose_eval=100,
    early_stopping_rounds=50
)

# Display last few CV results
print(cv_results.tail())

# Extract best boosting round
best_round = cv_results['test-rmse-mean'].idxmin()
best_rmse = cv_results['test-rmse-mean'][best_round]
print(f"Best round: {best_round}, Best CV RMSE: {best_rmse:.7f}")

# putting the n_estimator at the average early stopping point to avoid overfitting
last_round = len(cv_results) - 1
xgb_params["n_estimators"] = last_round + 10

"""# Final training and submitting"""

# Prepare training data
X_train = df.drop(columns=target)
y_train = df[target]

# Train XGBoost model
model = XGBRegressor(**xgb_params, enable_categorical=True)
model.fit(X_train, y_train)

# Predict on test set
pred = model.predict(df_test.drop(columns = "id"))

# Prepare submission
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})

# Save submission file
sub.to_csv("submission_Best_yet.csv", index=False)

"""Thanks to the Meta Models notebook, I obtained some of the feature engineering from it. The notebook is available here: [Link](http://www.kaggle.com/code/metamodels/single-simple-xgb-with-cv-0-05595)"""


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
!pip install --upgrade xgboost scikit-learn lightgbm catboost

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Data loading
try:
    df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
    df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
except FileNotFoundError:
    print("Ensure data files 'train.csv' and 'test.csv' are in the specified path.")
    raise

target = df.columns.tolist()[-1]
print(f"Training shape: {df.shape}")
print(f"Test shape: {df_test.shape}")
df.head()

def create_advanced_features(train_df, test_df, cols, num, cat):
    """
    Create comprehensive feature engineering including frequency, binning, 
    interactions, and statistical aggregations.
    """
    train, test = train_df.copy(), test_df.copy()
    
    # 1. Frequency encoding
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())
        
        # Binning for numeric features
        if col in num:
            for q in [5, 10, 15, 20]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0
    
    # 2. Interaction features between key numerical columns
    if 'speed_limit' in num and 'curvature' in num:
        train['speed_curvature'] = train['speed_limit'] * train['curvature']
        test['speed_curvature'] = test['speed_limit'] * test['curvature']
        
        train['speed_curvature_ratio'] = train['speed_limit'] / (train['curvature'] + 1)
        test['speed_curvature_ratio'] = test['speed_limit'] / (test['curvature'] + 1)
    
    if 'num_reported_accidents' in cols:
        train['accidents_squared'] = train['num_reported_accidents'] ** 2
        test['accidents_squared'] = test['num_reported_accidents'] ** 2
    
    # 3. Statistical aggregations by categorical features
    for cat_col in cat:
        if cat_col in train.columns:
            for num_col in ['speed_limit', 'curvature', 'num_reported_accidents']:
                if num_col in train.columns:
                    # Mean encoding
                    agg_mean = train.groupby(cat_col)[num_col].mean()
                    train[f'{cat_col}_{num_col}_mean'] = train[cat_col].map(agg_mean)
                    test[f'{cat_col}_{num_col}_mean'] = test[cat_col].map(agg_mean).fillna(agg_mean.mean())
                    
                    # Std encoding
                    agg_std = train.groupby(cat_col)[num_col].std()
                    train[f'{cat_col}_{num_col}_std'] = train[cat_col].map(agg_std)
                    test[f'{cat_col}_{num_col}_std'] = test[cat_col].map(agg_std).fillna(agg_std.mean())
    
    new_num = train.drop(columns=cat+[target] if target in train.columns else cat).columns.tolist()
    return train, test, new_num

"""# Data Processing"""

# Identify features
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object", "category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object", "category", "bool"] and col not in ["id", target]]

print(f"Categorical features: {cat}")
print(f"Numerical features: {num}")

# Creating advanced features
df, df_test, new_num = create_advanced_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat] = df[cat].astype("category")
df_test[cat] = df_test[cat].astype("category")

# Enhanced mapping for num_reported_accidents
map_col = "num_reported_accidents"
if map_col in df.columns:
    map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
    df[map_col] = df[map_col].map(map_num_reported)
    df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
df = df.drop(columns=[c for c in remove if c in df.columns])
df_test = df_test.drop(columns=[c for c in remove if c in df_test.columns])

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)

print(f"\nFinal training shape: {df.shape}")
print(f"Final features: {len(df.columns) - 1}")

"""# XGBoost CV and Training"""

# Prepare data
X_train = df.drop(columns=target)
y_train = df[target]
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

# Optimized XGBoost parameters
xgb_params = {
    'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'rmse',
    'random_state': 42, 'max_bin': 512, 'min_child_weight': 3,
    'max_delta_step': 1, 'max_depth': 11, 'learning_rate': 0.010453775390437146,
    'subsample': 0.8162196077561874, 'colsample_bytree': 0.8057453252225478,
    'gamma': 0.011515371568909936, 'reg_alpha': 0.1153674139991063,
    'reg_lambda': 0.4029264986439234, 'colsample_bylevel': 0.8675078626084138,
    'colsample_bynode': 0.8804930677965951, 'scale_pos_weight': 0.3615894752587659,
}

print("Starting XGBoost 7-Fold Cross-Validation...")
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=7,
    num_boost_round=2500,
    metrics='rmse',
    verbose_eval=100,
    early_stopping_rounds=75,
    seed=42
)

best_round_xgb = cv_results['test-rmse-mean'].idxmin()
best_rmse_xgb = cv_results['test-rmse-mean'][best_round_xgb]
print(f"XGBoost Best round: {best_round_xgb}, Best CV RMSE: {best_rmse_xgb:.7f}")

last_round_xgb = len(cv_results) - 1
xgb_params["n_estimators"] = last_round_xgb + 20

"""# LightGBM CV and Training"""

# Optimized LightGBM parameters
lgb_params = {
    'objective': 'rmse', 'metric': 'rmse', 'n_estimators': 2500,
    'learning_rate': 0.008, 'feature_fraction': 0.75, 'bagging_fraction': 0.75,
    'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    'max_depth': 12, 'min_child_samples': 15, 'num_leaves': 300,
    'reg_alpha': 0.1, 'reg_lambda': 0.3, 'min_split_gain': 0.01,
    'extra_trees': False
}

lgb_dtrain = lgb.Dataset(X_train, y_train, categorical_feature='auto')

print("\nStarting LightGBM 7-Fold Cross-Validation...")
lgb_cv_results = lgb.cv(
    params=lgb_params,
    train_set=lgb_dtrain,
    nfold=7,
    num_boost_round=2500,
    callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=75)],
    return_cvbooster=False,
    stratified=False,
    seed=42
)

best_round_lgb = len(lgb_cv_results['valid rmse-mean'])
best_rmse_lgb = min(lgb_cv_results['valid rmse-mean'])
print(f"LightGBM Best round: {best_round_lgb}, Best CV RMSE: {best_rmse_lgb:.7f}")

lgb_params["n_estimators"] = best_round_lgb + 20

"""# CatBoost Training (Additional Model)"""

# CatBoost parameters - using CPU as fallback
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.01,
    'depth': 10,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 100,
    'task_type': 'CPU',  # Changed to CPU for compatibility
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 75,
    'thread_count': -1  # Use all available CPU cores
}

# Identify categorical feature indices for CatBoost
cat_features_idx = [i for i, col in enumerate(X_train.columns) if col in cat]

print("\nTraining CatBoost model (using CPU)...")
try:
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(
        X_train, y_train,
        cat_features=cat_features_idx,
        verbose=100
    )
    use_catboost = True
except Exception as e:
    print(f"CatBoost training failed: {e}")
    print("Continuing with XGBoost + LightGBM ensemble only...")
    use_catboost = False

"""# Final Training and Ensemble Prediction"""

# 1. Train XGBoost model
print("\nTraining final XGBoost model...")
model_xgb = XGBRegressor(**xgb_params, enable_categorical=True)
model_xgb.fit(X_train, y_train)
model_xgb.save_model("xgboost_model.json")
print("XGBoost model saved as xgboost_model.json")

pred_xgb = model_xgb.predict(df_test.drop(columns="id"))

# 2. Train LightGBM model
print("\nTraining final LightGBM model...")
model_lgb = LGBMRegressor(**lgb_params)
model_lgb.fit(
    X_train, y_train,
    categorical_feature='auto',
    callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
)
model_lgb.booster_.save_model("lightgbm_model.txt")
print("LightGBM model saved as lightgbm_model.txt")

pred_lgb = model_lgb.predict(df_test.drop(columns="id"))

# 3. CatBoost predictions (if available)
if use_catboost:
    print("\nGenerating CatBoost predictions...")
    pred_cat = model_cat.predict(df_test.drop(columns="id"))
    model_cat.save_model("catboost_model.cbm")
    print("CatBoost model saved as catboost_model.cbm")
    
    # 4. Weighted Ensemble with 3 models
    weight_xgb = 0.35
    weight_lgb = 0.35
    weight_cat = 0.30
    pred_ensemble = (weight_xgb * pred_xgb + weight_lgb * pred_lgb + weight_cat * pred_cat)
else:
    print("\nUsing XGBoost + LightGBM ensemble only...")
    # 4. Two-model ensemble
    weight_xgb = 0.50
    weight_lgb = 0.50
    pred_ensemble = (weight_xgb * pred_xgb + weight_lgb * pred_lgb)

# Prepare submissions
sub_ensemble = pd.DataFrame({
    "id": df_test["id"],
    target: pred_ensemble
})
sub_ensemble.to_csv("submission_ensemble.csv", index=False)

sub_xgb = pd.DataFrame({
    "id": df_test["id"],
    target: pred_xgb
})
sub_xgb.to_csv("submission_xgb.csv", index=False)

sub_lgb = pd.DataFrame({
    "id": df_test["id"],
    target: pred_lgb
})
sub_lgb.to_csv("submission_lgb.csv", index=False)

if use_catboost:
    sub_cat = pd.DataFrame({
        "id": df_test["id"],
        target: pred_cat
    })
    sub_cat.to_csv("submission_cat.csv", index=False)

print("\n" + "="*50)
print("SUBMISSION FILES CREATED:")
print("="*50)
if use_catboost:
    print(f"1. submission_ensemble.csv - Weighted ensemble of all 3 models")
    print(f"2. submission_xgb.csv - XGBoost only (CV RMSE: {best_rmse_xgb:.7f})")
    print(f"3. submission_lgb.csv - LightGBM only (CV RMSE: {best_rmse_lgb:.7f})")
    print(f"4. submission_cat.csv - CatBoost only")
else:
    print(f"1. submission_ensemble.csv - XGBoost + LightGBM ensemble")
    print(f"2. submission_xgb.csv - XGBoost only (CV RMSE: {best_rmse_xgb:.7f})")
    print(f"3. submission_lgb.csv - LightGBM only (CV RMSE: {best_rmse_lgb:.7f})")
print("\nRecommendation: Try the ensemble first, then individual models")
print("="*50)


# from sklearn.model_selection import train_test_split
# train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
# # Converting Object Columns to Categorical Type

# for col in train.select_dtypes(include='object').columns:
#     train[col] = train[col].astype('category')
#     test[col] = test[col].astype('category')



# # Prepare input data
# X_train = train.drop(['accident_risk'], axis=1)
# y_train = train['accident_risk']

# # Split the dataset into training and validation data
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# print("X_train Summary: First Rows, Shape, Dtypes")

# display(X_train.head(10).T, X_train.shape, X_train.dtypes)

