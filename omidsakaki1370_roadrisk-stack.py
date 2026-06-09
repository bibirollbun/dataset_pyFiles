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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna  # For hyperparameter tuning

import shap  # For explainability
from scipy import stats  # For interactions
import gc  # Memory management

# Set styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 120


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
print("Load data")


orig_dfs = []
for k in [2, 10, 100]:
    orig_df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig_dfs.append(orig_df)
orig = pd.concat(orig_dfs, ignore_index=True)
orig['id'] = np.arange(len(orig)) + test['id'].max() + 1
orig = orig[train.columns]  # Align columns

print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")


all_data = pd.concat([train, test, orig], ignore_index=True)
all_data['source'] = ['train']*len(train) + ['test']*len(test) + ['orig']*len(orig)

TARGET = 'accident_risk'
ID = 'id'
FEATURES = [col for col in train.columns if col not in [ID, TARGET]]

# Categorical & Numerical
CATS = [col for col in FEATURES if all_data[col].dtype == 'object']
NUMS = [col for col in FEATURES if col not in CATS]

print(f"Categorical: {CATS}, Numerical: {NUMS}")


print("\n=== 2. Combine train + external data ===")
train_full = pd.concat([train, orig], ignore_index=True) if orig is not None else train.copy()
print(f"train_full shape : {train_full.shape}")


print("\n=== 2.1 Descriptive statistics (numeric only) ===")
numeric_cols = train_full[FEATURES + [TARGET]].select_dtypes(include=[np.number]).columns
desc = train_full[numeric_cols].describe().T
desc['skew'] = train_full[numeric_cols].skew()
desc['kurt'] = train_full[numeric_cols].kurtosis()
print(desc.round(3))


print("\n=== 2.2 Missing values & unique counts ===")
miss_unique = pd.DataFrame({
    'missing': train_full[FEATURES + [TARGET]].isnull().sum(),
    'unique' : train_full[FEATURES + [TARGET]].nunique()
})
print(miss_unique)


print("\n=== 2.3 Target distribution ===")

MAX_ROWS = 100_000
df_plot = train_full.sample(n=min(MAX_ROWS, len(train_full)), random_state=42)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Histogram + KDE', 'Boxplot', 'QQ Plot', 'CDF')
)

# ---- Histogram + KDE (same data, different norm) ----
fig.add_trace(
    go.Histogram(x=df_plot[TARGET], nbinsx=60,
                 name='Histogram', opacity=0.7),
    row=1, col=1
)
fig.add_trace(
    go.Histogram(x=df_plot[TARGET], histnorm='probability density',
                 name='KDE', opacity=0.5, nbinsx=60),
    row=1, col=1
)

# ---- Boxplot ------------------------------------------------
fig.add_trace(go.Box(y=df_plot[TARGET], name='Box'), row=1, col=2)

# ---- QQ‑plot ------------------------------------------------
prob = stats.probplot(df_plot[TARGET], dist="norm")
fig.add_trace(
    go.Scatter(x=prob[0][0], y=prob[0][1],
               mode='markers', name='QQ', marker=dict(size=4)),
    row=2, col=1
)
slope, intercept, _ = prob[1]
theoretical = slope * prob[0][0] + intercept
fig.add_trace(
    go.Scatter(x=prob[0][0], y=theoretical,
               mode='lines', name='Theoretical', line=dict(color='red')),
    row=2, col=1
)

# ---- CDF (use the *sorted* sample) -------------------------
sorted_target = np.sort(df_plot[TARGET])
cdf_y = np.arange(1, len(sorted_target)+1) / len(sorted_target)
fig.add_trace(
    go.Scatter(x=sorted_target, y=cdf_y,
               mode='lines', name='CDF', line=dict(color='green')),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Target Distribution Analysis (sample ≤ 100 k rows)",
    showlegend=False
)
fig.show()


print("\n=== 2.4 Correlation heatmap ===")
corr = train_full[NUMS + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap='coolwarm', center=0)
plt.title("Correlation matrix (numeric + target)")
plt.show()


print("\n=== 2.5 Outliers (IQR method) ===")
numeric_nums = [c for c in NUMS if train_full[c].dtype in ['float64','float32','int64','int32']]
outliers = {}
for col in numeric_nums:
    Q1 = train_full[col].quantile(0.25)
    Q3 = train_full[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers[col] = ((train_full[col] < lower) | (train_full[col] > upper)).sum()

out_df = pd.DataFrame(outliers.items(), columns=['Feature','Outliers Count'])
print(out_df.sort_values('Outliers Count', ascending=False))


print("\n=== 2.6 Pair‑plot (top numeric features) ===")
top_num = ['curvature','speed_limit','num_reported_accidents']
sns.pairplot(train_full[top_num + [TARGET]], diag_kind='kde', corner=True)
plt.suptitle("Pair‑plot of key numeric features + target", y=1.02)
plt.show()


print("\n=== 2.7 Violin plots (top 3 numeric vs target) ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(top_num):
    sns.violinplot(data=train_full, x=pd.cut(train_full[col], bins=5), y=TARGET,
                   ax=axes[i], inner='quartile')
    axes[i].set_title(f'{col} (binned) vs {TARGET}')
plt.tight_layout()
plt.show()


print("\n=== 2.8 Categorical analysis (train only) ===")
train_plot = train_full.dropna(subset=[TARGET]).copy()

# Optional: limit to a random sample for speed
if len(train_plot) > 50_000:
    train_plot = train_plot.sample(n=50_000, random_state=42)

for cat in CATS:
    if cat not in train_plot.columns:
        continue

    # Keep only the 10 most frequent categories (prevents huge plots)
    top_cats = train_plot[cat].value_counts().head(10).index
    df_sub = train_plot[train_plot[cat].isin(top_cats)]

    fig = px.histogram(
        df_sub,
        x=cat,
        color=TARGET,
        marginal='box',
        title=f'{cat} vs {TARGET}',
        barmode='overlay',
        histnorm='probability',
        opacity=0.7,
        nbins=50
    )
    fig.update_layout(height=500, legend_title=TARGET)
    fig.show()


print("\n=== 2.9 PCA projection (sampled) ===")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MAX_ROWS = 100_000
sample_df = train_full.sample(n=min(MAX_ROWS, len(train_full)), random_state=42)

scaler = StandardScaler()
num_scaled = scaler.fit_transform(sample_df[NUMS].fillna(0))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(num_scaled)

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df[TARGET] = sample_df[TARGET].values

fig = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    color=TARGET,
    title=f'PCA (2‑D) – {len(sample_df):,} rows sampled',
    color_continuous_scale='Viridis',
    opacity=0.7,
    hover_data={TARGET: False}
)
fig.update_layout(height=600, width=800)
fig.show()

print(
    f"Explained variance → PC1: {pca.explained_variance_ratio_[0]:.3f} "
    f"({pca.explained_variance_ratio_[0]*100:.1f}%), "
    f"PC2: {pca.explained_variance_ratio_[1]:.3f} "
    f"({pca.explained_variance_ratio_[1]*100:.1f}%)"
)
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")


print("\n=== 2.10 Target distribution by source ===")
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('Train', 'External (orig)', 'Test'))

# Train
fig.add_trace(go.Histogram(x=train[TARGET], name='train', opacity=0.7), row=1, col=1)

# External (orig) – only rows that have target
if orig is not None and TARGET in orig.columns:
    fig.add_trace(go.Histogram(x=orig[TARGET], name='orig', opacity=0.7), row=1, col=2)

# Test – we do NOT have target, so we show a placeholder
fig.add_trace(go.Histogram(x=[0]*len(test), name='test (no target)', opacity=0.3), row=1, col=3)

fig.update_layout(title_text="Target distribution by data source")
fig.show()


freq_enc = {}
for col in CATS:
    freq = all_data.groupby(col)[col].transform('count') / len(all_data)
    train[col + '_freq'] = freq.loc[train.index]
    test[col + '_freq'] = freq.loc[len(train):len(train)+len(test)]
    orig[col + '_freq'] = freq.loc[len(train)+len(test):]
    freq_enc[col] = True
    FEATURES.append(col + '_freq')


train['curv_speed_int'] = train['curvature'] * train['speed_limit']
test['curv_speed_int'] = test['curvature'] * test['speed_limit']
orig['curv_speed_int'] = orig['curvature'] * orig['speed_limit']
FEATURES.append('curv_speed_int')

train['weather_light_int'] = pd.factorize(train['weather'].astype(str) + '_' + train['lighting'].astype(str))[0]
test['weather_light_int'] = pd.factorize(test['weather'].astype(str) + '_' + test['lighting'].astype(str))[0]
orig['weather_light_int'] = pd.factorize(orig['weather'].astype(str) + '_' + orig['lighting'].astype(str))[0]
FEATURES.append('weather_light_int')


def advanced_physical_model(X):
    # Weights from research: curvature (0.35), speed (0.25), weather/lighting (0.2 each), accidents (0.15)
    weather_risk = (X['weather'] != 'clear').astype(int) * 0.2
    light_risk = (X['lighting'] == 'night').astype(int) * 0.2  # Assume 'night' from data
    speed_risk = (X['speed_limit'] >= 60).astype(int) * 0.25
    curv_risk = X['curvature'] * 0.35
    acc_risk = np.clip(X['num_reported_accidents'] / 10, 0, 1) * 0.15  # Normalized
    base_risk = curv_risk + speed_risk + weather_risk + light_risk + acc_risk
    # Clip with sigmoid for [0,1]
    return 1 / (1 + np.exp(- (base_risk - 0.5) * 10))  # Sigmoid for smoothness

train['physical_risk'] = advanced_physical_model(train)
test['physical_risk'] = advanced_physical_model(test)
orig['physical_risk'] = advanced_physical_model(orig)
FEATURES.append('physical_risk')


print("\n=== Scaling numeric features (RobustScaler) ===")
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
train[NUMS] = scaler.fit_transform(train[NUMS]) 

test.loc[:, NUMS] = scaler.transform(test[NUMS])
if orig is not None:
    orig.loc[:, NUMS] = scaler.transform(orig[NUMS])

print(f"Scaling completed – train: {train.shape}, test: {test.shape}, orig: {orig.shape if orig is not None else 'None'}")


print("\nWeighting external data (only numeric columns)...")
numeric_cols = orig.select_dtypes(include=[np.number]).columns
orig_weighted = orig.copy()
orig_weighted[numeric_cols] = orig[numeric_cols] * 0.3

train_full = pd.concat([train, orig_weighted], axis=0, ignore_index=True)
print(f"train_full shape: {train_full.shape}")


y_physical = train['physical_risk']
rmse_physical = np.sqrt(mean_squared_error(train[TARGET], y_physical))
print(f"Physical Baseline RMSE : {rmse_physical:.6f}")


train['residual'] = train[TARGET] - y_physical
X = train[FEATURES].copy()
y = train['residual']


from sklearn.preprocessing import LabelEncoder
le_dict = {}
print("Label‑encoding categorical columns...")
for col in CATS:
    le = LabelEncoder()
    # Fit on train + test → no unseen categories in test
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    if orig is not None and col in orig.columns:
        orig[col] = le.transform(orig[col].astype(str))
    le_dict[col] = le


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=pd.qcut(train[TARGET], 5, labels=False)
)


print("Initializing base models...")

lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=128,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False,
    task_type="GPU" if 'CUDA_VISIBLE_DEVICES' in os.environ else "CPU",
    cat_features=CATS          # CatBoost still needs the original column names
)


hgb_model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=8,
    min_samples_leaf=20, random_state=42,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=50
)

base_models = [lgb_model, cat_model, hgb_model]


print("Training base models on the single split…")

meta_val  = np.zeros((len(X_val), len(base_models)))   # validation
meta_test = np.zeros((len(test), len(base_models)))    # test

for i, model in enumerate(base_models):
    print(f"  → {model.__class__.__name__}")

    if i == 0:  # LightGBM
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
    elif i == 1:  # CatBoost
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=200,
            use_best_model=True
        )
    else:
        model.fit(X_train, y_train)

    meta_val[:, i]  = model.predict(X_val)
    meta_test[:, i] = model.predict(test[FEATURES])

    rmse = np.sqrt(mean_squared_error(y_val, meta_val[:, i]))
    print(f"    Val RMSE (residual) : {rmse:.6f}")


from xgboost import XGBRegressor
print("Training meta‑model (XGBoost)…")
meta_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
meta_model.fit(meta_val, y_val)


from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm   # nice progress bar (Jupyter)

print("\nRunning 7‑fold cross‑validation…")
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

# Wrap the fold loop with tqdm for a clean progress bar
for fold, (tr_idx, val_idx) in enumerate(tqdm(
        skf.split(X, pd.qcut(train[TARGET], 10, labels=False)),
        total=skf.n_splits, desc="Folds")):

    # ---- fold data -------------------------------------------------
    X_tr, X_vl = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[val_idx]
    phys_vl    = train.iloc[val_idx]['physical_risk'].values

    # ---- meta‑feature matrices for this fold -----------------------
    fold_meta_val  = np.zeros((len(X_vl), len(base_models)))
    fold_meta_test = np.zeros((len(test), len(base_models)))

    # ---- train base models on this fold ----------------------------
    for i, base in enumerate(base_models):
        model = clone(base)

        if i == 0:                                 # LightGBM
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
            )
        elif i == 1:                               # CatBoost – early stopping!
            model.fit(
                X_tr, y_tr,
                eval_set=(X_vl, y_vl),
                early_stopping_rounds=100,     # stop if no improvement for 100 rounds
                verbose=200,                   # print every 200 iterations
                use_best_model=True
            )
        else:                                      # HistGB / LinearRegression
            model.fit(X_tr, y_tr)

        # Store predictions
        fold_meta_val[:, i]  = model.predict(X_vl)
        fold_meta_test[:, i] = model.predict(test[FEATURES])

    # ---- OOF prediction for this fold -------------------------------
    oof_preds[val_idx] = meta_model.predict(fold_meta_val) + phys_vl

    # ---- test prediction (average over folds) -----------------------
    test_preds += (meta_model.predict(fold_meta_test) +
                   test['physical_risk'].values) / skf.n_splits

print("7‑fold CV completed!")


cv_rmse = np.sqrt(mean_squared_error(train[TARGET], oof_preds))
improvement = (rmse_physical - cv_rmse) / rmse_physical * 100
print(f"\nFinal CV RMSE : {cv_rmse:.6f}")
print(f"Improvement over physical baseline : {improvement:.2f}%")


print("\nGenerating SHAP summary…")
explainer = shap.TreeExplainer(lgb_model)
sample_X = X_train.sample(n=min(5_000, len(X_train)), random_state=42)
shap_vals = explainer.shap_values(sample_X)
shap.summary_plot(shap_vals, sample_X, max_display=15)


sub = sample_sub.copy()
TARGET_COL = sub.columns[1]

sub[TARGET_COL] = np.clip(test_preds, 0.0, 1.0)
sub.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")
print(sub.head())


# Clean up memory safely

import gc

# List of variables to delete (only if they exist)
to_delete = [
    'X', 'y', 'train', 'test', 'orig',
    'X_train', 'X_val', 'y_train', 'y_val',
    'meta_val', 'meta_test', 'oof_preds', 'test_preds',
    'base_models', 'meta_model',
    'lgb_model', 'cat_model', 'hgb_model'
]

# Delete only existing variables
for var in to_delete:
    if var in globals():
        del globals()[var]

# Force garbage collection
gc.collect()

