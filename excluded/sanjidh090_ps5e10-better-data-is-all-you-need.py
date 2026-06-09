import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('//kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


target = 'accident_risk'


orig_dfs = []
for k in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig_dfs.append(df)
orig = pd.concat(orig_dfs, axis=0, ignore_index=True)
orig['id'] = np.arange(len(orig)) + test['id'].max() + 1
orig = orig[train.columns]

TARGET = 'accident_risk'
print(f"Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")


train['id'].value_counts()


orig['id'].value_counts()


test['id'].value_counts()


train = pd.concat([train, orig], ignore_index=True, sort=False)


train.shape


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

def convert_all_to_numeric(train, test):
    """
    Converts ALL columns in train/test to numeric (LabelEncoding for objects),
    safely handling columns that exist only in one dataset.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    tr, te = train.copy(), test.copy()
    encoders = {}

    # unify columns intersection (ignore target like Dropout thatâ€™s train-only)
    common_cols = sorted(set(tr.columns).intersection(set(te.columns)))

    for col in common_cols:
        if tr[col].dtype == "O" or str(tr[col].dtype).startswith("category"):
            le = LabelEncoder()
            combined = pd.concat([tr[col].astype(str), te[col].astype(str)], axis=0)
            le.fit(combined)
            tr[col] = le.transform(tr[col].astype(str))
            te[col] = le.transform(te[col].astype(str))
            encoders[col] = le
        else:
            tr[col] = pd.to_numeric(tr[col], errors="coerce")
            te[col] = pd.to_numeric(te[col], errors="coerce")

    # handle train-only numeric columns like target
    for col in set(tr.columns) - set(te.columns):
        if tr[col].dtype == "O":
            le = LabelEncoder()
            tr[col] = le.fit_transform(tr[col].astype(str))
        else:
            tr[col] = pd.to_numeric(tr[col], errors="coerce")

    # clean infinities / NaNs
    tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    te.replace([np.inf, -np.inf], np.nan, inplace=True)
    tr.fillna(tr.mean(numeric_only=True), inplace=True)
    te.fillna(te.mean(numeric_only=True), inplace=True)

    num_cols = tr.select_dtypes(include="number").columns.tolist()
    return tr, te, num_cols


def dist_plots(train, test, num_features):
    """
    Plot KDE + Boxplots for numeric columns.
    """
    print("\nDistribution analysis (all numeric/object columns converted)\n")
    df = pd.concat(
        [train[num_features].assign(Source="Train"),
         test[num_features].assign(Source="Test")],
        axis=0, ignore_index=True
    )

    n = len(num_features)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(18, n * 4),
        gridspec_kw={"hspace": 0.3, "wspace": 0.2, "width_ratios": [0.70, 0.30]}
    )
    if n == 1:
        axes = np.array([axes])

    for i, col in enumerate(num_features):
        # KDE
        ax = axes[i, 0]
        sns.kdeplot(data=df, x=col, hue="Source",
                    palette=["#3cb371", "#0483ff"], ax=ax, linewidth=2)
        ax.set(xlabel="", ylabel="")
        ax.set_title(f"{col}")
        ax.grid()

        # Boxplot
        ax = axes[i, 1]
        sns.boxplot(data=df, y=col, x="Source", width=0.5,
                    linewidth=1, fliersize=1, ax=ax, palette=["#3cb371", "b"])
        ax.set(xlabel="", ylabel="")
        ax.set_title(f"{col}")
        ax.set_xticklabels(["Train", "Test"])

    plt.tight_layout()
    plt.show()
    
# 1ï¸�âƒ£ Convert everything to numeric
tr_all, te_all, numeric_cols = convert_all_to_numeric(train, test)

print("Numeric columns used for distribution plots:")
print(numeric_cols)

# 2ï¸�âƒ£ Plot all feature distributions
dist_plots(tr_all, te_all, [c for c in numeric_cols if c != target])




train.columns


# Distribution of Target Variable

y_train = train['accident_risk']

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.1) 
ax_hist = fig.add_subplot(grid[0:3, 0]) 
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_train, bins=50, kde=True, color='orange', ax=ax_hist, legend=False)
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

print("ğŸ�¨ Classic Donut Chart Comparison of Categorical Variables in Train & Test Datasets ğŸ�¨")

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
        f"ğŸ“Š Donut Comparison: {variable}",
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



from scipy import stats
#target visualization
target_col = "accident_risk"
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram with KDE
axes[0].hist(train[target_col], bins=50, density=True, alpha=0.7, 
             color='red', edgecolor='black')
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



train.info()


FEATURES = list( orig.columns[1:-1] )
TARGET = orig.columns[-1]
print(f"Features: {FEATURES}, Target: '{TARGET}'")


# # https://www.kaggle.com/competitions/playground-series-s5e10/discussion/609994#3296622
# import scipy

# def f(X):
#     return \
#     0.35 * X["curvature"] + \
#     0.05 * (X["lighting"] == "night").astype(int) + \
#     0.1 * (X["weather"] != "clear").astype(int) + \
#     0.35 * (X["speed_limit"] >= 60).astype(int) + \
#     0.2 * (X["num_reported_accidents"] > 2).astype(int)

# def clip(f):
#     def clip_f(X):
#         sigma = 0.05
#         mu = f(X)
#         a, b = -mu/sigma, (1-mu)/sigma
#         Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
#         phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
#         return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
#     return clip_f

# z = clip(f)(combine)
# combine["y"] = z.values
# FEATURES.append("y")


# # Optimized Road Accident Risk Prediction with Optuna & Adaptive Feature Engineering
# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import xgboost as xgb
# import scipy.stats
# import optuna
# from optuna.samplers import TPESampler

# # ======================== DATA LOADING ========================
# train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
# sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# # Load original data for target encoding
# orig_dfs = []
# for k in [2, 10, 100]:
#     df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
#     orig_dfs.append(df)
# orig = pd.concat(orig_dfs, axis=0, ignore_index=True)
# orig['id'] = np.arange(len(orig)) + test['id'].max() + 1
# orig = orig[train.columns]

# TARGET = 'accident_risk'
# print(f"Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")

# # ======================== ADAPTIVE FEATURE ENGINEERING ========================
# def engineer_features(df, trial=None):
#     """Adaptive feature engineering controlled by Optuna trial"""
#     df = df.copy()
    
#     # Default values if no trial
#     use_interaction = trial.suggest_categorical('use_interaction', [True, False]) if trial else True
#     use_polynomial = trial.suggest_categorical('use_polynomial', [True, False]) if trial else True
#     use_binning = trial.suggest_categorical('use_binning', [True, False]) if trial else True
#     use_risk_scores = trial.suggest_categorical('use_risk_scores', [True, False]) if trial else True
#     use_complex_interactions = trial.suggest_categorical('use_complex_interactions', [True, False]) if trial else True
#     poly_degree = trial.suggest_int('poly_degree', 2, 3) if trial else 2
#     n_bins = trial.suggest_int('n_bins', 3, 5) if trial else 4
    
#     if use_risk_scores:
#         # High-risk combinations
#         df['night_bad_weather'] = ((df['lighting'] == 'night') & (df['weather'] != 'clear')).astype(int)
#         df['high_speed_curve'] = ((df['speed_limit'] >= 60) & (df['curvature'] > 0.5)).astype(int)
#         df['freq_accidents_high_speed'] = ((df['num_reported_accidents'] > 2) & (df['speed_limit'] >= 60)).astype(int)
        
#         # Risk score components
#         df['speed_risk'] = (df['speed_limit'] >= 60).astype(int) * df['speed_limit'] / 100
#         df['accident_history_risk'] = np.clip(df['num_reported_accidents'] / 5, 0, 1)
#         df['curvature_risk'] = np.clip(df['curvature'], 0, 1)
        
#         # Weather severity encoding
#         weather_severity = {'clear': 0, 'fog': 0.3, 'rain': 0.5, 'snow': 0.7}
#         df['weather_severity'] = df['weather'].map(weather_severity).fillna(0.5)
        
#         # Lighting risk
#         lighting_risk = {'day': 0, 'dusk': 0.3, 'night': 0.5}
#         df['lighting_risk'] = df['lighting'].map(lighting_risk).fillna(0.25)
    
#     if use_interaction:
#         # Basic interactions
#         df['curvature_x_speed'] = df['curvature'] * df['speed_limit'] / 100
#         df['accidents_x_curvature'] = df['num_reported_accidents'] * df['curvature']
#         df['speed_x_accidents'] = (df['speed_limit'] / 100) * df['num_reported_accidents']
        
#         if use_risk_scores:
#             df['weather_x_lighting'] = df['weather_severity'] * df['lighting_risk']
    
#     if use_complex_interactions:
#         # Three-way interactions
#         df['speed_curve_accidents'] = (df['speed_limit'] / 100) * df['curvature'] * df['num_reported_accidents']
#         if use_risk_scores:
#             df['risk_composite'] = df['speed_risk'] * df['curvature_risk'] * df['accident_history_risk']
    
#     if use_polynomial:
#         # Polynomial features
#         df['curvature_pow'] = df['curvature'] ** poly_degree
#         df['speed_pow'] = (df['speed_limit'] / 100) ** poly_degree
#         df['accidents_pow'] = df['num_reported_accidents'] ** poly_degree
        
#         # Log transforms
#         df['log_speed'] = np.log1p(df['speed_limit'])
#         df['log_accidents'] = np.log1p(df['num_reported_accidents'])
#         df['sqrt_curvature'] = np.sqrt(df['curvature'])
    
#     if use_binning:
#         # Binned features
#         df['speed_bin'] = pd.cut(df['speed_limit'], bins=n_bins, labels=False).astype(int)
#         df['curvature_bin'] = pd.cut(df['curvature'], bins=n_bins, labels=False).astype(int)
#         df['accidents_bin'] = pd.cut(df['num_reported_accidents'], bins=n_bins, labels=False).astype(int)
    
#     return df

# # ======================== SYNTHETIC TARGET (Y) ========================
# def compute_synthetic_target(X):
#     return (0.3 * X["curvature"] +
#             0.2 * (X["lighting"] == "night").astype(int) +
#             0.1 * (X["weather"] != "clear").astype(int) +
#             0.1 * (X["speed_limit"] >= 60).astype(int) +
#             0.1 * (X["num_reported_accidents"] > 3).astype(int))

# def clip_target(f):
#     def clip_f(X):
#         sigma = 0.05
#         mu = f(X)
#         a, b = -mu/sigma, (1-mu)/sigma
#         Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
#         phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
#         return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b
#     return clip_f

# # ======================== OPTUNA OBJECTIVE ========================
# def objective(trial):
#     """Optuna objective function for hyperparameter tuning"""
    
#     # Apply feature engineering with trial
#     train_fe = engineer_features(train, trial)
#     test_fe = engineer_features(test, trial)
#     orig_fe = engineer_features(orig, trial)
    
#     # Combine all data
#     combine = pd.concat([train_fe, test_fe, orig_fe], axis=0, ignore_index=True)
#     combine["y"] = clip_target(compute_synthetic_target)(combine).values
    
#     # Identify all categorical columns
#     CATS = combine.select_dtypes(include=['object']).columns.tolist()
#     CATS = [c for c in CATS if c not in ['id', TARGET]]
    
#     # Label encode ALL categoricals
#     for c in CATS:
#         le = LabelEncoder()
#         combine[c] = le.fit_transform(combine[c].astype(str))
#         combine[c] = combine[c].astype('int32')
    
#     # Split back
#     train_split = combine.iloc[:len(train)].copy()
#     test_split = combine.iloc[len(train):len(train)+len(test)].copy()
#     orig_split = combine.iloc[-len(orig):].copy()
    
#     # Target encoding settings
#     use_te_mean = trial.suggest_categorical('use_te_mean', [True, False])
#     use_te_std = trial.suggest_categorical('use_te_std', [True, False])
#     use_te_interactions = trial.suggest_categorical('use_te_interactions', [True, False])
    
#     TE_FEATURES = []
#     base_features = ['lighting', 'weather', 'speed_limit', 'curvature', 'num_reported_accidents']
#     if 'speed_bin' in train_split.columns:
#         base_features.extend(['speed_bin', 'curvature_bin', 'accidents_bin'])
    
#     for c in base_features:
#         if c not in orig_split.columns:
#             continue
        
#         if use_te_mean:
#             te_mean = orig_split.groupby(c)[TARGET].mean()
#             te_col = f"TE_mean_{c}"
#             train_split[te_col] = train_split[c].map(te_mean).fillna(orig_split[TARGET].mean())
#             test_split[te_col] = test_split[c].map(te_mean).fillna(orig_split[TARGET].mean())
#             TE_FEATURES.append(te_col)
        
#         if use_te_std:
#             te_std = orig_split.groupby(c)[TARGET].std()
#             te_col_std = f"TE_std_{c}"
#             train_split[te_col_std] = train_split[c].map(te_std).fillna(orig_split[TARGET].std())
#             test_split[te_col_std] = test_split[c].map(te_std).fillna(orig_split[TARGET].std())
#             TE_FEATURES.append(te_col_std)
    
#     if use_te_interactions:
#         interaction_pairs = [('lighting', 'weather')]
#         if 'speed_bin' in train_split.columns:
#             interaction_pairs.extend([('speed_bin', 'curvature_bin'), ('weather', 'speed_bin')])
        
#         for c1, c2 in interaction_pairs:
#             if c1 not in orig_split.columns or c2 not in orig_split.columns:
#                 continue
#             # Create interaction column as numeric hash
#             orig_split[f'{c1}_{c2}'] = orig_split[c1] * 1000 + orig_split[c2]
#             train_split[f'{c1}_{c2}'] = train_split[c1] * 1000 + train_split[c2]
#             test_split[f'{c1}_{c2}'] = test_split[c1] * 1000 + test_split[c2]
            
#             te_inter = orig_split.groupby(f'{c1}_{c2}')[TARGET].mean()
#             te_col = f"TE_{c1}_{c2}"
#             train_split[te_col] = train_split[f'{c1}_{c2}'].map(te_inter).fillna(orig_split[TARGET].mean())
#             test_split[te_col] = test_split[f'{c1}_{c2}'].map(te_inter).fillna(orig_split[TARGET].mean())
#             TE_FEATURES.append(te_col)
#             # Drop the interaction column as we only need the TE
#             train_split.drop(f'{c1}_{c2}', axis=1, inplace=True)
#             test_split.drop(f'{c1}_{c2}', axis=1, inplace=True)
    
#     NUMS = [c for c in train_split.columns if c not in CATS + ['id', TARGET, 'y'] and c not in TE_FEATURES]
#     FEATURES = NUMS + CATS + TE_FEATURES + ['y']
    
#     # XGBoost parameters
#     params = {
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#         "max_depth": trial.suggest_int("max_depth", 5, 10),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
#         "subsample": trial.suggest_float("subsample", 0.7, 0.95),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
#         "gamma": trial.suggest_float("gamma", 0.0, 1.0),
#         "seed": 42,
#         "device": "cuda",
#     }
    
#     # Quick 3-fold CV for optimization
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)
#     cv_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_split)):
#         X_train = train_split.iloc[train_idx][FEATURES].copy()
#         y_train = train_split.iloc[train_idx][TARGET] - train_split.iloc[train_idx]['y']
        
#         X_valid = train_split.iloc[val_idx][FEATURES].copy()
#         y_valid = train_split.iloc[val_idx][TARGET] - train_split.iloc[val_idx]['y']
#         y_valid_synthetic = train_split.iloc[val_idx]['y'].values
        
#         dtrain = xgb.DMatrix(X_train, label=y_train)
#         dval = xgb.DMatrix(X_valid, label=y_valid)
        
#         model = xgb.train(
#             params=params,
#             dtrain=dtrain,
#             num_boost_round=5000,
#             evals=[(dval, "valid")],
#             early_stopping_rounds=100,
#             verbose_eval=False
#         )
        
#         preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) + y_valid_synthetic
#         fold_rmse = np.sqrt(np.mean((preds - train_split.iloc[val_idx][TARGET].values) ** 2))
#         cv_scores.append(fold_rmse)
    
#     return np.mean(cv_scores)

# # ======================== OPTUNA OPTIMIZATION ========================
# print("\n" + "="*60)
# print("Starting Optuna Hyperparameter Optimization")
# print("="*60)

# study = optuna.create_study(
#     direction="minimize",
#     sampler=TPESampler(seed=42),
#     study_name="road_accident_optimization"
# )

# study.optimize(objective, n_trials=50, timeout=10000, show_progress_bar=True)

# print("\n" + "="*60)
# print("Optimization Complete!")
# print("="*60)
# print(f"Best RMSE: {study.best_value:.6f}")
# print("\nBest Parameters:")
# for key, value in study.best_params.items():
#     print(f"  {key}: {value}")

# # ======================== FINAL TRAINING WITH BEST PARAMS ========================
# print("\n" + "="*60)
# print("Training Final Model with Best Parameters")
# print("="*60)

# # Reconstruct best trial
# best_trial = study.best_trial
# train_final = engineer_features(train, best_trial)
# test_final = engineer_features(test, best_trial)
# orig_final = engineer_features(orig, best_trial)

# combine = pd.concat([train_final, test_final, orig_final], axis=0, ignore_index=True)
# combine["y"] = clip_target(compute_synthetic_target)(combine).values

# # Encode categoricals
# CATS = combine.select_dtypes(include=['object']).columns.tolist()
# CATS = [c for c in CATS if c not in ['id', TARGET]]

# for c in CATS:
#     le = LabelEncoder()
#     combine[c] = le.fit_transform(combine[c].astype(str))
#     combine[c] = combine[c].astype('int32')

# train_final = combine.iloc[:len(train)].copy()
# test_final = combine.iloc[len(train):len(train)+len(test)].copy()
# orig_final = combine.iloc[-len(orig):].copy()

# # Target encoding with best params
# TE_FEATURES = []
# base_features = ['lighting', 'weather', 'speed_limit', 'curvature', 'num_reported_accidents']
# if 'speed_bin' in train_final.columns:
#     base_features.extend(['speed_bin', 'curvature_bin', 'accidents_bin'])

# for c in base_features:
#     if c not in orig_final.columns:
#         continue
    
#     if best_trial.params.get('use_te_mean', True):
#         te_mean = orig_final.groupby(c)[TARGET].mean()
#         te_col = f"TE_mean_{c}"
#         train_final[te_col] = train_final[c].map(te_mean).fillna(orig_final[TARGET].mean())
#         test_final[te_col] = test_final[c].map(te_mean).fillna(orig_final[TARGET].mean())
#         TE_FEATURES.append(te_col)
    
#     if best_trial.params.get('use_te_std', False):
#         te_std = orig_final.groupby(c)[TARGET].std()
#         te_col_std = f"TE_std_{c}"
#         train_final[te_col_std] = train_final[c].map(te_std).fillna(orig_final[TARGET].std())
#         test_final[te_col_std] = test_final[c].map(te_std).fillna(orig_final[TARGET].std())
#         TE_FEATURES.append(te_col_std)

# if best_trial.params.get('use_te_interactions', False):
#     interaction_pairs = [('lighting', 'weather')]
#     if 'speed_bin' in train_final.columns:
#         interaction_pairs.extend([('speed_bin', 'curvature_bin'), ('weather', 'speed_bin')])
    
#     for c1, c2 in interaction_pairs:
#         if c1 not in orig_final.columns or c2 not in orig_final.columns:
#             continue
#         # Create interaction column as numeric hash (SAME AS OPTIMIZATION)
#         orig_final[f'{c1}_{c2}'] = orig_final[c1] * 1000 + orig_final[c2]
#         train_final[f'{c1}_{c2}'] = train_final[c1] * 1000 + train_final[c2]
#         test_final[f'{c1}_{c2}'] = test_final[c1] * 1000 + test_final[c2]
        
#         te_inter = orig_final.groupby(f'{c1}_{c2}')[TARGET].mean()
#         te_col = f"TE_{c1}_{c2}"
#         train_final[te_col] = train_final[f'{c1}_{c2}'].map(te_inter).fillna(orig_final[TARGET].mean())
#         test_final[te_col] = test_final[f'{c1}_{c2}'].map(te_inter).fillna(orig_final[TARGET].mean())
#         TE_FEATURES.append(te_col)
#         # Drop the interaction column as we only need the TE
#         train_final.drop(f'{c1}_{c2}', axis=1, inplace=True)
#         test_final.drop(f'{c1}_{c2}', axis=1, inplace=True)

# NUMS = [c for c in train_final.columns if c not in CATS + ['id', TARGET, 'y'] and c not in TE_FEATURES]
# FEATURES = NUMS + CATS + TE_FEATURES + ['y']

# print(f"Total features: {len(FEATURES)}")

# # Extract best XGBoost params
# best_xgb_params = {
#     "objective": "reg:squarederror",
#     "eval_metric": "rmse",
#     "seed": 42,
#     "device": "cuda",
# }
# for key in ['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 
#             'colsample_bytree', 'colsample_bylevel', 'reg_alpha', 'reg_lambda', 'gamma']:
#     best_xgb_params[key] = best_trial.params[key]

# # Full training with 11 folds
# FOLDS = 7
# oof_preds = np.zeros(len(train_final))
# test_preds = np.zeros(len(test_final))
# feature_importance = np.zeros(len(FEATURES))

# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# for fold, (train_idx, val_idx) in enumerate(kf.split(train_final)):
#     print(f"\nFold {fold+1}/{FOLDS}")
    
#     X_train = train_final.iloc[train_idx][FEATURES].copy()
#     y_train = train_final.iloc[train_idx][TARGET] - train_final.iloc[train_idx]['y']
    
#     X_valid = train_final.iloc[val_idx][FEATURES].copy()
#     y_valid = train_final.iloc[val_idx][TARGET] - train_final.iloc[val_idx]['y']
#     y_valid_synthetic = train_final.iloc[val_idx]['y'].values
    
#     X_test = test_final[FEATURES].copy()
#     y_test_synthetic = test_final['y'].values
    
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dval = xgb.DMatrix(X_valid, label=y_valid)
#     dtest = xgb.DMatrix(X_test)
    
#     model = xgb.train(
#         params=best_xgb_params,
#         dtrain=dtrain,
#         num_boost_round=100_000,
#         evals=[(dtrain, "train"), (dval, "valid")],
#         early_stopping_rounds=300,
#         verbose_eval=500
#     )
    
#     oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) + y_valid_synthetic
#     test_preds += (model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) + y_test_synthetic) / FOLDS
    
#     importance = model.get_score(importance_type='gain')
#     for i, feat in enumerate(FEATURES):
#         feature_importance[i] += importance.get(f'f{i}', 0) / FOLDS
    
#     fold_rmse = np.sqrt(np.mean((oof_preds[val_idx] - train_final.iloc[val_idx][TARGET].values) ** 2))
#     print(f"Fold {fold+1} RMSE: {fold_rmse:.6f}")

# # ======================== RESULTS ========================
# cv_rmse = np.sqrt(np.mean((oof_preds - train_final[TARGET].values) ** 2))
# baseline_rmse = np.sqrt(np.mean((train_final['y'].values - train_final[TARGET].values) ** 2))

# print(f"\n{'='*60}")
# print(f"Final CV RMSE: {cv_rmse:.6f}")
# print(f"Baseline RMSE: {baseline_rmse:.6f}")
# print(f"Improvement: {baseline_rmse - cv_rmse:.6f}")
# print(f"{'='*60}")

# # Top features
# feat_df = pd.DataFrame({'feature': FEATURES, 'importance': feature_importance})
# feat_df = feat_df.sort_values('importance', ascending=False).head(20)
# print("\nTop 20 Features:")
# print(feat_df.to_string(index=False))

# # ======================== SUBMISSION ========================
# sub[TARGET] = test_preds
# sub.to_csv("submission_cared.csv", index=False)
# print(f"\nSubmission saved! Predicted range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
# print(sub.head(10))


import pandas as pd
import numpy as np
import scipy.stats
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import optuna  # <-- 1. Import Optuna
from optuna.samplers import TPESampler # Tree-structured Parzen Estimator sampler

# --- No changes to your functions ---

def f(X):
    return \
    0.35 * X["curvature"] + \
    0.05 * int(X["lighting"] == "night") + \
    0.1 * int(X["weather"] != "clear") + \
    0.35 * int(X["speed_limit"] >= 60) + \
    0.2 * int(X["num_reported_accidents"] > 2)

class Clipper:
    def __init__(self, f_func):
        self.f_func = f_func
        self.sigma = 0.05

    def __call__(self, X):
        mu = self.f_func(X)
        sigma = self.sigma
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b

def feature_engineering_with_clip(train_df, test_df, target):
    train, test = train_df.copy(), test_df.copy()
    cols = train.drop(columns=target).columns.tolist()
    cat = [col for col in cols if train[col].dtype in ["object", "category"] and col != target]
    num = [col for col in cols if train[col].dtype not in ["object", "category", "bool"] and col not in ["id", target]]

    for col in cat:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

    for col in num:
        for q in [5, 10, 15]:
            try:
                train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
            except Exception:
                train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    map_col = "num_reported_accidents"
    if map_col in train.columns:
        map_num_reported = {0: 0, 1: 0, 2: 0, 3: 2, 4: 4, 5: 3, 6: 1, 7: 0}
        train[map_col] = train[map_col].map(map_num_reported)
        test[map_col] = test[map_col].map(map_num_reported)

    remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq", "id"]
    train.drop(columns=[col for col in remove if col in train.columns], inplace=True)
    test.drop(columns=[col for col in remove if col in test.columns], inplace=True)
    train.drop_duplicates(inplace=True)

    cat = [col for col in cat if col in train.columns]
    for col in cat:
        if col in test.columns:
            train[col] = train[col].astype("category")
            test[col] = test[col].astype("category")

    clipper = Clipper(f)
    train["curvature_clipped"] = train.apply(clipper, axis=1)
    test["curvature_clipped"] = test.apply(clipper, axis=1)

    new_num = train.drop(columns=cat + [target]).columns.tolist()
    return train, test, new_num, clipper

# --- Data Loading and Prep ---
df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test_original = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test_ids = df_test_original["id"]
target = df.columns.tolist()[-1]
df, df_test, new_num, clipper_object = feature_engineering_with_clip(df, df_test_original, target)

print("Processed training data head:")
print(df.head())

# Prepare data for XGBoost
X_train = df.drop(columns=target)
y_train = df[target]
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

# --- 2. Start Optuna Hyperparameter Tuning ---

# Define static parameters (these won't be tuned)
static_params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'random_state': 42,
    'max_bin': 512,
    # Using your pre-calculated scale_pos_weight
    'scale_pos_weight': 0.3615894752587659,
}

def objective(trial):
    """
    The objective function for Optuna to minimize.
    It suggests hyperparameters, runs xgb.cv, and returns the best RMSE.
    """
    # 3. Define the hyperparameter search space
    # I've centered the search around your original parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 5),
    }

    # Add the static parameters
    params.update(static_params)

    # Run cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        nfold=7,
        num_boost_round=2000,
        metrics='rmse',
        verbose_eval=False,  # Set to True if you want to see CV output for each trial
        early_stopping_rounds=50
    )

    # Get the best RMSE and the round it occurred at
    best_rmse = cv_results['test-rmse-mean'].min()
    best_round = cv_results['test-rmse-mean'].idxmin() + 1 # +1 since idxmin is 0-indexed

    # 4. Store the best round as a "user attribute"
    trial.set_user_attr("best_round", best_round)
    
    return best_rmse

# 5. Create and run the Optuna study
print("Starting hyperparameter tuning...")
sampler = TPESampler(seed=42)  # Use a seeded sampler for reproducibility
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=50, show_progress_bar=True) # You can increase n_trials for a better search

print(f"Tuning finished. Best trial RMSE: {study.best_value:.7f}")
print("Best params found: ")
print(study.best_params)

# --- 6. Final Model Training using Best Parameters ---

# Get the best hyperparameters from the study
best_params = study.best_params

# Get the optimal number of estimators from the best trial's attributes
best_n_estimators = study.best_trial.user_attrs["best_round"]
print(f"Optimal n_estimators found: {best_n_estimators}")

# Create the final parameter set
final_params = static_params.copy()
final_params.update(best_params)
final_params["n_estimators"] = best_n_estimators

# Train the final XGBoost model
print("Training final model with best parameters...")
model = XGBRegressor(**final_params, enable_categorical=True)
model.fit(X_train, y_train)

# --- No changes to model saving and prediction ---

# Save the model using joblib
joblib.dump(model, "xgboost_model.pkl")
print("Model saved to xgboost_model.pkl")

# Save the clipper object
joblib.dump(clipper_object, "feature_engineering_clipper.pkl")
print("Feature engineering clipper saved to feature_engineering_clipper.pkl")

# Predict on test set
print("Predicting on test set...")
pred = model.predict(df_test)

# Prepare submission
sub = pd.DataFrame({
    "id": test_ids,
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)
print("Submission file created successfully!")


# # Predict on test set
# print("Predicting on test set...")
# pred = model.predict(df_test)

# # Prepare submission
# sub = pd.DataFrame({
#     "id": test_ids,
#     target: pred
# })

# # Save submission file
# sub.to_csv("submission.csv", index=False)
# print("Submission file created successfully!")


train


# import matplotlib.pyplot as plt

# plt.scatter(train[TARGET].values,oof_preds,s=0.25)
# plt.plot([0,1],[0,1],'--',color='black')
# plt.title("True vs Predicted")
# plt.xlabel("True Target")
# plt.ylabel("Predicted Target")
# plt.show()

