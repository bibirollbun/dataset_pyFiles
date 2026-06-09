import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

import seaborn as sns
import matplotlib.pyplot as plt

import random

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import category_encoders as ce

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice

sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor": "white",    
    "figure.autolayout": True,     
})

SEED = 42

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv").set_index("id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv").set_index("id")
train.head()


train.describe()


train.isna().sum()


train['dataset'] = 'train'
test['dataset'] = 'test'

df_all = pd.concat([train, test], ignore_index=True)


def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    return lower, upper, len(outliers), len(outliers)/len(series)*100


features = [col for col in train.columns if col  not in ['dataset', 'BeatsPerMinute']]
plt.figure(figsize=(18, 12))
for i, col in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(data=df_all, x=col, hue='dataset', kde=True, bins=30, palette=['skyblue','salmon'], alpha=0.5)
    plt.xticks(rotation=30)
    plt.title(f'{col}')

    skew_train = skew(train[col])
    kurt_train = kurtosis(train[col])
    lower_train, upper_train, out_train_count, out_train_pct = iqr_outliers(train[col])

    skew_test = skew(test[col])
    kurt_test = kurtosis(test[col])
    lower_test, upper_test, out_test_count, out_test_pct = iqr_outliers(test[col])

    plt.text(0.95, 0.95,
             f'Train: skew={skew_train:.2f}, kurt={kurt_train:.2f}, outliers={out_train_count} ({out_train_pct:.2f}%)\n'
             f'Test: skew={skew_test:.2f}, kurt={kurt_test:.2f}, outliers={out_test_count} ({out_test_pct:.2f}%)',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
             fontsize=9)
plt.show()


plt.figure(figsize=(6, 4))
sns.histplot(data=train, x="BeatsPerMinute", kde=True, bins=30, palette=['skyblue'], alpha=0.5)
plt.show()


plt.figure(figsize=(18, 12))
for i, col in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df_all, x='dataset', y=col, palette=['skyblue','salmon'])
    plt.title(f'Boxplot of {col}')
plt.show()


corr_train = train[features].corr()
corr_test = test[features].corr()

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

sns.heatmap(corr_train, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5, ax=axes[0])
axes[0].set_title("Train Feature Correlation")

sns.heatmap(corr_test, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5, ax=axes[1])
axes[1].set_title("Test Feature Correlation")

plt.show()


train.drop(columns=["dataset"], inplace=True)
test.drop(columns=["dataset"], inplace=True)


train['TrackDurationMs'] = np.log1p(train['TrackDurationMs'])
test['TrackDurationMs']  = np.log1p(test['TrackDurationMs'])


train['LivePerformanceLikelihood'] = np.sqrt(train['LivePerformanceLikelihood'])
test['LivePerformanceLikelihood']  = np.sqrt(test['LivePerformanceLikelihood'])


train['InstrumentalScore'] = np.sqrt(train['InstrumentalScore'])
test['InstrumentalScore']  = np.sqrt(test['InstrumentalScore'])


train['AcousticQuality'] = np.log1p(train['AcousticQuality'])
test['AcousticQuality']  = np.log1p(test['AcousticQuality'])


train['VocalContent'] = np.sqrt(train['VocalContent'])
test['VocalContent']  = np.sqrt(test['VocalContent'])


def apply_binning(train, test, column, labels, quantiles=None, bins=None, include_lowest=True, new_col=None):
    new_col = new_col or column + '_bin'
    
    if quantiles is not None:
        q_values = train[column].quantile(quantiles).values
        bins = [-np.inf] + q_values.tolist() + [np.inf]
    
    train[new_col] = pd.cut(train[column], bins=bins, labels=labels, include_lowest=include_lowest)
    test[new_col]  = pd.cut(test[column],  bins=bins, labels=labels, include_lowest=include_lowest)
    
    return train, test


train, test = apply_binning(train, test, 'Energy', 
                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], 
                            quantiles=[0.2, 0.4, 0.6, 0.8])

train, test = apply_binning(train, test, 'TrackDurationMs',
                            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'], 
                            quantiles=[0.2, 0.4, 0.6, 0.8])

train, test = apply_binning(train, test, 'MoodScore',
                            labels=['Very Sad', 'Sad', 'Neutral', 'Happy', 'Very Happy'], 
                            quantiles=[0.2, 0.4, 0.6, 0.8])

train, test = apply_binning(train, test, 'LivePerformanceLikelihood',
                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], 
                            quantiles=[0.2, 0.4, 0.6, 0.8])

train, test = apply_binning(train, test, 'InstrumentalScore',
                            labels=['Vocal', 'Mostly Vocal', 'Mixed', 'Instrumental'], 
                            quantiles=[0.3, 0.6, 0.9])

train, test = apply_binning(train, test, 'AcousticQuality',
                            labels=['Non-Acoustic', 'Low', 'Medium', 'Highly Acoustic'], 
                            quantiles=[0.3, 0.6, 0.9])

train, test = apply_binning(train, test, 'VocalContent',
                            labels=['Non-Vocal', 'Low', 'Medium', 'Highly Vocal'], 
                            quantiles=[0.3, 0.6, 0.9])


loudness_bins = [-30, -15, -10, -5, 0]
loudness_labels = ['Very Quiet', 'Quiet', 'Medium', 'Loud']
train, test = apply_binning(train, test, 'AudioLoudness', labels=loudness_labels, bins=loudness_bins)

rhythm_bins = [0, 0.4, 0.6, 0.8, 1.0]
rhythm_labels = ['Low Rhythm', 'Medium Rhythm', 'High Rhythm', 'Very High Rhythm']
train, test = apply_binning(train, test, 'RhythmScore', labels=rhythm_labels, bins=rhythm_bins)



# 1. Energy Intensity - Energy normalized by loudness
train['EnergyIntensity'] = train['Energy'] / (train['AudioLoudness'] * -1 + 1e-9)
test['EnergyIntensity']  = test['Energy']  / (test['AudioLoudness'] * -1 + 1e-9)

# 2. Rhythm-to-Energy Ratio - Ratio of rhythm to energy
train['RhythmEnergyRatio'] = train['RhythmScore'] / (train['Energy'] + 1e-9)
test['RhythmEnergyRatio']  = test['RhythmScore']  / (test['Energy'] + 1e-9)

# 3. Vocal Presence Score - Vocal probability adjusted by loudness
train['VocalPresence'] = train['VocalContent'] * (train['AudioLoudness'] * -1)
test['VocalPresence']  = test['VocalContent']  * (test['AudioLoudness'] * -1)

# 4. Acoustic-Organic Score - Acoustic quality × live performance likelihood
train['AcousticOrganicScore'] = train['AcousticQuality'] * train['LivePerformanceLikelihood']
test['AcousticOrganicScore']  = test['AcousticQuality']  * test['LivePerformanceLikelihood']

# 5. Focus/Background Score - Backgroundness indicator
train['FocusBackgroundScore'] = (1 - train['Energy']) * train['AcousticQuality']
test['FocusBackgroundScore']  = (1 - test['Energy'])  * test['AcousticQuality']


num_cols = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy',
    
    'EnergyIntensity',  'RhythmEnergyRatio', 'VocalPresence',
    'AcousticOrganicScore', 'FocusBackgroundScore'
]

descrete_cols  = [
    ...
]

cat_cols= [
    'Energy_bin','TrackDurationMs_bin', 'MoodScore_bin', 'LivePerformanceLikelihood_bin',
    'InstrumentalScore_bin', 'AcousticQuality_bin', 'VocalContent_bin',
    'AudioLoudness_bin', 'RhythmScore_bin'
]


X = train.drop(columns=["BeatsPerMinute"])
target = train["BeatsPerMinute"]


X_xgb = X.copy()
test_xgb = test.copy()


te_encoder = ce.TargetEncoder(smoothing=7, cols=cat_cols)

# --- XGB: TE ---
X_xgb[cat_cols] = te_encoder.fit_transform(X_xgb[cat_cols], target)
test_xgb[cat_cols] = te_encoder.transform(test_xgb[cat_cols])

# --- TE × num for XGB ---
for col in cat_cols:
    for num in num_cols:
        inter_col = f"{col}_x_{num}"
        X_xgb[inter_col] = X_xgb[col] * X_xgb[num]
        test_xgb[inter_col] = test_xgb[col] * test_xgb[num]

# --- CatBoost: TE  ---
X_te = te_encoder.transform(X[cat_cols])
test_te = te_encoder.transform(test[cat_cols])

for col in cat_cols:
    te_col = f"te_{col}"
    X[te_col] = X_te[col]
    test[te_col] = test_te[col]

    for num in num_cols:
        inter_col = f"{te_col}_x_{num}"
        X[inter_col] = X[te_col] * X[num]
        test[inter_col] = test[te_col] * test[num]



def objective_xgb(trial):
    max_depth=trial.suggest_int("max_depth", 2, 6)
    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    subsample=trial.suggest_float("subsample", 0.5, 1.0)
    # colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0)
    n_estimators=trial.suggest_int("n_estimators", 100, 500)

    model = XGBRegressor(
        tree_method="gpu_hist",
        random_state=SEED,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        # colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
    )

    scores = cross_val_score(
        model, 
        X_xgb, 
        target, 
        cv=3,  
        scoring="neg_mean_squared_error"  
    )

    rmse_scores = np.sqrt(-scores)  
    
    return rmse_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=50)


plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_contour(study, params=["n_estimators", "learning_rate"]).show()
plot_slice(study, params=["max_depth", "learning_rate", "n_estimators"]).show()


model_xgb = XGBRegressor(
    tree_method="gpu_hist",
    random_state=SEED,
    **study.best_trial.params
)


def objective_catboost(trial):
    depth = trial.suggest_int("depth", 2, 6)  
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 600)
    # l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        # l2_leaf_reg=l2_leaf_reg,
        loss_function="RMSE",
        random_seed=SEED,
        eval_metric="RMSE",
        verbose=0,
        task_type="GPU",
        cat_features=cat_cols
    )

    scores = cross_val_score(
        model,
        X, 
        target,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(-scores)

    return rmse_scores.mean()


study = optuna.create_study(direction="minimize")
study.optimize(objective_catboost, n_trials=50)

print("Best params:", study.best_trial.params)
print("Best RMSE:", study.best_value)



model_catboost = CatBoostRegressor(
    task_type="GPU",
    random_state=SEED,
    cat_features=cat_cols,
    **study.best_trial.params
)


NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_xgb = np.zeros(len(X_xgb))
oof_cb  = np.zeros(len(X))

for train_idx, val_idx in kf.split(X_xgb):
    # --- XGB ---
    X_tr, X_val = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    
for train_idx, val_idx in kf.split(X):
    # --- CatBoost ---
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model_catboost.fit(X_tr, y_tr, cat_features=cat_cols, verbose=0)
    oof_cb[val_idx] = model_catboost.predict(X_val)



stacked_train = pd.DataFrame({
    "xgb_pred": oof_xgb,
    "cb_pred": oof_cb
})

meta_model = LGBMRegressor(random_state=SEED)
meta_model.fit(stacked_train, target)


xgb_pred_test = model_xgb.fit(X_xgb, target).predict(test_xgb)
cb_pred_test  = model_catboost.fit(X, target, verbose=0).predict(test)

stacked_test = pd.DataFrame({
    "xgb_pred": xgb_pred_test,
    "cb_pred": cb_pred_test
})

final_pred = meta_model.predict(stacked_test)


submission = pd.DataFrame({
    'id': test.index,
    "BeatsPerMinute": final_pred
})


submission


submission.to_csv("submission.csv", index=False)

