# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical functions
from scipy.stats import skew
from scipy.stats import ttest_rel
from scipy.signal import find_peaks

# Machine learning preprocessing and modeling
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from cuml.preprocessing import TargetEncoder

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Read the data
train_df = pd.read_csv('../input/playground-series-s5e9/train.csv', index_col='id')
test_df = pd.read_csv('../input/playground-series-s5e9/test.csv', index_col='id')

# Verify shapes
print("Train Data Shape:", train_df.shape)
print("\nTest Data Shape:", test_df.shape)

# Remove rows with missing target, separate target from predictors
train_df.dropna(axis=0, subset=['BeatsPerMinute'], inplace=True) #inplace = True means that we remove it from X


#X_train = train_df.drop("BeatsPerMinute", axis=1)
#y_train = train_df["BeatsPerMinute"]


print(train_df.columns.values)


train_df.head()


train_df.describe()


train_df.info()


# Missing values
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("✅ No missing values found.")


# Duplicates
print("Duplicate rows:", train_df.duplicated().sum())


X_train_percentage = train_df.drop(["AudioLoudness", "TrackDurationMs", "BeatsPerMinute"], axis = 1)
X_train_percentage.boxplot(figsize=(10,6))
plt.xticks(rotation=45)  # rotate labels if column names are long
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(12, 5))  # 1 row, 2 columns

train_df["TrackDurationMs"].plot.box(ax=axes[0], title="Track Duration (ms)")
train_df["AudioLoudness"].plot.box(ax=axes[1], title="Audio Loudness")
train_df["BeatsPerMinute"].plot.box(ax=axes[2], title="BPM")

plt.tight_layout()
plt.show()


# ========================
# 3. Target Variable Analysis (Target: BeatsPerMinute)
# ========================
target = "BeatsPerMinute"

plt.figure(figsize=(8,5))
sns.histplot(train_df[target], kde=True, bins=30)
plt.title(f"Distribution of {target}")
plt.show()


# ========================
# 4. Feature Distributions
# ========================
#All of them are num_features but whatevs
num_features = train_df.select_dtypes(include=[np.number]).columns.tolist() 
#num_features.remove("id") # remove id if present

# Plot histograms for all numeric features
train_df[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


# Compute correlation matrix
corr = train_df[num_features].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask applied
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center = 0, #so the middle thing is gray
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap")
plt.show()

# Correlation with target
print("\n--- Correlation with Target ---")
print(train_df.corr()[target].sort_values(ascending=False))



from sklearn.feature_selection import mutual_info_regression
X_train = train_df.drop("BeatsPerMinute", axis=1)
y_train = train_df["BeatsPerMinute"]

mi_scores = mutual_info_regression(X_train, y_train)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


sns.relplot(x="VocalContent", y="BeatsPerMinute", data=train_df);


cv = KFold(n_splits=5, shuffle=True, random_state=42)


from sklearn.dummy import DummyRegressor
X_train = train_df.drop("BeatsPerMinute", axis=1)
y_train = train_df["BeatsPerMinute"]

init_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', DummyRegressor(strategy='mean'))
])

# Multiply by -1 since sklearn calculates *negative* RMSE
dummy_rmse_scores = -1 * cross_val_score(init_pipeline, X_train, y_train,
                              cv=cv,
                              scoring='neg_root_mean_squared_error')

print("DummyRegressor RMSE scores:\n", dummy_rmse_scores)
print("DummyRegressor Mean RMSE scores:\n", dummy_rmse_scores.mean())


from xgboost import XGBRegressor
X_train = train_df.drop("BeatsPerMinute", axis=1)
y_train = train_df["BeatsPerMinute"]

init_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', XGBRegressor(random_state = 0))])


xgb_rmse_scores = -1 * cross_val_score(init_pipeline, X_train, y_train,
                              cv=cv,
                              scoring='neg_root_mean_squared_error')

print("XGBRegressor RMSE scores:\n", xgb_rmse_scores)
print("XGBRegressor Mean RMSE scores:\n", xgb_rmse_scores.mean())




from lightgbm import LGBMRegressor
X_train = train_df.drop("BeatsPerMinute", axis=1)
y_train = train_df["BeatsPerMinute"]

init_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', LGBMRegressor(random_state = 0, verbose = -1))])


lgbm_rmse_scores = -1 * cross_val_score(init_pipeline, X_train, y_train,
                              cv=cv,
                              scoring='neg_root_mean_squared_error')

print("LGBMRegressor RMSE scores:\n", lgbm_rmse_scores)
print("LGBMRegressor Mean RMSE scores:\n", lgbm_rmse_scores.mean())


#easier to make this function and then pass both train_df and test_df through it, rather than doing each change for both train_df and test_df!!
def fe1(df: pd.DataFrame) -> pd.DataFrame:
    df["Energy_AudioLoudness"] = df["Energy"] * df["AudioLoudness"]
    df["Mood_Acoustic"] = df["MoodScore"] * df["AcousticQuality"]
    #df["is_high_energy"] = (df["Energy"] > 0.7).astype(int) -> When I added these features my model actually did worse!!! I think XGBoost models do well at creating thresholds anyways...
    #df["is_acoustic"] = (df["AcousticQuality"] > 0.5).astype(int)
    #df["is_live"] = (df["LivePerformanceLikelihood"] > 0.5).astype(int)
    # The track duration expressed in minutes (converted from milliseconds).
    df["TrackDurationMin"] = df["TrackDurationMs"] / 60000
    # The ratio of overall energy to acoustic quality.
    df["Energy_Acoustic_Ratio"] = df["Energy"] / (df["AcousticQuality"] + 1e-5)
    # Measures the balance between vocal and instrumental elements.
    df["Vocal_Instrument_Balance"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"] = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"] = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"] = df["RhythmScore"] * df["Energy"]

    return df

train_df = fe1(train_df)
test_df = fe1(test_df)
train_df.head()


# Simple one-liner for each numeric column
for col in train_df.select_dtypes(include=['float']).columns:
    max_dec = train_df[col].astype(str).str.split('.').str[1].str.len().max()
    print(f"{col}: {max_dec} decimal places")


def fe2(df: pd.DataFrame) -> pd.DataFrame:
    """For each column, excluding the target "BeatsPerMinute", rounds to 8 decimal places"""
    FEATURES = [col for col in df.columns if col != "BeatsPerMinute"]
    for c in FEATURES:
        n = f"{c}_r{8}"
        df[n] = df[c].round(8)
            
    return df

train_df = fe2(train_df)
test_df = fe2(test_df)
train_df.head()


import lightgbm as lgb
FOLDS = 5
SEED = 42

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)
    
    FEATURES = list(train_df.columns)
    FEATURES.remove("BeatsPerMinute")
    X_train = train_df.iloc[train_idx][FEATURES].copy()
    y_train = train_df.iloc[train_idx]["BeatsPerMinute"]
    
    X_valid = train_df.iloc[val_idx][FEATURES].copy()
    y_valid = train_df.iloc[val_idx]["BeatsPerMinute"]
    X_test = test_df[FEATURES].copy()

    # target encoding
    CC = FEATURES
    print(f"Target encoding {len(CC)} features... ", end="")
    for i, c in enumerate(CC):
        if i % 5 == 0:
            print(f"{i}, ", end="")
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
        X_train[n] = TE0.fit_transform(X_train[c], y_train).astype('float32')
        X_valid[n] = TE0.transform(X_valid[c]).astype('float32')
        X_test[n] = TE0.transform(X_test[c]).astype('float32')
    print()

    # create LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    # LightGBM parameters (translated from your XGBoost setup)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "seed": SEED
    }

    # train LightGBM model
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10_000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )

    oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / FOLDS



m = np.sqrt( np.mean( (oof_preds - train_df["BeatsPerMinute"].values)**2. ) )
print(f" Overall CV RMSE = {m}")


FOLDS = 5
SEED = 42

#Hypertuned params
params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.0013060824237770417,
    "max_depth": 9,                    
    "subsample": 0.7359374953084611,
    "colsample_bytree": 0.5976713841429055,
    "seed": SEED,
    "device": "cuda",
    #new ones below
    "min_child_weight":4,
    'reg_lambda': 1.746976232994707,
    'reg_alpha': 0.01068937980102936
    
}


import xgboost as xgb

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)
    #copy required because we will be doing target encoding on features but we don't want it to edit the original...
    FEATURES = list(train_df.columns)
    FEATURES.remove("BeatsPerMinute")
    X_train = train_df.iloc[train_idx][FEATURES].copy() #pop (edits og dataset?) remove(), drop()
    y_train = train_df.iloc[train_idx]["BeatsPerMinute"]
    
    X_valid = train_df.iloc[val_idx][FEATURES].copy()
    y_valid = train_df.iloc[val_idx]["BeatsPerMinute"]
    X_test = test_df[FEATURES].copy() #will edit this guy by also doing target encoding I suppose


    #I want to target encode all the features
    CC = FEATURES
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%5==0: print(f"{i}, ",end="")
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
        X_train[n] = TE0.fit_transform(X_train[c],y_train).astype('float32')
        X_valid[n] = TE0.transform(X_valid[c]).astype('float32')
        X_test[n] = TE0.transform(X_test[c]).astype('float32')            
    print() #makes a new line

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, enable_categorical=False)

    model = xgb.train(
        params=params, #want to find optimal parameters by doing hyperparameter tuning? - can we do randomized search cv? weird because supervised
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


m = np.sqrt( np.mean( (oof_preds - train_df["BeatsPerMinute"].values)**2. ) )
print(f" Overall CV RMSE = {m}")


fig, ax = plt.subplots(figsize=(10, 25))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


"""In the following code I:
    - create TargetEncoder based on all of the training data
    - create an XGBoost model for each fold 
        (the model is different because we use a different random_state/seed
        the validation data due to cross-validation fold is only used for logging cv rmse)
    - Note that I don't use a pipeline so I have to 'manually' code the cross validation for loop"""

test_preds_full = np.zeros(len(test_df))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    FEATURES = list(train_df.columns)
    FEATURES.remove("BeatsPerMinute")
    # WE NOW USE 100% TRAIN HERE
    X_train = train_df[FEATURES].copy()
    y_train = train_df["BeatsPerMinute"]
    
    X_valid = train_df.iloc[val_idx][FEATURES].copy()
    y_valid = train_df.iloc[val_idx]["BeatsPerMinute"]
    X_test = test_df[FEATURES].copy()

    CC = FEATURES
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%5==0: print(f"{i}, ",end="")
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
        X_train[n] = TE0.fit_transform(X_train[c],y_train).astype('float32')
        X_valid[n] = TE0.transform(X_valid[c]).astype('float32')
        X_test[n] = TE0.transform(X_test[c]).astype('float32')            
    print()

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, enable_categorical=False)

    # WE CHANGE SEED EACH FOLD - see under cvscore for explanation
    params['seed'] = fold
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1739,
        evals=[(dtrain, "train"), (dval, "valid")],
        verbose_eval=200 #logs every 200 rounds
    )

    test_preds_full += model.predict(dtest) / FOLDS


sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
sub.BeatsPerMinute = test_preds_full
sub.to_csv("submission_refit_full.csv",index=False)
sub.head()

