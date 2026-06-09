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


X_train = train_df.drop("BeatsPerMinute", axis=1)
y_train = train_df["BeatsPerMinute"]
X_test = test_df.copy()


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
    # Measures the balance between vocal and instrumental~ elements.
    df["Vocal_Instrument_Balance"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"] = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"] = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"] = df["RhythmScore"] * df["Energy"]

    return df

train_df = fe1(train_df)
test_df = fe1(test_df)


def fe2(df: pd.DataFrame) -> pd.DataFrame:
    """For each column, excluding the target "BeatsPerMinute", rounds to 8 decimal places"""
    FEATURES = [col for col in df.columns if col != "BeatsPerMinute"]
    for c in FEATURES:
        n = f"{c}_r{8}"
        df[n] = df[c].round(8)
            
    return df

train_df = fe2(train_df)
test_df = fe2(test_df)


from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor

#Hypertuned params
xgb_params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.00284886312081176,
    "max_depth": 6,                    
    "subsample": 0.5840802415554563,
    "colsample_bytree": 0.6928401363009189,
    "device": "cuda",
    #new ones below
    "min_child_weight":10,
    'reg_lambda': 0.49133739895509515,
    'reg_alpha': 0.8220375350636525
    
}

lgbm_params = {
    "objective": "regression",   
    "metric": "rmse",
    "device": "gpu",
    "learning_rate": 0.026827678729981763,
    "max_depth": 5,
    "num_leaves": 194,
    "subsample": 0.6307028685963693,
    "colsample_bytree": 0.5684607760413074,
    #new ones below
    "min_child_samples":79,
    'reg_lambda': 0.039745131271073486,
    'reg_alpha': 0.010956300085694048
    
}

model1 = XGBRegressor(n_estimators=10_000,
            **xgb_params)
model2 = LGBMRegressor(
    n_estimators=1000, **lgbm_params)


# Prepare out-of-fold predictions for stacking
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds1 = np.zeros(len(X_train))
oof_preds2 = np.zeros(len(X_train))
test_preds1 = np.zeros((len(X_test), kf.n_splits)) #this is a 2-d array which we didn't do before! Averaged predictions
test_preds2 = np.zeros((len(X_test), kf.n_splits))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Fold {fold+1}")
#The bit below is different - understand what's done in og and then here
    #WARNING: we must do .copy() below so we don't edit the original (which would ruin next fold)
    X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[valid_idx].copy() 
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    # Target encoding only on training fold, then transform val and test
    #I want to target encode all of the features
    FEATURES = [c for c in train_df.columns if c in X_train.columns and c != "BeatsPerMinute"]

    #s5e9
    X_test_enc = X_test.copy()
    
    CC = FEATURES #what is the point of this?
    for i,c in enumerate(CC): #index, element
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean') #note that you should look at cuml api for parameter descriptions (not scikit learn!)
        X_tr[n] = TE0.fit_transform(X_tr[c],y_tr).astype('float32') #I think here we add new, target-encoded features to og features
        X_val[n] = TE0.transform(X_val[c]).astype('float32')
        X_test_enc[n] = TE0.transform(X_test[c]).astype('float32')
    

    # Fit base models (fit vs .train hmmm)
    model1.fit(X_tr, y_tr)
    model2.fit(X_tr, y_tr)

    # Store out-of-fold preds as np array? val index are distinct for each fold
    oof_preds1[valid_idx] = model1.predict(X_val)
    oof_preds2[valid_idx] = model2.predict(X_val)

    # Store test preds for this fold
    test_preds1[:, fold] = model1.predict(X_test_enc)
    test_preds2[:, fold] = model2.predict(X_test_enc)


# Stack the OOF predictions
train_stack = np.column_stack((oof_preds1, oof_preds2)) #does this create an array with 2 columns then? which are like a feature!
test_stack = np.column_stack((test_preds1.mean(axis=1), test_preds2.mean(axis=1)))


from sklearn.ensemble import RandomForestRegressor

meta_model = RandomForestRegressor(
    random_state=42, 
    n_estimators=50, 
    max_depth=10,
    n_jobs=-1
)

cv_rmse_scores = -1 * cross_val_score(
    meta_model, train_stack, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)


print("Blended RMSE scores:", cv_rmse_scores)
print("Blended mean RMSE score:", cv_rmse_scores.mean())


meta_model.fit(train_stack, y_train)
final_predictions = meta_model.predict(test_stack)


sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
sub.BeatsPerMinute = final_predictions
sub.to_csv("submission.csv", index=False)
sub.head()




