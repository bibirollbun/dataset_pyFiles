
import os, sys, gc, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# LightGBM 
try:
    from lightgbm import LGBMRegressor, early_stopping
    LGBM_AVAILABLE = True
except Exception as e:
    print("LightGBM not available:", e)
    LGBM_AVAILABLE = False




# Paths
DATA_DIR = "/kaggle/input/playground-series-s5e9" if os.path.exists("/kaggle/input") else "."

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# If running locally, place train.csv and test.csv in working dir.
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train.shape, test.shape)
display(train.head())



import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns

# Define color palette
palette = sns.color_palette("husl", len(numerical_cols))

#to show Outliers in data 
plt.figure(figsize=(25, 18))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=train[col], color=palette[i-1])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# Define color palette
#palette = sns.color_palette("husl", len(numerical_cols))

#to show Distribution
#plt.figure(figsize=(25, 15))
#for i, col in enumerate(numerical_cols, 1):
#    plt.subplot(4, 4, i)
#    sns.histplot(train[col], kde=True, color=palette[i-1], bins=30)
#    plt.title(f'Distribution of {col}')
#plt.tight_layout()
#plt.show()



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Duration features
    df["TrackDurationMin"] = df["TrackDurationMs"] / 60000.0
    df["TrackDurationSec"] = df["TrackDurationMs"] / 1000.0
    df["LogDuration"] = np.log1p(df["TrackDurationMs"])

    # Interaction features
    df["Rhythm_Energy"] = df["RhythmScore"] * df["Energy"]
    df["Acoustic_Vocal"] = df["AcousticQuality"] * df["VocalContent"]
    df["Mood_Live"] = df["MoodScore"] * df["LivePerformanceLikelihood"]

    # Pairwise ratios (safe division)
    eps = 1e-6
    df["Energy_over_Rhythm"] = df["Energy"] / (df["RhythmScore"] + eps)
    df["Vocal_over_Acoustic"] = df["VocalContent"] / (df["AcousticQuality"] + eps)

    # Polynomial terms
    for col in ["RhythmScore","Energy","MoodScore","AudioLoudness","AcousticQuality","VocalContent"]:
        df[f"{col}_sq"] = df[col] ** 2

    # Binning duration
    df["DurationBin"] = pd.qcut(df["TrackDurationMin"], q=10, duplicates='drop').cat.codes

    return df

train_fe = add_features(train)
test_fe = add_features(test)

print("Train columns:", len(train_fe.columns))
display(train_fe.head())




TARGET = "BeatsPerMinute"
ID = "id"

features = [c for c in train_fe.columns if c not in [TARGET, ID]]
X = train_fe[features]
y = train_fe[TARGET]
X_test = test_fe[features]

X.shape, X_test.shape



# different model from LGBM

#models = [
#    LGBMRegressor(                         # Model 1
#        num_leaves=25, learning_rate=0.005, n_estimators=550,
#        max_depth=30, min_child_samples=15,
#        subsample=1.0, colsample_bytree=1.0,
#        reg_alpha=1.0, reg_lambda=0.5,
#        random_state=42, n_jobs=-1
#    ),
#    LGBMRegressor(                         # Model 2
#        num_leaves=38, learning_rate=0.009, n_estimators=800,
#        max_depth=35, min_child_samples=20,
#        subsample=0.8, colsample_bytree=0.8,
#        reg_alpha=0.5, reg_lambda=0.5,
#        random_state=52, n_jobs=-1
#    ),
#    LGBMRegressor(                         # Model 3
#        num_leaves=30, learning_rate=0.004, n_estimators=650,
#        max_depth=25, min_child_samples=10,
#        subsample=0.9, colsample_bytree=0.9,
#        reg_alpha=1.0, reg_lambda=1.0,
#        random_state=99, n_jobs=-1
#    )
#]
#
# trainig & evalute
#oof_preds = []
#test_preds = []

#for i, model in enumerate(models, 1):
#    model.fit(X, y)
#    pred_train = model.predict(X)
#    pred_test = model.predict(X_test)
#
#    rmse = mean_squared_error(y, pred_train, squared=False)
#    print(f"Model {i} RMSE: {rmse:.6f}")

#    oof_preds.append(pred_train)
#    test_preds.append(pred_test)

 #average ensemble
#oof_blend = np.mean(oof_preds, axis=0)
#test_blend = np.mean(test_preds, axis=0)

#rmse_blend = mean_squared_error(y, oof_blend, squared=False)
#print(f"Ensemble Blend RMSE: {rmse_blend:.6f}")

# final result
#final_preds = test_blend



from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Define Models (only 2)
models = [
    {
        'name': 'LGBM_Fast1',
        'model': LGBMRegressor(
            num_leaves=33, learning_rate=0.03, n_estimators=300,
            max_depth=10, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1, verbose=-1, boosting_type='gbdt'
        )
    },
    {
        'name': 'LGBM_Fast2',
        'model': LGBMRegressor(
            num_leaves=28, learning_rate=0.02, n_estimators=250,
            max_depth=12, min_child_samples=15, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
            random_state=52, n_jobs=-1, verbose=-1, boosting_type='gbdt'
        )
    }
]


# Training Function
def train_and_evaluate_models_fast(X, y, X_test, n_splits=3):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), len(models)))
    cv_scores = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for model_idx, model_config in enumerate(models):
        model = model_config['model']
        model_name = model_config['name']
        
        print(f"\nTraining {model_name}...")
        print("=" * 30)
        
        # Cross-validation
        cv_rmse = np.sqrt(-cross_val_score(
            model, X, y, scoring='neg_mean_squared_error',
            cv=kf, n_jobs=1
        ))
        
        print(f"CV RMSE Scores: {[f'{x:.6f}' for x in cv_rmse]}")
        print(f"Mean CV RMSE: {cv_rmse.mean():.6f} (Â±{cv_rmse.std():.6f})")
        cv_scores.append(cv_rmse.mean())
        
        # Train full model
        model.fit(X, y)
        
        # Predictions
        pred_train = model.predict(X)
        pred_test = model.predict(X_test)
        
        oof_preds += pred_train / len(models)
        test_preds[:, model_idx] = pred_test
        
        # Training RMSE
        train_rmse = mean_squared_error(y, pred_train, squared=False)
        print(f"Training RMSE: {train_rmse:.6f}")
    
    return oof_preds, test_preds, cv_scores


# Run Training
oof_blend, test_preds_matrix, cv_scores = train_and_evaluate_models_fast(X, y, X_test)

# Simple Ensemble (average)
test_blend = np.mean(test_preds_matrix, axis=1)

# Weighted Ensemble (better)
weights = 1 / np.array(cv_scores)
weights = weights / weights.sum()
test_weighted_blend = np.average(test_preds_matrix, axis=1, weights=weights)

# Final predictions
final_preds = test_weighted_blend  # or test_blend


# Save Submission
ID = "id"  
TARGET = "BeatsPerMinute"  

sub = pd.DataFrame({
    ID: test[ID],
    TARGET: final_preds
})

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"\nâœ… Saved submission file to: {sub_path}")
print(sub.head())


