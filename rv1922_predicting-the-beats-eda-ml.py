import pandas as pd
import numpy as np
import os 
import time 
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from scipy.special import logit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp
import plotly.figure_factory as ff  
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.head()


train.info()


train.describe()


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])


train.isnull().sum()


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)
print("-"*30)
print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


num_col = ['RhythmScore','MoodScore','TrackDurationMs','Energy', 'BeatsPerMinute']
num_col1 = ['AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood']


ncols = 2
nrows = math.ceil(len(num_col) / ncols)

fig = sp.make_subplots(
    rows=nrows,
    cols=ncols,
    subplot_titles=num_col
)

colors = px.colors.qualitative.Dark24

for i, col in enumerate(num_col):
    row = i // ncols + 1
    col_pos = i % ncols + 1
    fig.add_trace(
        go.Histogram(
            x=train[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=300 * nrows,   
    width=750,
    title_text='Distribution of Numerical Features',
    showlegend=False,
    template='simple_white'
)

fig.show()


ncols = 2
nrows = math.ceil(len(num_col1) / ncols)

fig = sp.make_subplots(
    rows=nrows,
    cols=ncols,
    subplot_titles=num_col1
)

colors = px.colors.qualitative.Dark24

for i, col in enumerate(num_col1):
    row = i // ncols + 1
    col_pos = i % ncols + 1
    fig.add_trace(
        go.Histogram(
            x=train[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=300 * nrows,   # scale height dynamically
    width=750,
    title_text='Distribution of Numerical Features',
    showlegend=False,
    template='simple_white'
)

fig.show()


corr = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Numeric Feature Correlations", fontsize=16)
plt.show()


train['TrackDurationMin'] = train['TrackDurationMs'] / 60000

epsilon = 1e-6
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['Vocal_Instrument_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + epsilon)

train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['RhythmEnergy'] = train['RhythmScore'] * train['Energy']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']
train['Vocal_Energy_Interaction'] = train['VocalContent'] * train['Energy']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']

train['Electronic_Proxy'] = (1 - train['AcousticQuality']) * train['Energy'] * train['RhythmScore']
train['Ambient_Proxy'] = train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])

train['Ballad_Proxy'] = train['VocalContent'] * train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])
train['Instrumental_Intensity_Proxy'] = train['InstrumentalScore'] * train['Energy'] * train['AudioLoudness']
train['Mood_Rhythm_Energy'] = train['MoodScore'] * train['RhythmScore'] * train['Energy']

train['Mood_Energy_Dissonance'] = train['MoodScore'] - train['Energy']
train['Rhythm_Loudness_Ratio'] = train['RhythmScore'] / (train['AudioLoudness'] + epsilon)


test['TrackDurationMin'] = test['TrackDurationMs'] / 60000

epsilon = 1e-6
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + epsilon)

test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['RhythmEnergy'] = test['RhythmScore'] * test['Energy']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']
test['Vocal_Energy_Interaction'] = test['VocalContent'] * test['Energy']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']

test['Electronic_Proxy'] = (1 - test['AcousticQuality']) * test['Energy'] * test['RhythmScore']
test['Ambient_Proxy'] = test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])

test['Ballad_Proxy'] = test['VocalContent'] * test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])
test['Instrumental_Intensity_Proxy'] = test['InstrumentalScore'] * test['Energy'] * test['AudioLoudness']
test['Mood_Rhythm_Energy'] = test['MoodScore'] * test['RhythmScore'] * test['Energy']

test['Mood_Energy_Dissonance'] = test['MoodScore'] - test['Energy']
test['Rhythm_Loudness_Ratio'] = test['RhythmScore'] / (test['AudioLoudness'] + epsilon)


train.head()


X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42)


def objective(trial):
    param = {
        "device": "cuda",
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        
        # --- Parameter Suggestions ---
        
        # Use a fixed, large number of estimators and let early stopping find the optimal number.
        # This is more efficient than tuning n_estimators directly.
        "n_estimators": 2000, 
        
        # Focus on a more common and effective range for the learning rate.
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        
        # Your original range is good. A slightly smaller upper bound can prevent overfitting and speed up trials.
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        
        # Your original ranges for these are excellent for regularization. No changes needed.
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        
        # Widen the search space for regularization slightly.
        "lambda": trial.suggest_float("lambda", 1e-4, 100.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        
        # ADD THIS: min_child_weight is a great parameter for controlling model complexity.
        # It's the minimum sum of instance weight needed in a child.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = xgb.XGBRegressor(**param)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50, # Keep this, it's very important!
            verbose=False
        )
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=100)


#trial = study.best_trial
#print(f"Best RMSE: {trial.value}")
#print("Best Params:")
#for key, value in trial.params.items():
#    print(f"  {key}: {value}")


best_params = {
  'eta': 0.01888559460164984,
  'max_depth': 6,
  'subsample': 0.725740978593917,
  'colsample_bytree': 0.7937168801440118,
  'lambda': 67.57288689124256,
  'alpha': 4.759611913770927,
  'min_child_weight': 5,
  'objective': 'reg:squarederror',
  'eval_metric': 'rmse',
  'device': 'cuda',
  # Set a high n_estimators; we'll use early stopping during the final training.
  'n_estimators': 2000 
}


kf = KFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
valid_preds = np.zeros(len(X))
rmse_scores = []
feature_importances = pd.DataFrame(index=X.columns) # Create a dataframe to store importances

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"--- Training fold {fold+1} ---")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Store feature importances for this fold
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_
    
    # Validation preds
    valid_preds[valid_idx] = model.predict(X_val)
    
    # Test preds (accumulate for averaging later)
    test_preds += model.predict(X_test) / kf.n_splits
    
    # RMSE for this fold
    rmse = mean_squared_error(y_val, valid_preds[valid_idx], squared=False)
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nMean CV RMSE: {np.mean(rmse_scores):.4f} Â± {np.std(rmse_scores):.4f}")


# Calculate mean importance across folds
feature_importances['mean'] = feature_importances.mean(axis=1)

# Sort by mean importance
feature_importances = feature_importances.sort_values(by='mean', ascending=False)

# Display the top 20 features
print("\nTop 20 Feature Importances (averaged over 5 folds):")
print(feature_importances['mean'].head(26))


submission = pd.DataFrame({
    "id": submission["id"],
    "BeatsPerMinute": test_preds
})
submission.to_csv("xgb_kfold_predictions.csv", index=False)
print("âœ… Saved predictions to xgb_kfold_predictions.csv")

