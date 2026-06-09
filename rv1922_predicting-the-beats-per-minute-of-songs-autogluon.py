%%time
!pip install autogluon==1.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import sklearn
print("AutoGluon ready, sklearn version:", sklearn.__version__)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


train = pd.concat([train, original], axis=0, ignore_index=True)
tran = train.drop_duplicates().reset_index(drop=True)


train.head()


epsilon = 1e-6

# --- Base conversions ---
train['TrackDurationMin'] = train['TrackDurationMs'] / 60000

# --- Ratios & balances ---
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['Vocal_Instrument_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + epsilon)
train['Vocal_share'] = train['VocalContent'] / (train['VocalContent'] + train['InstrumentalScore'] + epsilon)
train['Instrumental_to_Total'] = train['InstrumentalScore'] / (train['VocalContent'] + train['InstrumentalScore'] + epsilon)
train['Acoustic_to_Energy'] = train['AcousticQuality'] / (train['Energy'] + epsilon)

# --- Multiplicative interactions ---
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

# --- Differences ---
train['Energy_Mood_diff'] = train['Energy'] - train['MoodScore']
train['Energy_Mood_absdiff'] = np.abs(train['Energy_Mood_diff'])
train['Rhythm_Instrument_diff'] = train['RhythmScore'] - train['InstrumentalScore']
train['Vocal_Instrument_diff'] = train['VocalContent'] - train['InstrumentalScore']

# --- Log / nonlinear transforms ---
train['TrackDurationMin_log'] = np.log1p(train['TrackDurationMin'])
train['Rhythm_sqrt'] = np.sqrt(train['RhythmScore'])
train['AudioLoudness_sq'] = train['AudioLoudness'] ** 2
train['Energy_cbrt'] = np.cbrt(train['Energy'])
train['log_Rhythm_over_Acoustic'] = np.log1p(train['RhythmScore'] / (train['AcousticQuality'] + epsilon))
train['log_Instrumental_Energy'] = np.log1p(train['InstrumentalScore'] * train['Energy'])

# --- Composite interactions ---
train['Rhythm_Energy_Loudness'] = train['RhythmScore'] * train['Energy'] * (train['AudioLoudness'] + 20)
train['Instrumental_Energy'] = train['InstrumentalScore'] * train['Energy']
train['Vocal_Rhythm_Energy'] = train['VocalContent'] * train['RhythmScore'] * train['Energy']
train['RhythmEnergy_log'] = np.log1p(train['RhythmScore'] * train['Energy'])
train['RhythmEnergy_over_Instrument'] = (train['RhythmScore'] * train['Energy']) / (train['InstrumentalScore'] + epsilon)

# --- Proxies with sigmoid (soft categories) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))

x = 3*(train['Energy'] - 0.5) + 2*(train['RhythmScore'] - 0.5) + 0.2*(train['AudioLoudness'] + 8)
train['Dance_Proxy'] = sigmoid(x)

y = 4*(train['VocalContent']) + 3*(train['AcousticQuality']) - 5*(train['Energy']) - 4*(train['RhythmScore'])
train['Ballad_Proxy_Score'] = sigmoid(y)

# --- Relative rankings ---
train['Rhythm_rank'] = train['RhythmScore'].rank(pct=True)
train['Energy_rank'] = train['Energy'].rank(pct=True)
train['RhythmEnergy_rank_prod'] = train['Rhythm_rank'] * train['Energy_rank']

# --- Ratios with loudness ---
train['Rhythm_Loudness_Ratio'] = train['RhythmScore'] / (train['AudioLoudness'] + epsilon)
train['Rhythm_Loudness_Ratio_clipped'] = np.clip(train['RhythmScore'] / (train['AudioLoudness'] + 20 + epsilon), -100, 100)

# --- Duration transforms ---
train['TrackDurationMin_inv'] = 1 / (train['TrackDurationMin'] + epsilon)

# --- NEW: Statistical moment features ---
# Calculate rolling statistics for key features
for col in ['Energy', 'RhythmScore', 'AudioLoudness', 'MoodScore']:
    train[f'{col}_zscore'] = (train[col] - train[col].mean()) / train[col].std()
    train[f'{col}_skewness'] = (train[col] - train[col].mean())**3 / train[col].std()**3

# --- NEW: Polynomial features ---
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    train[f'{col}_squared'] = train[col] ** 2
    train[f'{col}_cubed'] = train[col] ** 3

# --- NEW: Audio properties combinations ---
train['Loudness_Energy_Ratio'] = train['AudioLoudness'] / (train['Energy'] + epsilon)
train['Dynamic_Range_Proxy'] = train['AudioLoudness'].max() - train['AudioLoudness'].min()  # This might need adjustment

# --- NEW: Genre-like proxies based on statistics ---
train['Mellow_Proxy'] = train['AcousticQuality'] * train['MoodScore'] * (1 - train['Energy'])

# --- NEW: Vocal prominence score ---
train['Vocal_Prominence'] = train['VocalContent'] * (1 - train['InstrumentalScore']) * train['AudioLoudness']

# --- NEW: Performance quality indicators ---
train['Live_Energy_Balance'] = train['LivePerformanceLikelihood'] * train['Energy']
train['Studio_Polish_Proxy'] = (1 - train['LivePerformanceLikelihood']) * train['AudioLoudness']

# --- NEW: Complex interaction terms ---
train['Full_Production_Score'] = (
    train['Energy'] * train['AudioLoudness'] * 
    (1 - train['AcousticQuality']) * train['RhythmScore']
)


epsilon = 1e-6

# --- Base conversions ---
test['TrackDurationMin'] = test['TrackDurationMs'] / 60000

# --- Ratios & balances ---
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + epsilon)
test['Vocal_share'] = test['VocalContent'] / (test['VocalContent'] + test['InstrumentalScore'] + epsilon)
test['Instrumental_to_Total'] = test['InstrumentalScore'] / (test['VocalContent'] + test['InstrumentalScore'] + epsilon)
test['Acoustic_to_Energy'] = test['AcousticQuality'] / (test['Energy'] + epsilon)

# --- Multiplicative interactions ---
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

# --- Differences ---
test['Energy_Mood_diff'] = test['Energy'] - test['MoodScore']
test['Energy_Mood_absdiff'] = np.abs(test['Energy_Mood_diff'])
test['Rhythm_Instrument_diff'] = test['RhythmScore'] - test['InstrumentalScore']
test['Vocal_Instrument_diff'] = test['VocalContent'] - test['InstrumentalScore']

# --- Log / nonlinear transforms ---
test['TrackDurationMin_log'] = np.log1p(test['TrackDurationMin'])
test['Rhythm_sqrt'] = np.sqrt(test['RhythmScore'])
test['AudioLoudness_sq'] = test['AudioLoudness'] ** 2
test['Energy_cbrt'] = np.cbrt(test['Energy'])
test['log_Rhythm_over_Acoustic'] = np.log1p(test['RhythmScore'] / (test['AcousticQuality'] + epsilon))
test['log_Instrumental_Energy'] = np.log1p(test['InstrumentalScore'] * test['Energy'])

# --- Composite interactions ---
test['Rhythm_Energy_Loudness'] = test['RhythmScore'] * test['Energy'] * (test['AudioLoudness'] + 20)
test['Instrumental_Energy'] = test['InstrumentalScore'] * test['Energy']
test['Vocal_Rhythm_Energy'] = test['VocalContent'] * test['RhythmScore'] * test['Energy']
test['RhythmEnergy_log'] = np.log1p(test['RhythmScore'] * test['Energy'])
test['RhythmEnergy_over_Instrument'] = (test['RhythmScore'] * test['Energy']) / (test['InstrumentalScore'] + epsilon)

# --- Proxies with sigmoid (soft categories) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))

x = 3*(test['Energy'] - 0.5) + 2*(test['RhythmScore'] - 0.5) + 0.2*(test['AudioLoudness'] + 8)
test['Dance_Proxy'] = sigmoid(x)

y = 4*(test['VocalContent']) + 3*(test['AcousticQuality']) - 5*(test['Energy']) - 4*(test['RhythmScore'])
test['Ballad_Proxy_Score'] = sigmoid(y)

# --- Relative rankings ---
test['Rhythm_rank'] = test['RhythmScore'].rank(pct=True)
test['Energy_rank'] = test['Energy'].rank(pct=True)
test['RhythmEnergy_rank_prod'] = test['Rhythm_rank'] * test['Energy_rank']

# --- Ratios with loudness ---
test['Rhythm_Loudness_Ratio'] = test['RhythmScore'] / (test['AudioLoudness'] + epsilon)
test['Rhythm_Loudness_Ratio_clipped'] = np.clip(test['RhythmScore'] / (test['AudioLoudness'] + 20 + epsilon), -100, 100)

# --- Duration transforms ---
test['TrackDurationMin_inv'] = 1 / (test['TrackDurationMin'] + epsilon)

# --- NEW: Statistical moment features ---
# Calculate rolling statistics for key features
for col in ['Energy', 'RhythmScore', 'AudioLoudness', 'MoodScore']:
    test[f'{col}_zscore'] = (test[col] - test[col].mean()) / test[col].std()
    test[f'{col}_skewness'] = (test[col] - test[col].mean())**3 / test[col].std()**3

# --- NEW: Polynomial features ---
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    test[f'{col}_squared'] = test[col] ** 2
    test[f'{col}_cubed'] = test[col] ** 3

# --- NEW: Audio properties combinations ---
test['Loudness_Energy_Ratio'] = test['AudioLoudness'] / (test['Energy'] + epsilon)
test['Dynamic_Range_Proxy'] = test['AudioLoudness'].max() - test['AudioLoudness'].min()  # This might need adjustment

# --- NEW: Genre-like proxies based on statistics ---
test['Mellow_Proxy'] = test['AcousticQuality'] * test['MoodScore'] * (1 - test['Energy'])

# --- NEW: Vocal prominence score ---
test['Vocal_Prominence'] = test['VocalContent'] * (1 - test['InstrumentalScore']) * test['AudioLoudness']

# --- NEW: Performance quality indicators ---
test['Live_Energy_Balance'] = test['LivePerformanceLikelihood'] * test['Energy']
test['Studio_Polish_Proxy'] = (1 - test['LivePerformanceLikelihood']) * test['AudioLoudness']

# --- NEW: Complex interaction terms ---
test['Full_Production_Score'] = (
    test['Energy'] * test['AudioLoudness'] * 
    (1 - test['AcousticQuality']) * test['RhythmScore']
)


train.head()


target = 'BeatsPerMinute'

predictors = [col for col in train.columns if col not in ['id', target]]
eval_metric = 'rmse'
save_path = 'saved_models'
time_limit =  20000


fit_auto = TabularPredictor(
    label=target,
    problem_type='regression',
    eval_metric=eval_metric,
    path=save_path
)

fit_auto.fit(
    train_data=train[predictors + [target]],
    presets='best_quality',
    num_bag_folds=10,
    num_stack_levels = 1,
    excluded_model_types=['KNN'],  
    fit_weighted_ensemble=True,   
    time_limit=time_limit
)


print("\nLeaderboard:")
leaderboard = fit_auto.leaderboard(silent=True)
print(leaderboard)


print("\nFeature Importance:")
vi = fit_auto.feature_importance(train[predictors + [target]])


fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(y=vi.index, width=vi.importance, color='steelblue')
ax.invert_yaxis()
plt.title("Feature Importance")
plt.grid(True)
plt.show()


pred_train = fit_auto.predict(train[predictors])

# plot predictions
plt.figure(figsize=(10, 4))
plt.hist(pred_train, bins=50, color='steelblue')
plt.title('Predictions on Training Data')
plt.grid(True)
plt.show()


print("\nPrediction Summary Stats:")
print(pred_train.describe())


preds = fit_auto.predict(test[predictors])

submission[target] = preds

submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved as submission.csv")
submission.head()


vi_sorted = vi.sort_values("importance", ascending=False).reset_index(drop=True)
vi_sorted["importance_pct"] = 100 * vi_sorted["importance"] / vi_sorted["importance"].sum()

# print full table to console (all rows)
print(vi_sorted.to_string(index=False))

# display nicely in a Jupyter notebook
from IPython.display import display
display(vi_sorted)

# save to CSV
vi_sorted.to_csv("feature_importances_full.csv", index=False)
print("Saved feature importances to feature_importances_full.csv")


try:
    vi_raw = vi
except NameError:
    # fallback: compute vi from your automl/model if you haven't created `vi`
    # make sure `predictors`, `target`, `train`, and `fit_auto` exist in your session
    vi_raw = fit_auto.feature_importance(train[predictors + [target]])

# --- normalize vi_raw into a DataFrame with columns ['feature','importance'] ---
if isinstance(vi_raw, pd.DataFrame):
    vi_df = vi_raw.copy()
    if 'feature' in vi_df.columns and 'importance' in vi_df.columns:
        pass
    elif vi_df.shape[1] == 1:
        vi_df = vi_df.reset_index().rename(columns={'index': 'feature', vi_df.columns[0]: 'importance'})
    else:
        # try index as feature, first numeric column as importance
        if vi_df.index.nlevels == 1 and vi_df.select_dtypes(include=[np.number]).shape[1] >= 1:
            num_col = vi_df.select_dtypes(include=[np.number]).columns[0]
            vi_df = vi_df.reset_index().rename(columns={'index': 'feature', num_col: 'importance'})[['feature', 'importance']]
        else:
            # fallback: take first two columns
            vi_df = vi_df.iloc[:, :2].copy()
            vi_df.columns = ['feature', 'importance']
elif isinstance(vi_raw, pd.Series):
    vi_df = vi_raw.reset_index()
    vi_df.columns = ['feature', 'importance']
elif isinstance(vi_raw, (np.ndarray, list, tuple)):
    # assume order matches predictors if available
    features = predictors if 'predictors' in globals() else [f"f{i}" for i in range(len(vi_raw))]
    vi_df = pd.DataFrame({'feature': features, 'importance': np.asarray(vi_raw).flatten()[:len(features)]})
else:
    # last-resort conversion
    vi_df = pd.DataFrame(vi_raw)
    if vi_df.shape[1] == 1:
        vi_df = vi_df.reset_index().rename(columns={'index': 'feature', 0: 'importance'})
    else:
        vi_df = vi_df.iloc[:, :2]
        vi_df.columns = ['feature', 'importance']

# --- clean, aggregate (if duplicates), sort, and compute percent ---
vi_df['importance'] = pd.to_numeric(vi_df['importance'], errors='coerce').fillna(0.0)
vi_df = vi_df.groupby('feature', as_index=False)['importance'].sum()
vi_df = vi_df.sort_values('importance', ascending=False).reset_index(drop=True)

total_imp = vi_df['importance'].sum()
vi_df['importance_pct'] = (vi_df['importance'] / total_imp * 100) if total_imp != 0 else 0.0

# --- print everything to console (all features) ---
print("\nFeature Importance (all features):")
# show full table even if long
print(vi_df.to_string(index=False, float_format="%.6f"))

# also display nicely in notebook
display(vi_df)

# save to csv for later
vi_df.to_csv("feature_importances_full.csv", index=False)
print("\nSaved feature importances to 'feature_importances_full.csv'")

# --- plot: horizontal bar (all features) ---
plot_df = vi_df.sort_values('importance', ascending=True)  # ascending so largest on top
fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(plot_df))))
ax.barh(y=plot_df['feature'], width=plot_df['importance'], color='steelblue')
ax.invert_yaxis()
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
ax.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

