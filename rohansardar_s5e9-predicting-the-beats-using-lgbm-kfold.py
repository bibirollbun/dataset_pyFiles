import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


train.info()


target = "BeatsPerMinute"
plt.figure(figsize=(6, 3))
sns.histplot(train[target], kde=True, bins=50, color='#bd26d4')
plt.title(f"Distribution of {target}")
plt.show()


num_features = train.select_dtypes(include=[np.number]).columns.tolist()
colors = sns.color_palette("Set2", len(num_features))

plt.figure(figsize=(15, 20))
for i, col in enumerate(num_features):
    plt.subplot(5, 2, i + 1)
    sns.histplot(train[col], bins=50, kde=True, color=colors[i])
    plt.title(col)

plt.tight_layout()
plt.suptitle("Feature Distributions", fontsize=20, y=1.02)
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train[num_features].corr(), annot=True, fmt=".2f", cmap="mako")
plt.title("Correlation Heatmap", fontsize=20, y=1.02)
plt.show()


def fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000  
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)  
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs']) 
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01) 
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / df['Duration_Minutes']

    return df

train_fe = fe(train)
test_fe = fe(test)


X = train_fe.drop('BeatsPerMinute', axis=1)
y = train_fe['BeatsPerMinute']


def train_lightgbm(train, test, target):
    X = train
    y = target
    X_test = test.copy()
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_preds = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.0025,
            num_leaves=100,
            max_depth=10,               
            min_child_samples=10,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            boosting_type='gbdt',
            metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200)
            ]
        )
        
        models.append(model)
        y_preds += model.predict(X_test) / n_splits
    
    print("\nLightGBM regression model training complete.")
    return y_preds, models


y_probs, models = train_lightgbm(X, test_fe, y)


sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission = pd.DataFrame({
    'id': sub['id'],
    'BeatsPerMinute': y_probs
})

submission.to_csv('submission.csv', index=False)

