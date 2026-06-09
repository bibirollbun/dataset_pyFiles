import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.head()


train.shape


train.info()


train.isnull().sum()


train.describe()


test.head()


test.shape


test.info()


test.isnull().sum()


sns.histplot(train['BeatsPerMinute'], kde=True, bins=50, color='orange')
plt.xlabel('BPM')
plt.ylabel('Frequency');


plt.figure(figsize=(8, 8))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='plasma')
plt.title('Correlation Matrix');


sample_df = train.sample(1000, random_state=42)
features_to_plot = ['Energy', 'RhythmScore', 'AudioLoudness', 'TrackDurationMs']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(data=sample_df, x=col, y='BeatsPerMinute', alpha=0.6, color='skyblue')
    plt.title(f'{col} vs BPM')
    plt.xlabel(col)
    plt.ylabel('BPM')
plt.tight_layout();


def create_features(df):
    df = df.copy()
    # Energy - Rhythm
    df['Energy_x_Rhythm'] = df['Energy'] * df['RhythmScore']
    df['Energy_x_Loudness'] = df['Energy'] * df['AudioLoudness']
    df['Mood_x_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    # Ratios
    df['Inst_Vocal_Ratio'] = df['InstrumentalScore'] / (df['VocalContent'] + 1)
    df['Loudness_Energy_Ratio'] = df['AudioLoudness'] / (df['Energy'] + 1)
    # milliseconds to minutes 
    df['Duration_Min'] = df['TrackDurationMs'] / 60000
    # Squared for importance
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Rhythm_Squared'] = df['RhythmScore'] ** 2
    
    return df

train = create_features(train)
test = create_features(test)


x = train.drop(columns=['id', 'BeatsPerMinute'])
y = train['BeatsPerMinute']
test_df = test.drop(columns=['id'])


kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test_df))
scores = []

for fold, (train_index, val_index) in enumerate(kf.split(x, y)):
    X_train, X_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = CatBoostRegressor(iterations=1500,learning_rate=0.05,depth=6,loss_function='RMSE',verbose=0,random_seed=42)
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    val_pred = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(score)
    
    print(f"Fold {fold+1} RMSE: {score:.5f}")
    
    preds += model.predict(test_df) / kf.get_n_splits()

print(f"RMSE: {np.mean(scores):.5f}")


submission['BeatsPerMinute'] = preds
submission.to_csv('submission.csv', index=False)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

def create_features(df):
    df = df.copy()
    df['Energy_x_Rhythm'] = df['Energy'] * df['RhythmScore']
    df['Energy_x_Loudness'] = df['Energy'] * df['AudioLoudness']
    df['Mood_x_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Inst_Vocal_Ratio'] = df['InstrumentalScore'] / (df['VocalContent'] + 1)
    df['Loudness_Energy_Ratio'] = df['AudioLoudness'] / (df['Energy'] + 1)
    df['Duration_Min'] = df['TrackDurationMs'] / 60000
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Rhythm_Squared'] = df['RhythmScore'] ** 2
    return df

train_fe = create_features(train)

X = train_fe.drop(columns=['id', 'BeatsPerMinute'])
y = train_fe['BeatsPerMinute']

model = CatBoostRegressor(iterations=1500,learning_rate=0.05,depth=6,loss_function='RMSE',verbose=100,random_seed=42)

model.fit(X, y)

model.save_model("catboost_model.cbm")




