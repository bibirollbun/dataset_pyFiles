import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
print("Shape of dataset:", df.shape)

# Quick look
df.head()


df.info()


print(df.describe().T)


df.isnull().sum()


plt.figure(figsize=(8,5))
sns.histplot(df['BeatsPerMinute'], kde=True, bins=30)
plt.title(f"Distribution of {'BeatsPerMinute'}")
plt.show()


df=df.drop(['id'],axis=1)


df


num_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Plot histograms for all numeric features
df[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


# ========================
plt.figure(figsize=(10,8))
sns.heatmap(df[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


for col in num_features:
    if col != 'BeatsPerMinute':
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df['BeatsPerMinute'])
        plt.title(f"{col} vs {'BeatsPerMinute'}")
        plt.show()


for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()


features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality']

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Winsorization (capping)
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))


df[col]


df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)
df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']


test['TrackDurationMin'] = test['TrackDurationMs'] / 60000
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + 1e-5)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + 1e-5)
test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']
test['RhythmEnergy'] = test['RhythmScore'] * test['Energy']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']


test


df


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
X = df.drop(columns=['BeatsPerMinute'])
y = df['BeatsPerMinute']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.001,
    'num_leaves': 35,
    'min_data_in_leaf': 90,
    'feature_fraction': 0.8786957228471932,
    'bagging_fraction': 0.7966824793412932,
    'bagging_freq': 6,
    'lambda_l1': 7.151613714286091,
    'lambda_l2': 5.489198722797788,
    'min_gain_to_split': 2.4913261623670584,
    'max_depth': 17,
    'verbose': -1,
    'random_state': 42
}

# Initialize model
model = lgb.LGBMRegressor(**params)

# Fit model
model.fit(X_train, y_train)



y_pred=model.predict(X_test)


y_pred


val_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


val_rmse


test_pred = model.predict(test.drop("id", axis=1))


submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_pred
})


submission.to_csv("Beat_preM.csv", index=False)
print("✅ Submission file saved as Beat_preM.csv")





