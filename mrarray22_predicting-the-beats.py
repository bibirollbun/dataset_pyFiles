# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)



df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
print("Shape of dataset:", df.shape)

df.head()


df.info()
df.describe(include='all').T


missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("✅ No missing values found.")

print("Duplicate rows:", df.duplicated().sum())


target = "BeatsPerMinute"

plt.figure(figsize=(8,5))
sns.histplot(df[target], kde=True, bins=30)
plt.title(f"Distribution of {target}")
plt.show()


num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("id")

df[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\n--- Correlation with Target ---")
print(df.corr()[target].sort_values(ascending=False))


for col in num_features:
    if col != target:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
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


df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)
df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
X = df.drop(columns=['id', target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'max_depth': [3, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}


xgb = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(
estimator=xgb,
param_distributions=param_dist,
n_iter=20,
scoring='neg_mean_squared_error',
cv=3,
verbose=1,
n_jobs=-1,
random_state=42
)


random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(10,6))
plt.barh(X.columns, best_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test_df['TrackDurationMin'] = test_df['TrackDurationMs'] / 60000
test_df['Energy_Acoustic_Ratio'] = test_df['Energy'] / (test_df['AcousticQuality'] + 1e-5)
test_df['Vocal_Instrument_Balance'] = test_df['VocalContent'] / (test_df['InstrumentalScore'] + 1e-5)
test_df['MoodRhythm'] = test_df['MoodScore'] * test_df['RhythmScore']
test_df['PerformanceIntensity'] = test_df['LivePerformanceLikelihood'] * test_df['AudioLoudness']
test_df['RhythmEnergy'] = test_df['RhythmScore'] * test_df['Energy']
test_df['MoodAcoustic'] = test_df['MoodScore'] * test_df['AcousticQuality']

train_features = best_model.get_booster().feature_names
X_test_final = test_df[train_features] 
y_pred_test = best_model.predict(X_test_final)


output = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_pred_test
})

output.to_csv("final_test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")
print(output.head())

