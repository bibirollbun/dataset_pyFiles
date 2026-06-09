import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train_data.head()


train_data.info()


train_data.isnull().sum()


train_data.duplicated().sum()


train_data.describe()


# Histograms
df = train_data.copy()
df.drop('id',axis=1,inplace=True)
df.hist(bins=30, figsize=(16, 10))
plt.tight_layout()
plt.show()


# Boxplots
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)

# Turn off unused axes (if any)
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.heatmap(train_data.drop('id',axis=1).corr(),annot=True,cmap='coolwarm',fmt=".2f")
plt.show()


skew_vals = df.skew().sort_values(ascending=False)
print(skew_vals)


# 1. Rhythm + Energy interaction
# Captures combined effect of rhythm strength and track energy on BPM
df["EnergyRhythm"] = df["Energy"] * df["RhythmScore"]

# 2. Danceability estimate
# Proxy for "danceability": higher for energetic, rhythmic, and less vocal-heavy tracks
df["Danceability_est"] = df["Energy"] * df["RhythmScore"] * (1 - df["VocalContent"])

# 3. Mood × Rhythm interaction
# Combines mood score and rhythm score; helps model genre or emotional influence on tempo
df["MoodRhythm"] = df["MoodScore"] * df["RhythmScore"]

# 4. Energy squared
# Captures non-linear impact of energy; high-energy tracks may have disproportionately higher BPM
df["Energy_sq"] = df["Energy"] ** 2

# 5. Vocal to Acoustic ratio
# Measures vocal-heavy vs acoustic-heavy content; influences BPM and track style
df["Vocal_to_Acoustic"] = df["VocalContent"] / (df["AcousticQuality"] + 1e-6)

# 6. Live factor
# Captures live, vocal-heavy performances; live tracks often have distinct tempo patterns
df["LiveFactor"] = df["LivePerformanceLikelihood"] * (1 - df["InstrumentalScore"])

# 7. Instrumental × Acoustic quality
# Interaction between instrumental and acoustic quality; can indicate style/genre affecting BPM
df["InstrumentalQuality"] = df["InstrumentalScore"] * df["AcousticQuality"]
df.head()


# Split
X = df.drop('BeatsPerMinute',axis=1)
y = df['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape,y_train.shape)
print("Test set shape:", X_val.shape,y_val.shape)


# Define Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200,min_samples_split=3, min_samples_leaf=2,max_depth=10,random_state=42)

# Fit
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_val)

rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf = r2_score(y_val, y_pred_rf)

print("Test RMSE Random forest:", rmse_rf)
print("Test R² Random forest:", r2_rf)


# Feature Importance scores
importances = rf.feature_importances_

# Put into a DataFrame
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# Define XGBoost Regressor

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define hyperparameter grid
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist_xgb,
    n_iter=15,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

# Fit
random_search_xgb.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", random_search_xgb.best_params_)

# Predict on test set
best_xgb = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_val)

rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
r2_xgb = r2_score(y_val, y_pred_xgb)

print("Test RMSE XGB:", rmse_xgb)
print("Test R² XGB :", r2_xgb)


best_xgb = random_search_xgb.best_estimator_
best_xgb.fit(X,y)


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test.head()


subdata = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
subdata.head()


sub = test.copy()
sub.drop('id',axis=1,inplace=True)
# create new features same as train data
sub["EnergyRhythm"] = sub["Energy"] * sub["RhythmScore"]
sub["Danceability_est"] = sub["Energy"] * sub["RhythmScore"] * (1 - sub["VocalContent"])
sub["MoodRhythm"] = sub["MoodScore"] * sub["RhythmScore"]
sub["Energy_sq"] = sub["Energy"] ** 2
sub["Vocal_to_Acoustic"] = sub["VocalContent"] / (sub["AcousticQuality"] + 1e-6)
sub["LiveFactor"] = sub["LivePerformanceLikelihood"] * (1 - sub["InstrumentalScore"])
sub["InstrumentalQuality"] = sub["InstrumentalScore"] * sub["AcousticQuality"]

y_pred = best_xgb.predict(sub)


subdf = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': y_pred})
subdf.to_csv('Submission.csv', index=False)


subdf.head()

