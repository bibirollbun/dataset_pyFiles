import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


df_train.head()


df_train.shape


df_train.info()


df_test.info()


df_train.describe()


features = ['RhythmScore', 'AudioLoudness', 'Energy', 'BeatsPerMinute', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs']
rows = 5
cols = 2

plt.figure(figsize=(12, 15))

for i, feat in enumerate(features, 1):
    plt.subplot(rows, cols, i)  
    sns.histplot(df_train[feat], kde=True)
    plt.title(f'Distribution of {feat}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 15))

for i, feat in enumerate(features, 1):
    plt.subplot(rows, cols, i)  
    sns.boxplot(df_train[feat])
    plt.title(f'BoxPlot of {feat}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(
    df_train.corr(),
    annot=True,         
    cmap='coolwarm',   
)

plt.title('Correlation Heatmap of Features')
plt.show()


def outlier_removal(df_train, col):
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f'Lower bound: {lower_bound}')
    print(f'Upper bound: {upper_bound}')
    df_train = df_train[(df_train[col] >= lower_bound) & (df_train[col] <= upper_bound)]


outlier_removal(df_train, 'InstrumentalScore')
outlier_removal(df_train, 'TrackDurationMs')
outlier_removal(df_train, 'RhythmScore')
outlier_removal(df_train, 'AudioLoudness')


df_train.isna().sum()


df_test.isna().sum()


test_ids = df_test['id']


df_train["TrackDuration_log"] = np.log1p(df_train["TrackDurationMs"])
df_train["VocalContent_log"] = np.log1p(df_train["VocalContent"])
df_train["InstrumentalScore_log"] = np.log1p(df_train["InstrumentalScore"])
df_train["AcousticQuality_log"] = np.log1p(df_train["AcousticQuality"])
df_train["LivePerfLikelihood_log"] = np.log1p(df_train["LivePerformanceLikelihood"])

df_train["AudioLoudness_log"] = np.log1p(df_train["AudioLoudness"] - df_train["AudioLoudness"].min() + 1)

cols = ["RhythmScore","AudioLoudness","VocalContent","AcousticQuality",
        "InstrumentalScore","LivePerformanceLikelihood","MoodScore","Energy"]

df_train["mean_score"] = df_train[cols].mean(axis=1)
df_train["std_score"] = df_train[cols].std(axis=1)

# df_train['acoustic_instrumental_ratio'] = df_train['AcousticQuality'] / (df_train['InstrumentalScore'] + 1e-6)
# df_train['RhythmEnergyRatio'] = df_train['RhythmScore'] / (df_train['Energy'] + 1e-8)
# df_train['VocalInstrumentalRatio'] = df_train['VocalContent'] / (df_train['InstrumentalScore'] + 1e-8)




df_test["TrackDuration_log"] = np.log1p(df_test["TrackDurationMs"])
df_test["VocalContent_log"] = np.log1p(df_test["VocalContent"])
df_test["InstrumentalScore_log"] = np.log1p(df_test["InstrumentalScore"])
df_test["AcousticQuality_log"] = np.log1p(df_test["AcousticQuality"])
df_test["LivePerfLikelihood_log"] = np.log1p(df_test["LivePerformanceLikelihood"])

df_test["AudioLoudness_log"] = np.log1p(df_test["AudioLoudness"] - df_train["AudioLoudness"].min() + 1)

cols = ["RhythmScore","AudioLoudness","VocalContent","AcousticQuality",
        "InstrumentalScore","LivePerformanceLikelihood","MoodScore","Energy"]

df_test["mean_score"] = df_test[cols].mean(axis=1)
df_test["std_score"] = df_test[cols].std(axis=1)

# df_test['acoustic_instrumental_ratio'] = df_test['AcousticQuality'] / (df_test['InstrumentalScore'] + 1e-6)
# df_test['RhythmEnergyRatio'] = df_test['RhythmScore'] / (df_test['Energy'] + 1e-8)
# df_test['VocalInstrumentalRatio'] = df_test['VocalContent'] / (df_test['InstrumentalScore'] + 1e-8)





st = StandardScaler()
mm = MinMaxScaler()
X = df_train.drop(columns=['BeatsPerMinute', 'id'])
y = df_train['BeatsPerMinute']
X = st.fit_transform(X)
test_df_X = st.transform(df_test.drop(columns=['id']))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)


parameters = {'iterations': 500, 'learning_rate': 0.012238274912528409, 'depth': 5, 'l2_leaf_reg': 2.906115521467334, 'bagging_temperature': 0.7813863069325937, 'random_strength': 0.10698086866196457, 'border_count': 65}


lgb = LGBMRegressor(random_state=2, n_estimators=10)
lin_svr = LinearSVR(
    C=0.1,              
    epsilon=0.05,      
    loss="squared_epsilon_insensitive",  
    dual=True,         
    random_state=2
)
cbr = CatBoostRegressor(**parameters,loss_function="RMSE", verbose=0)
estimators = [("lgb",lgb),("dt", DecisionTreeRegressor(random_state=2, max_depth=3)), ('cbr', cbr)]


for model in estimators:
    model[1].fit(X_train, y_train)
    pred1 = model[1].predict(X_test)
    print(model[0], (mean_squared_error(y_test, pred1)**0.5))


voting_reg = VotingRegressor(estimators)


rmse_scores = cross_val_score(voting_reg, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -rmse_scores
print("Cross-validated RMSE scores:", rmse_scores)
print("Mean RMSE:", np.mean(rmse_scores))


voting_reg.fit(X_train, y_train)


# model_predictions = model.predict(test_df_X)
voter_predictions = voting_reg.predict(test_df_X)
# final_predictions = model_predictions*0.05 + voter_predictions*0.95
submission_df = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': voter_predictions
    })

submission_df.to_csv("submission.csv", index=False)

