import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train.head()


train.shape


train.isnull().sum()


train.duplicated().sum()


train.describe()


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test.head()


test.shape


test.isnull().sum()


test.duplicated().sum()


combined = pd.concat([train, test], axis = 0)
combined.shape


sns.displot(train['BeatsPerMinute'], kde = True)


input_variables = combined.drop('BeatsPerMinute', axis = 1)


def replace_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr
    median_value = df[column].median()
    df[column] = np.where((df[column]<lower)|(df[column]>upper), median_value, df[column])
    return df


fig, axes = plt.subplots(2, 5, figsize=(15,5))
axes = axes.flatten()

for i, col in enumerate(input_variables.columns):
    sns.boxplot(data = combined, y=col, ax=axes[i])
    axes[i].set_title(f'Bloxplot of {col}')

plt.tight_layout()
plt.show()


for col in input_variables.columns:
    concat_df = replace_outliers(combined, col)


fig, axes = plt.subplots(2, 5, figsize=(15,5))
axes = axes.flatten()

for i, col in enumerate(input_variables.columns):
    sns.boxplot(data = combined, y=col, ax=axes[i])
    axes[i].set_title(f'Bloxplot of {col}')

plt.tight_layout()
plt.show()


combined["TrackDurationMin"] = combined["TrackDurationMs"] / 60000  
combined["EnergyPerMin"] = combined["Energy"] / (combined["TrackDurationMin"] + 1e-5)
combined["VocalToInstrumental"] = combined["VocalContent"] / (combined["InstrumentalScore"] + 1e-5)
combined["RhythmToMood"] = combined["RhythmScore"] / (combined["MoodScore"] + 1e-5)
combined["AcousticToLoudness"] = combined["AcousticQuality"] / (combined["AudioLoudness"] + 1e-5)
combined["EnergyMoodInteraction"] = combined["Energy"] * combined["MoodScore"]
combined["EnergyRhythmInteraction"] = combined["Energy"] * combined["RhythmScore"]
combined["NormRhythm"] = combined["RhythmScore"] / (combined["RhythmScore"].max() + 1e-5)
combined["NormAcoustic"] = combined["AcousticQuality"] / (combined["AcousticQuality"].max() + 1e-5)
combined["NormLoudness"] = combined["AudioLoudness"] / (combined["AudioLoudness"].max() + 1e-5)
combined["IsLive"] = (combined["LivePerformanceLikelihood"] > 0.5).astype(int)  
combined["IsLongTrack"] = (combined["TrackDurationMin"] > 5).astype(int) 
combined["LogDuration"] = np.log1p(combined["TrackDurationMs"])
combined["SquaredEnergy"] = combined["Energy"] ** 2
combined["SquaredRhythm"] = combined["RhythmScore"] ** 2
combined["OverallMusicQuality"] = (combined["RhythmScore"] + combined["AcousticQuality"] + combined["InstrumentalScore"] + combined["VocalContent"]) / 4  
combined["PerformanceIndex"] = (combined["LivePerformanceLikelihood"] * combined["Energy"])


newtrain = combined.iloc[0:524164, :]
newtest = combined.iloc[524164:, :].drop('BeatsPerMinute', axis = 1)


newtrain.head()


newtrain.shape


newtest.shape


X = newtrain.drop(columns=['BeatsPerMinute'])
y = newtrain['BeatsPerMinute']
k = 12  
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X, y)
mask = selector.get_support()
selected_features = X.columns[mask]
print(f"Top {k} selected features:")
print(selected_features)


correlation_matrix = newtrain.corr()
target_corr = correlation_matrix["BeatsPerMinute"].sort_values(ascending=False)
print(target_corr)


x = newtrain[['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy',
       'TrackDurationMin', 'EnergyPerMin', 'NormRhythm', 'LogDuration',
       'SquaredEnergy', 'SquaredRhythm', 'OverallMusicQuality']]
y = newtrain['BeatsPerMinute']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(),
         'Bagging': BaggingRegressor(), 'Extra Tree': ExtraTreesRegressor(), 'LightGBM': LGBMRegressor(),
         'Gradient Boosting': GradientBoostingRegressor(), 'Adaboost': AdaBoostRegressor(),
         'XGB': XGBRegressor(), 'lasso': Lasso(), 'ridge' : Ridge()}


def evaluate_models(x_train, x_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        predictions = model.fit(x_train, y_train).predict(x_test)
        accuracy = np.sqrt(mean_squared_error(y_test, predictions))
        results[name] = accuracy
    return results


results = evaluate_models(x_train, x_test, y_train, y_test, models)


best_model_name = min(results, key = results.get)
best_model = models[best_model_name]


print(f"best model is {best_model_name} with r2_square {results[best_model_name]}")


y_pred = best_model.fit(x_train, y_train).predict(x_test)


print(np.sqrt(mean_squared_error(y_test, y_pred)))


x_train = newtrain[['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy',
       'TrackDurationMin', 'EnergyPerMin', 'NormRhythm', 'LogDuration',
       'SquaredEnergy', 'SquaredRhythm', 'OverallMusicQuality']]
y_train = newtrain['BeatsPerMinute']
x_test = newtest[['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 'Energy',
       'TrackDurationMin', 'EnergyPerMin', 'NormRhythm', 'LogDuration',
       'SquaredEnergy', 'SquaredRhythm', 'OverallMusicQuality']]
y_pred = best_model.fit(x_train, y_train).predict(x_test)


solution = pd.DataFrame({'id': test.id,'BeatsPerMinute': y_pred})
solution.head()


solution.to_csv('Solution.csv', index = False)
##26.39


x_train = newtrain[['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs', 
       'TrackDurationMin', 'NormRhythm', 'LogDuration',
        'SquaredRhythm', 'OverallMusicQuality']]
y_train = newtrain['BeatsPerMinute']
x_test = newtest[['RhythmScore', 'VocalContent', 'MoodScore', 'TrackDurationMs',
       'TrackDurationMin', 'NormRhythm', 'LogDuration',
        'SquaredRhythm', 'OverallMusicQuality']]
y_pred = best_model.fit(x_train, y_train).predict(x_test)
solution = pd.DataFrame({'id': test.id,'BeatsPerMinute': y_pred})
solution.to_csv('Solution11.csv', index = False)




