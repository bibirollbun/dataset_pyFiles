import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression
import warnings
from sklearn.cluster import KMeans


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train_df = train_df.drop(columns = "id") # don't need ids in train set


train_df.head()


print(train_df.shape)
print(test_df.shape)


train_df.describe()


train_df.isna().sum()


warnings.filterwarnings('ignore')
sns.pairplot(data = train_df)
plt.show()


train_df.hist(bins = 30, figsize = (15,13), layout = (5,2))
plt.suptitle("Feature Distributions")
plt.show()


X = train_df.copy()
y = X.pop('BeatsPerMinute')


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features = discrete_features)
    mi_scores = pd.Series(mi_scores, name = "MI Scores", index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
discrete_features = X.dtypes == int

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi = 100, figsize = (8, 5))
plot_mi_scores(mi_scores)


X.AudioLoudness = X.AudioLoudness * -1
X.AudioLoudness = np.log(X.AudioLoudness)
X.VocalContent = np.log(X.VocalContent)
X.AcousticQuality = np.log(X.AcousticQuality)
X.InstrumentalScore = np.log(X.InstrumentalScore)
X['TrackDuration'] = X.TrackDurationMs / 60000 # duration in minutes
X.pop('TrackDurationMs')
X.TrackDuration = np.log(X.TrackDuration)
#X['TotalScore'] = X.RhythmScore + X.InstrumentalScore + X.MoodScore

test_df.AudioLoudness = test_df.AudioLoudness * -1
test_df.AudioLoudness = np.log(test_df.AudioLoudness)
test_df.VocalContent = np.log(test_df.VocalContent)
test_df.AcousticQuality = np.log(test_df.AcousticQuality)
test_df.InstrumentalScore = np.log(test_df.InstrumentalScore)
test_df['TrackDuration'] = test_df.TrackDurationMs / 60000 # duration in minutes
test_df.pop('TrackDurationMs')
test_df.TrackDuration = np.log(X.TrackDuration)
#test_df['TotalScore'] = test_df.RhythmScore + test_df.InstrumentalScore + test_df.MoodScore


inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 336, n_init = 10)
    kmeans.fit(X[['RhythmScore', 'MoodScore', 'InstrumentalScore']])
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker = 'o')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


kmeans = KMeans(n_clusters = 3, random_state = 336, n_init = 10)
kmeans.fit(X[['RhythmScore', 'MoodScore', 'InstrumentalScore']])
labels = kmeans.labels_
X['ScoreCluster'] = labels


test_labels = kmeans.predict(test_df[['RhythmScore', 'MoodScore', 'InstrumentalScore']])
test_df['ScoreCluster'] = test_labels


transformer_num = StandardScaler()
transformer_cat = OneHotEncoder(handle_unknown = "ignore")

features_num = (X.columns[0:9]).tolist()
features_cat = ['ScoreCluster']

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
    remainder = "drop"
)

X = preprocessor.fit_transform(X)
encoded_features = preprocessor.named_transformers_['onehotencoder'].get_feature_names_out(features_cat)
columns = features_num + list(encoded_features)
X = pd.DataFrame(X, columns = columns)

X_test = test_df.copy()
ids = X_test.pop('id')
X_test = preprocessor.transform(X_test)
X_test = pd.DataFrame(X_test, columns = columns)


model = XGBRegressor(use_label_encoder = False, eval_metric = 'rmse')

param_grid = {
    "n_estimators": [300],
    "max_depth": [5],
    "learning_rate": [0.01],
    "subsample": [0.7],
    "colsample_bytree": [0.8]
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_root_mean_squared_error',
    cv = 10,
    verbose = 1,
    n_jobs = -1
)

grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)


best_model = grid_search.best_estimator_
importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending = False)

print(feature_importance_df)


best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)
submission = pd.DataFrame(ids)
submission['BeatsPerMinute'] = preds
submission.to_csv('submission.csv', index = False)

