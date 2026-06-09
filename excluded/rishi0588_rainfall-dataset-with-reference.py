!pip install -q category_encoders


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


def mi_score(X, y):
  scores = mutual_info_regression(X, y)
  return pd.DataFrame(scores, index=X.columns, columns=['mi_score']).sort_values(by="mi_score", ascending=False)

def plot_mi_score(data, plot=False):
  X, y = data.drop("rainfall", axis=1), data[["rainfall"]]
  mi_scores = mi_score(X, y)
  mi_scores = mi_scores.sort_values(by="mi_score", ascending=True)
  if plot:
    plt.barh(mi_scores.index, mi_scores["mi_score"])
  else:
    print(mi_scores)

def LogisticModel(data, frac=0.2, random_state=42, algo=LogisticRegression, **params):
  X, y = data.drop("rainfall", axis=1), data[["rainfall"]]
  model = algo(**params)
  cv_score = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
  print("Roc Score mean : ", cv_score.mean())
  print("Roc Scores : ", cv_score)

  return


def plotDailyChanges(feature:str, func=None):
  # sns.scatterplot("day", feature, data=data)
  copyTrain = trainset.copy()
  copyOrigin = original_train.copy()
  if func:
    copyTrain[feature] = func(copyTrain[feature])
    copyOrigin[feature] = func(copyOrigin[feature])
  plt.figure(figsize=(18, 6))
  plt.subplot(1, 2, 1)
  plt.title("ScatterPlot : Trainset vs OriginalSet")
  sns.scatterplot(x="day", y=feature, label=f"Trainset {feature}", data=copyTrain)
  sns.scatterplot(x="day", y=feature, label=f"OriginalSet {feature}", data=copyOrigin)
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.title("Smooth Distribution : Trainset vs OriginalSet")
  sns.kdeplot(copyTrain[feature], shade=True, fill=True, color="blue", label=f"Trainset {feature}", alpha=0.5)
  sns.kdeplot(copyOrigin[feature], shade=True, fill=True, color="red", label=f"Trainset {feature}", alpha=0.5)
  plt.legend()
  plt.show()




import kagglehub
original = kagglehub.dataset_download('subho117/rainfall-prediction-using-machine-learning')


trainset = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
testset = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
original_train = pd.read_csv(original + "/Rainfall.csv")

trainset.head()


features = trainset.select_dtypes(include=float)
sns.set_theme(
    style="whitegrid",  # Options: "darkgrid", "whitegrid", "dark", "white", "ticks"
    font="serif",       # You can change this to "sans-serif" or other available fonts
    rc={"axes.grid": True}  # Ensure grid is enabled
)
plt.figure(figsize=(12, 6))
sns.relplot(x="day", y="value", col="hour",
    data=trainset.melt(id_vars="day", var_name="hour", value_vars=features), col_wrap=3)

plt.show()


original_train.columns = [col.strip() for col in original_train.columns]
original_train["rainfall"] = original_train["rainfall"].map(lambda x : 1 if x=="yes" else 0)
features = original_train.columns
features = features[original_train.isnull().sum().gt(0)]
for name in features:
  original_train[name].fillna(original_train[name].mean(), inplace=True)

original_train["day"] = original_train.index.values + 1


trainset.drop("id", axis=1, inplace=True)
trainset = pd.concat((original_train[original_train.rainfall == 1], trainset), axis=0)


from sklearn.feature_selection import mutual_info_regression

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_mi_score(trainset, True)
plt.title("Trainset")

plt.subplot(1, 2, 2)
plot_mi_score(original_train, True)
plt.title("OriginalSet")

plt.subplots_adjust(wspace=1)

plt.show()


fig, ax = plt.subplots(1, 2, figsize=[12, 6])
sns.countplot(x="rainfall", data=trainset, ax=ax[0])
sns.countplot(x="rainfall", data=original_train, ax=ax[1])
plt.show()


LogisticModel(trainset)
print(40 * "--")
LogisticModel(original_train)


testset["winddirection"].fillna(testset["winddirection"].median(), inplace=True)


copyset = trainset.copy()

trainset["LCL"] = (trainset.temparature - trainset.dewpoint) / 0.008
original_train["LCL"] = (original_train.temparature - original_train.dewpoint) / 0.008
testset["LCL"] = (testset.temparature - testset.dewpoint) / 0.008
plot_mi_score(trainset[["LCL", "rainfall"]])


from sklearn.linear_model import LinearRegression

linear = LinearRegression()
X = trainset[["humidity", "pressure", "temparature"]]
y = trainset[["rainfall"]]

linear.fit(X, y)
a = linear.intercept_
b, c, d  = linear.coef_[0]

trainset["PRain"] = X["humidity"] * b + c * X["pressure"] + d * X["temparature"] + a
original_train["PRain"] = original_train["humidity"] * b + c * original_train["pressure"] + d * original_train["temparature"] + a
testset["PRain"] = testset["humidity"] * b + c * testset["pressure"] + d * testset["temparature"] + a
plot_mi_score(trainset[["PRain", "rainfall"]])


plotDailyChanges("humidity")


plotDailyChanges("windspeed")


plotDailyChanges("temparature")


plotDailyChanges("temparature", np.log)


plotDailyChanges("pressure")


plotDailyChanges("cloud")


plotDailyChanges("LCL")


plotDailyChanges("PRain")


trainset["potential_temperature"] = trainset["temparature"] * ((1000 / trainset["pressure"]) ** 0.286)
testset["potential_temperature"] = testset["temparature"] * ((1000 / testset["pressure"]) ** 0.286)
original_train["potential_temperature"] = original_train["temparature"] * ((1000 / original_train["pressure"]) ** 0.286)
plot_mi_score(trainset[["potential_temperature", "rainfall"]])


plotDailyChanges("potential_temperature")


trainset["Date"] = pd.to_datetime(trainset["day"], unit="D", origin="2024-12-31")
trainset["month"] = trainset["Date"].dt.month
trainset["week"] = trainset["Date"].dt.isocalendar().week


original_train["Date"] = pd.to_datetime(original_train["day"], unit="D", origin="2024-12-31")
original_train["month"] = original_train["Date"].dt.month
original_train["week"] = original_train["Date"].dt.isocalendar().week

testset["Date"] = pd.to_datetime(testset["day"], unit="D", origin="2024-12-31")
testset["month"] = testset["Date"].dt.month
testset["week"] = testset["Date"].dt.isocalendar().week


def LinePlotOnSeasons(feature:str, func="mean"):
  plt.figure(figsize=(12, 6))
  plt.title("Trend on Each Season")
  sns.lineplot(x="day", y=feature, data=trainset, label="Original Trend", color="green")
  sns.lineplot(x=trainset["day"], y=trainset.groupby("month")[feature].transform(func), label=f"Monthly {func} Trend", color='red')
  sns.lineplot(x=trainset["day"], y=trainset.groupby("week")[feature].transform(func), label=f"Weekly {func} Trend", color='blue')
  plt.legend()
  plt.show()
LinePlotOnSeasons("humidity")


LinePlotOnSeasons("pressure")


trainset.reset_index(inplace=True)
trainset.drop("index", axis=1, inplace=True)


from sklearn.decomposition import PCA


features = testset.columns[2:]
X = trainset[features]
y = trainset[["rainfall"]]

X_scaled = ((X - X.mean(axis=0)) / X.std(axis=0))
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()


plot_mi_score(pd.concat([X_pca, y], axis=1), True)


trainset = pd.concat((trainset, X_pca[['PC2', 'PC3', 'PC9', 'PC5']]), axis=1)


X = testset[features]

X_scaled = ((X - X.mean(axis=0)) / X.std(axis=0))
X_pca = pca.transform(X_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
testset = pd.concat((testset, X_pca[['PC2', 'PC3', 'PC9', 'PC5']]), axis=1)


loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings


def FeatureEngineering(trainset):
  eps = 1e-3
  trainset["midRange"] = trainset["maxtemp"] - trainset["mintemp"]
  trainset["maxDiffTemp"] = trainset["maxtemp"] - trainset["temparature"]
  trainset["minDiffTemp"] = trainset["temparature"] - trainset["mintemp"]
  trainset["windDirectionPressure"] = 0.6 * trainset["winddirection"] + 0.4 * trainset["pressure"]
  trainset["DewpointPressure"] = 0.4 * trainset["dewpoint"] + 0.6 * trainset["pressure"]
  trainset["HumidityCloud"] = trainset["humidity"] / ( trainset["cloud"] + eps)
  trainset["HumiditySunshine"] = trainset["humidity"] / ( trainset["cloud"] + eps)
  trainset["CloudSunshine"] = trainset["cloud"] / ( trainset["sunshine"] + eps)
  trainset["HumidityWindDirection"] = (0.6 * trainset["humidity"] )/ (( 0.3 * trainset["sunshine"]) + eps)
  trainset["SunshineWindDirection"] = (0.6 * trainset["sunshine"] )/ (( 0.3 * trainset["winddirection"]) + eps)
  return trainset

trainset = FeatureEngineering(trainset)
original_train = FeatureEngineering(original_train)
testset = FeatureEngineering(testset)


def Bucketizing(dataset, col, feature="day", func="median"):
  dataset[col + f"_Bucket_{feature}"] = dataset.groupby(feature)[col].transform(func).astype("category")
  return dataset
features = list(original_train.select_dtypes(include=float).columns)

for col in features:
  trainset = Bucketizing(trainset, col)
  original_train = Bucketizing(original_train, col)
  testset = Bucketizing(testset, col)
  trainset = Bucketizing(trainset, col, feature="week")
  original_train = Bucketizing(original_train, col, feature="week")
  testset = Bucketizing(testset, col, feature="week")



from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
feature = copyset.select_dtypes(include=float).columns

X = trainset[feature]
y = trainset["rainfall"]

X_scaled = ((X - X.mean(axis=0)) / X.std(axis=0))
gm = GaussianMixture(n_components=10)
kmeans = KMeans(n_clusters=10)
gm  = gm.fit(X_scaled)
kmeans = kmeans.fit(X_scaled)

trainset["KmeansLabel"] =  gm.predict(X_scaled).astype("object")
trainset["GMLabel"] =  kmeans.predict(X_scaled).astype("object")
kmeans_Features = [f"kmeans_{i}" for i in feature]
distance = pd.DataFrame(kmeans.transform(X_scaled), columns=kmeans_Features)
trainset = pd.concat([trainset, distance], axis=1)

X = testset[feature]
# y = testset["rainfall"]

X_scaled = ((X - X.mean(axis=0)) / X.std(axis=0))
testset["KmeansLabel"] =  gm.predict(X_scaled).astype("object")
testset["GMLabel"] =  kmeans.predict(X_scaled).astype("object")
kmeans_Features = [f"kmeans_{i}" for i in feature]
distance = pd.DataFrame(kmeans.transform(X_scaled), columns=kmeans_Features)
testset = pd.concat([testset, distance], axis=1)


from category_encoders import TargetEncoder
features = trainset.select_dtypes(include=["category", "object"]).columns

for col in features:
  encoder = TargetEncoder()
  trainset[col + "_encoded"] = encoder.fit_transform(trainset[[col]], trainset[["rainfall"]])
  testset[col + "_encoded"] = encoder.transform(testset[[col]])


from category_encoders import MEstimateEncoder


for col in features:
  encoder = MEstimateEncoder(cols=[col], m=5.0)
  trainset[col + "_encoded_ME"] = encoder.fit_transform(trainset[[col]], trainset[["rainfall"]])
  testset[col + "_encoded_ME"] = encoder.transform(testset[[col]])


for col in trainset.select_dtypes(include=["object", "category"]).columns:
  trainset[col] = trainset[col].astype(float)
  testset[col] = testset[col].astype(float)

trainset.drop("Date", axis=1, inplace=True)
testset.drop("Date", axis=1, inplace=True)
testset.drop("id", axis=1, inplace=True)


from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_preds = []

oof_ens = np.zeros(len(trainset))
pred_ens = np.zeros(len(testset))


for i, (train_idx, test_idx) in enumerate(kf.split(trainset)):
  print("--"*20)
  print(f"Fold {i+1}")
  print("--"*20)


  X_train, X_test = trainset.iloc[train_idx].drop("rainfall", axis=1), trainset.iloc[test_idx].drop("rainfall", axis=1)
  y_train, y_test = trainset.iloc[train_idx][["rainfall"]], trainset.iloc[test_idx][["rainfall"]]
  ensemble = VotingClassifier(estimators=[
    ("xgb", XGBClassifier(n_estimators=200, max_depth=2)),
    ("et", ExtraTreesClassifier(n_estimators=200, max_depth=2)),
    ("lg", LogisticRegression()),
    ("knn", KNeighborsClassifier(n_neighbors=200)),
    ("svm", SVC(probability=True))

], voting="soft")

  ensemble.fit(X_train, y_train)
  oof_ens[test_idx] = ensemble.predict_proba(X_test)[:, 1]
  pred_ens += ensemble.predict_proba(testset)[:, 1]


pred_ens /= 5


from sklearn.metrics import roc_auc_score
true = trainset.rainfall.values
m = roc_auc_score(true, oof_ens)
print(f"Voting CV Score AUC = {m:.3f}")


submission["rainfall"] = pred_ens
submission.to_csv("submission.csv", index=False)

