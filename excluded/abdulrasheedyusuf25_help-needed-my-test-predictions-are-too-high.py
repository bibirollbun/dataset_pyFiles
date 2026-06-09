import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os
os.listdir("/kaggle/input/")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col=0)
df_original = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
df_raw = df_train.copy()


df_original.shape


df_original.duplicated().sum()


df_original = df_original.dropna(subset=["Listening_Time_minutes"])
df_original.shape


# df_raw = pd.concat([df_raw, df_original])
# df_raw.shape


df_raw.duplicated().sum()


"This dataset has {0} rows, and {1} columns".format(df_raw.shape[0], df_raw.shape[1])


df_raw.head()


for col in df_raw.columns:
  print("{0}: {1}".format(col, df_raw[col].nunique()))
  print(df_raw[col].unique())


missing_values_percentage = df_raw.isnull().sum() / df_raw.shape[0] * 100
missing_values_percentage[missing_values_percentage != 0]


df_raw.columns


num_features = df_raw.select_dtypes(include=np.number).columns.to_list()
cat_features = df_raw.select_dtypes(exclude=np.number).columns.to_list()
num_features.remove("Listening_Time_minutes")
num_features, cat_features


target = "Listening_Time_minutes"
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes = axes.flatten()

for i, col in enumerate(num_features):
  axes[i].scatter(df_raw[col], df_raw[target])
  axes[i].set_xlabel(col)
  axes[i].set_ylabel(target)
  axes[i].set_title("{0} vs {1}".format(col, target))

plt.tight_layout()
plt.show()


df_raw = df_raw.drop(df_raw[ (df_raw["Episode_Length_minutes"] > 300) & (df_raw["Listening_Time_minutes"] > 40) ].index)
df_raw = df_raw.drop(df_raw[ (df_raw["Host_Popularity_percentage"] > 105) | (df_raw["Host_Popularity_percentage"] < 10) ].index)
df_raw = df_raw.drop(df_raw[ (df_raw["Number_of_Ads"] > 10) ].index)
df_raw = df_raw.drop(df_raw[ (df_raw["Guest_Popularity_percentage"] > 105) ].index)



target = "Listening_Time_minutes"
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes = axes.flatten()

for i, col in enumerate(num_features):
  axes[i].scatter(df_raw[col], df_raw[target])
  axes[i].set_xlabel(col)
  axes[i].set_ylabel(target)
  axes[i].set_title("{0} vs {1}".format(col, target))

plt.tight_layout()
plt.show()


df_raw.describe()


import seaborn as sns
from scipy.stats import norm

plt.hist(df_raw["Listening_Time_minutes"], bins=50)
plt.show()


a = df_raw["Listening_Time_minutes"]
a_log = np.log(a+1)
plt.hist(a_log, bins=50)
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
a_scaler = scaler.fit_transform(a.values.reshape(-1, 1))
plt.hist(a_scaler, bins=50)
plt.show()


corr_matrix = df_raw.corr(numeric_only=True)
plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


from scipy.stats import skew

# Select only numerical features
numeric_feats = df_raw.select_dtypes(include=np.number).columns

# Check the skew of all numerical features
skewed_feats = df_raw[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


plt.hist(df_raw["Number_of_Ads"])
plt.xlabel("Number_of_Ads")
plt.show()


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipeline = make_pipeline(StandardScaler())
num_pipeline


sample = df_raw.sample(10)
sample


sample['Episode_Number'] = sample['Episode_Title'].str.extract(r'(\d+)').astype(int)


sample


from sklearn.preprocessing import OrdinalEncoder

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
encoder = OrdinalEncoder(categories=[days_order])
sample["Publication_Day"] = encoder.fit_transform(sample[["Publication_Day"]])


sample


publication_time_order = ['Morning', 'Afternoon', 'Evening', "Night"]
encoder = OrdinalEncoder(categories=[publication_time_order])
sample["Publication_Time"] = encoder.fit_transform(sample[["Publication_Time"]])


sample.head()


Episode_sentiments_order = ['Positive', 'Neutral', 'Negative']
encoder = OrdinalEncoder(categories=[Episode_sentiments_order])
sample["Episode_Sentiment"] = encoder.fit_transform(sample[["Episode_Sentiment"]])


sample.head()


podcast_means = df_raw.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
podcast_means


sample["Podcast_Name"] = sample["Podcast_Name"].map(podcast_means)
sample


sample_corr = sample.corr(numeric_only=True)
plt.figure(figsize=(7, 6))
sns.heatmap(sample_corr, annot=True, cmap="coolwarm")
plt.show()


from colorsys import ONE_SIXTH
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class CategoryTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, podcast_means):
    self.podcast_means = podcast_means

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X.copy()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    publication_time_order = ['Morning', 'Afternoon', 'Evening', "Night"]
    Episode_sentiments_order = ['Positive', 'Neutral', 'Negative']

    #Drop id
    if "id" in X.columns:
        X = X.drop("id", axis=1)

    #Transform Episode Title
    X['Episode_Number'] = X['Episode_Title'].str.extract(r'(\d+)').astype(int)
    X = X.drop("Episode_Title", axis=1)

    #Encode Publication Day
    encode_day = OrdinalEncoder(categories=[days_order])
    X["Publication_Day"] = encode_day.fit_transform(X[["Publication_Day"]])

    #Encode Publication Time
    encode_time = OrdinalEncoder(categories=[publication_time_order])
    X["Publication_Time"] = encode_time.fit_transform(X[["Publication_Time"]])

    #Encode Sentiments
    encode_sentiments = OrdinalEncoder(categories=[Episode_sentiments_order])
    X["Episode_Sentiment"] = encode_sentiments.fit_transform(X[["Episode_Sentiment"]])

    #Transform Podcast names
    X = pd.get_dummies(X, columns=["Podcast_Name"])

    #Transform Genre
    X = pd.get_dummies(X, columns=["Genre"])

    return X


a_sample = df_raw.sample(10)
a_sample_proc = CategoryTransformer(podcast_means).fit_transform(a_sample)
a_sample_proc


cat_pipeline = make_pipeline(CategoryTransformer(podcast_means))
cat_pipeline


processor = ColumnTransformer([
    ("numerical", num_pipeline, num_features),
    ("categorical", cat_pipeline, cat_features)
])
processor


b_sample = df_raw.sample(10)
b_sample_proc = processor.fit_transform(b_sample)
b_sample_proc


train_data = processor.fit_transform(df_raw)
train_data.shape


target = df_raw["Listening_Time_minutes"]
target.shape


from sklearn.model_selection import train_test_split

X_train, X_, y_train, y_ = train_test_split(train_data, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.5, random_state=42)


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)


xgb_pipeline = Pipeline([
    ("preprocessor", processor),
    ("model", xgb_model)
])
xgb_pipeline


df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
df_test["Number_of_Ads"] = df_test["Number_of_Ads"].clip(0, 3)
test_df = processor.fit_transform(df_test)

predictions = xgb_model.predict(test_df)
predictions.max()


sub1 = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub1["Listening_Time_minutes"] = predictions
sub1.to_csv("submission3.csv", index=False)


df_test.head()


test_df[:5]


predictions[:5]


plt.hist(df_test["Number_of_Ads"], bins=50)
plt.show()


yhat_test = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, yhat_test)
rmse = np.sqrt(mse)
rmse


submission = pd.DataFrame({"id": df_test["id"], "Listening_Time_minutes": predictions})
submission.to_csv("submission.csv", index=False)


submission.head()


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data)
    rmse= np.sqrt(-cross_val_score(model, train_data, target, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)


KRR = KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)


model_forest = RandomForestRegressor(n_estimators=1200, min_samples_split=10, min_samples_leaf=1,
                                  max_features='sqrt', max_depth=60, bootstrap=False)


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score = rmsle_cv(model_forest)
# print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score = rmsle_cv(ENet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score = rmsle_cv(GBoost)
# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

