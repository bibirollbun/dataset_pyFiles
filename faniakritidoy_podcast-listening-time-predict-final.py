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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


!pip install category_encoders


#!nvidia-smi  # this should display information about available GPUs
%load_ext cudf.pandas
cuml.__version__


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from IPython import display


from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train.info()


df_test.info()


clipped = df_train[(df_train.Episode_Length_minutes < 150) & (df_train.Listening_Time_minutes < 150)]
fig, axes = plt.subplots(1,2, figsize=(15, 5))
sns.histplot(data = clipped , x="Episode_Length_minutes", ax=axes[0])
sns.histplot(data = clipped, x = "Listening_Time_minutes", ax=axes[1])
plt.tight_layout()
plt.show()


sns.regplot(data = clipped, x = "Episode_Length_minutes", y = "Listening_Time_minutes")


fig, axes = plt.subplots(1,2, figsize=(15, 5))
sns.boxplot(data = df_train, x='Episode_Sentiment', y="Episode_Length_minutes",showfliers=False, ax=axes[0])
sns.boxplot(data = df_train, x='Number_of_Ads', y="Episode_Length_minutes",showfliers=False, ax=axes[1])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2, figsize=(15, 5))
sns.boxplot(data = df_train, x='Publication_Time', y="Episode_Length_minutes",showfliers=False, ax=axes[0])
sns.boxplot(data = df_train, x='Publication_Day', y="Episode_Length_minutes",showfliers=False, ax=axes[1])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2, figsize=(15, 5))
sns.scatterplot(data = df_train, x='Host_Popularity_percentage', y="Episode_Length_minutes",  ax=axes[0])
sns.scatterplot(data = df_train, x='Guest_Popularity_percentage', y="Episode_Length_minutes", ax=axes[1])
plt.tight_layout()
plt.show()


df_train[df_train['Host_Popularity_percentage'] <= 20].shape


sns.heatmap(clipped.select_dtypes(["float64"]).corr(), annot=True, vmin=-1, vmax=1)


def clean_column_names(df):
  """ rename columns to lowercase and remove whitespaces """
  x = df.copy()
  x.rename(columns=lambda c : c.lower().strip(), inplace=True)
  x.rename(columns= lambda x : x.replace(" ", "_"), inplace=True)
  return x

def correct_dtypes(df):
  X = df.copy()
  # there is only 1 missing value in number_of_ads in the train set - drop it
  X = X.dropna(axis=0, subset=['number_of_ads'])
  X.number_of_ads = X.number_of_ads.astype(int)
  return X

def round_fields(df, target = None, wrangle = False,):
  x = df.copy()
  if wrangle:
    """ help with finding hidden duplicates """
    x["episode_length_minutes"] = np.ceil(x.episode_length_minutes)
    x["host_popularity_percentage"] = np.ceil(x.host_popularity_percentage)
    x["guest_popularity_percentage"] = np.ceil(x.guest_popularity_percentage)
  else:
    """ help with fixing the issue: (train_df.Episode_Length_minutes < train_df.Listening_Time_minutes) """
    x["episode_length_minutes"] = np.round(x.episode_length_minutes,1)
    """ clearer data """
    x["host_popularity_percentage"] = np.round(x.host_popularity_percentage, 1)
    x["guest_popularity_percentage"] = np.round(x.guest_popularity_percentage,1)
  if target is not None:
    x[target] = np.round(x[target],1)
  return x

def validate_percentages(df, cols):
  """ check that percentage values are in range 0-100 """
  x = df.copy()
  #return x[(x[col] > 100) | (x[col] < 0)]
  for col in cols:
    x.loc[x[col] > 100, col] = 100
    x.loc[x[col] < 0, col] = 0
  return x

def handle_zero_duration_values(df):
  """ manipulate non sensical values - only exist in train, not test """
  x = df.copy()
  drop_condition = x.episode_length_minutes == 0 # only 1 such value
  # x.drop(index=x[drop_condition].index)
  x = x[~drop_condition]
  return x

def break_down(df):
    X = df.copy()
    pat = r"\s+"
    X["episode_title"] =  X.episode_title.str.strip().str.replace(pat, "").str.split("Episode", n=1, expand=True)[1].astype(int)
    return X


def clean_dataset(df_train, df_test):
  x_train = df_train.copy()
  x_test = df_test.copy()
  """ rename columns to lowercase and remove whitespaces """
  x_train = clean_column_names(x_train)
  x_test = clean_column_names(x_test)
  """ correct dtype for number of adds to 'int'"""
  x_train = correct_dtypes(x_train)
  x_test = correct_dtypes(x_test)
  """ round fields to 1 decimal """
  x_train = round_fields(x_train, "listening_time_minutes")
  x_test = round_fields(x_test)
  """ correct percentages over 100% or negative values """
  x_train = validate_percentages(x_train, ["host_popularity_percentage",  "guest_popularity_percentage"])
  x_test = validate_percentages(x_test,["host_popularity_percentage",  "guest_popularity_percentage"])

  """ break down Episode Title Values """
  x_train = break_down(x_train)
  x_test = break_down(x_test)

  """ zero length values only exist in train, not test """
  x_train = handle_zero_duration_values(x_train)


  return x_train, x_test


features_nom = ["podcast_name", "genre", "publication_day", "publication_time"]
features_ord =["episode_sentiment"]
features_ord_levels={"episode_sentiment" : ["Negative", "Neutral", "Positive"]}


# TODO LATER FOR Tree Based Methods
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}

#df_train['Publication_Day'].map(day_mapping).dtypes


def encode(df, features_nom = features_nom, features_ord_levels = features_ord_levels):
    """ encode nominative and ordinal features """
    # Nominal categories
    for name in features_nom:
        
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        # if "None" not in df[name].cat.categories:
        #     df[name] = df[name].cat.add_categories("None")

    # Ordinal categories
    # Add a None category for missing values
    #features_ord =  {key: ["None"] + value for key, value in features_ord_levels.items()}
    for name, levels in features_ord_levels.items():
        df[name] = df[name].astype(pd.CategoricalDtype(levels,
                                                    ordered=True))
    return df


def load_data():

    # train_df = pd.read_csv('./train.csv', index_col='id', header=0)
    # test_df = pd.read_csv('./test.csv', index_col='id', header=0)

    # leave the ID column when loading to track indices
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", header=0)
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', header=0)

    train, test = clean_dataset(train_df, test_df)

    ### encode
    train = encode(train)
    test = encode(test)

    return train, test


train, test = load_data()


train.isna().sum()


test.isna().sum()


train.drop_duplicates(subset=train.columns[1:], inplace=True)


def get_outlier_limits(df, column, r = 1.5):
  col_iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
  upper_limit = df[column].quantile(0.75) + (r * col_iqr)
  lower_limit = df[column].quantile(0.25) - (r * col_iqr)
  col_outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
  return (lower_limit, upper_limit, col_outliers)


ads_lower_limit, ads_upper_limit, ads_column_outliers = get_outlier_limits(train, "number_of_ads")
print(ads_lower_limit, ads_upper_limit)
train.drop(index=ads_column_outliers.index, inplace=True)
train[(train.number_of_ads < ads_lower_limit) | (train.number_of_ads > ads_upper_limit)]


length_lower_limit, length_upper_limit, length_column_outliers = get_outlier_limits(train, "episode_length_minutes")
length_lower_limit, length_upper_limit
print(length_lower_limit, length_upper_limit)
train = train.drop(index=length_column_outliers.index)
train[(train.episode_length_minutes < length_lower_limit) | (train.episode_length_minutes > length_upper_limit)]


cond = train.listening_time_minutes > train.episode_length_minutes
train.loc[cond, "listening_time_minutes"] = train.loc[cond, "episode_length_minutes"]


# check again for duplicates after imputing
train.duplicated(subset=train.columns[1:]).sum()


duration_proportion_listened = np.where(train.episode_length_minutes.notna(), train.listening_time_minutes / train.episode_length_minutes, np.nan)
train.loc[:, "duration_proportion_listened"] = duration_proportion_listened.round(2)
sns.boxplot(train.duration_proportion_listened, showfliers=False)


train.loc[:, "min_proportion"] = (train.groupby("episode_length_minutes")["duration_proportion_listened"]).transform("min")
train["min_proportion"] = np.power(train["min_proportion"], 4)


sns.catplot(x="number_of_ads", y ="min_proportion", data = train, kind="box", col="episode_sentiment", hue="episode_sentiment", showfliers=False)
plt.show()


display.display(train.groupby("number_of_ads").min_proportion.median())


train.groupby(["podcast_name","genre"]).duration_proportion_listened.median()


sns.relplot(data = train, x='host_popularity_percentage', y="min_proportion", col="episode_sentiment")


# To Target Encode
def podcast_name_genre_group(df):
  X = pd.DataFrame()
  X.loc[:, "podcast_name_genre"] = df.podcast_name.astype("str") + "_" + df.genre.astype("str")
  X["podcast_name_genre"] = X["podcast_name_genre"].astype("category")
  #print(len(X.podcast_name_genre.unique()))
  return X
    
def create_date_time(df):
  X = pd.DataFrame()
  X.loc[:, "date_time"] = df.publication_day.astype(str) + "_" + df.publication_time.astype(str)
  X.date_time = X.date_time.astype("category")
  #print(X.date_time.value_counts())
  return X


train = train.join(podcast_name_genre_group(train))
test = test.join(podcast_name_genre_group(test))
train = train.join(create_date_time(train))
test = test.join(create_date_time(test))


train.columns


def number_of_ads_encode(df):
    """If we don't convert to categories then we are examining the change in response for 1 unit change in ads"""
    df.number_of_ads = df.number_of_ads.astype(pd.CategoricalDtype(categories=sorted(df.number_of_ads.unique()), ordered=True))
    return df


def label_encode(df):

    X = df.copy()

    """ we have already encoded the columns in EDA, so we can now just take the codes """
    # for colname in X.select_dtypes(["category"]):
    #     X[colname] = X[colname].cat.codes

    """ also use the number of adds as category to create dummies """
    number_of_ads_encode(X)

    return X


def create_length_moderator(df):
  X = pd.DataFrame()
  # length = df.copy()["episode_length_minutes"]
  # length -= length.mean()
  # c = pd.qcut(length, 5).map(lambda x : x.right).astype(float)
  X.loc[:, "episode_length_quantiles"] =  pd.qcut(df.episode_length_minutes, q=5).cat.codes
  X.loc[:, "is_long"] = (X.episode_length_quantiles >= 3).astype(int)
  return X


def pca_inspired_continuous_features(df, linear_regression = True):
  X = pd.DataFrame()
  X.loc[:, "popularity"] = (df.host_popularity_percentage) + df.episode_title
  if linear_regression:
    X.loc[:, "negativity"] = (df.number_of_ads.cat.codes / (df.episode_sentiment_f.astype(int))).round(2)
    X.loc[X.negativity == np.inf, "negativity"] = df.loc[X.negativity == np.inf, "number_of_ads"].cat.codes
  else:
      X.loc[:, "negativity"] = (df.number_of_ads_f / (df.episode_sentiment_f.astype(int))).round(2)
      X.loc[X.negativity == np.inf, "negativity"] = df.loc[X.negativity == np.inf, "number_of_ads_f"]
  return X


def create_features(train, test, linear_regression = True, prediction = True):

  # 2. create Moderator for length
  if prediction:

    train_moderator = create_length_moderator(train)
    test_moderator = create_length_moderator(test)

    train = train.join(train_moderator)
    test = test.join(test_moderator)

  # 3. create PCA inspired features
  train_pca = pca_inspired_continuous_features(train, linear_regression)
  test_pca =  pca_inspired_continuous_features(test, linear_regression)

  train = train.join(train_pca)
  test = test.join(test_pca)

  return train, test


def create_num_inputs(df, target = None, linear_regression = True, features_ord = [], features_nom = []):
  """ the length of the episode should act as Moderator because it is included in the response variable """
  # X = df.copy()
  # X.reset_index(drop=True, inplace=True)
  ## return a new DataFrame
  X = pd.DataFrame()

  if linear_regression:
    features_cat = features_ord
  else: # for tree ensemble better to use label-encoding for all
    features_cat = features_nom + features_ord

  for colname in features_cat: # X.select_dtypes(["object", "category"]):
      t = f"{colname}_f"
      # label encoding for ordered
      X[t] = df[colname].cat.codes # TODO: check that levels are preserved!!!!!!!!!!!!
      X[t] += 1 # help for feature engineering

  #############
  # for Linear Regression with Regularization it is better to use one-hot encoding
  # for tree ensemble better to use label-encoding
  ##############
  if linear_regression:
    #features_nom_intersection = np.intersect1d(features_nom_lg, list(X.select_dtypes(["category"]).columns.values))
    # pd.get_dummies(X[features_nom], drop_first=True)
    dummies = pd.get_dummies(df[features_nom], drop_first=True, dtype=int)
    X = pd.concat([X, dummies], axis=1)
  #############
  # we  don't want these changed
  X.loc[:, "id"] = df["id"]
  if target is not None:
    X.loc[:, target] = df[target]

  return X


from category_encoders import QuantileEncoder
from sklearn.model_selection import KFold


# target_encoding for categorical variables with high cardinality
class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation.
    # We also now have one trained encoder on EACH fold.
    def fit_transform(self, X, y, cols):
        # we hav one trained encoder on EACH fold.
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


from category_encoders import SumEncoder


def effect_encode(df, col):
    effect_encoder = SumEncoder(cols=[col], verbose=False)
    df_genre_encoded = effect_encoder.fit_transform(df[[col]])
    return df_genre_encoded


#CHECK EFFECT_ENCODING
effect_encode(train, 'genre')


train_df = train.copy()
test_df = test.copy()


features_lg_ord = ['episode_sentiment'] # helper for pca transforms
features_lg_nom = ['genre','episode_sentiment','number_of_ads'] # add number_of_adds as category so it can be dummied


#label_encode.number_of_ads.cat.categories
train_impute = label_encode(train_df)
test_impute = label_encode(test_df)


train_impute_n = create_num_inputs(train_impute, None, linear_regression = True, features_nom=features_lg_nom, features_ord=features_lg_ord)

test_impute_n = create_num_inputs(test_impute, None, linear_regression = True, features_nom=features_lg_nom, features_ord=features_lg_ord)

train_impute_n.dtypes


train_impute_n.rename(columns = lambda x : x.replace(" ", "_"), inplace=True)
test_impute_n.rename(columns = lambda x : x.replace(" ", "_"), inplace=True)


train_impute_n = train_impute_n.merge(train_impute[["id", "episode_length_minutes", "episode_title", "listening_time_minutes", "number_of_ads", "host_popularity_percentage","podcast_name", "podcast_name_genre", "date_time","duration_proportion_listened", "time_diff"]], on="id")
test_impute_n = test_impute_n.merge(test_impute[["id", "episode_length_minutes", "episode_title", "number_of_ads", "host_popularity_percentage", "podcast_name", "podcast_name_genre", "date_time"]], on="id")


train_impute_n, test_impute_n = create_features(train_impute_n, test_impute_n, linear_regression=True, prediction = False)


encoder= CrossFoldEncoder(QuantileEncoder, m=5, quantile=0.5)
#X_encoded = encoder.fit_transform(train_lg_n, train_lg_n.listening_time_minutes, cols=["podcast_name_genre", "date_time"])
# try with duration_proportion_listened
cols = ["id","podcast_name_genre", "date_time"]
target =  train_impute_n["listening_time_minutes"]
X_encoded = encoder.fit_transform(train_impute_n.loc[:, cols], target, cols=["podcast_name_genre", "date_time"])
X_test_encoded = encoder.transform(test_impute_n.loc[:, cols])


train_impute_n = train_impute_n.join(X_encoded)
test_impute_n = test_impute_n.join(X_test_encoded)


train_impute_n.columns


figure, axes = plt.subplots(1,2, figsize=(10,4))
sns.histplot(data = train_impute_n, x="popularity", ax=axes[0])
sns.countplot(data = train_impute_n, x="negativity", ax=axes[1])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


train_impute_n[["negativity", "popularity", "episode_length_minutes"]].corr()


sns.boxplot(data=train_impute_n, x="negativity", y="time_diff")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# negativity_dummies = pd.get_dummies(train_impute_n["negativity"], drop_first=True, dtype=int)
# negativity_dummies.rename(columns={1.0: "negativity_1",2.0: "negativity_2",3.0: "negativity_3"}, inplace=True)


figure, axes = plt.subplots(1,2,figsize=(6,6))
sns.scatterplot(data = train_impute_n, x = "podcast_name_genre_encoded", y="listening_time_minutes", ax=axes[0])
sns.scatterplot(data = train_impute_n, x = "date_time_encoded", y="listening_time_minutes", ax=axes[1])
plt.tight_layout()
plt.show()


(train_impute_n[train_impute_n.negativity > 0.5]).shape


train_impute_n.loc[:, "is_negative"] = train_impute_n.negativity > 0.5
train_impute_n.is_negative = train_impute_n.is_negative.astype(int)
train_impute_n.loc[:, "is_popular"] = train_impute_n.popularity > 110
train_impute_n.is_popular = train_impute_n.is_popular.astype(int)


train_impute_n.loc[:, "min_proportion"] = (train_impute_n.groupby("episode_length_minutes")["duration_proportion_listened"]).transform("min")
train_impute_n["min_proportion"] = np.power(train_impute_n["min_proportion"], 4)


sns.pairplot(data= train_impute_n[["episode_length_minutes", "listening_time_minutes", "min_proportion"]])


train_impute_n.columns


imp_features =  ['genre_Comedy', 'genre_Education', 'genre_Health', 'genre_Lifestyle',
       'genre_Music', 'genre_News', 'genre_Sports', 'genre_Technology',
       'genre_True_Crime', 'episode_sentiment_Neutral',
       'episode_sentiment_Positive', 'number_of_ads_1', 'number_of_ads_2',
       'number_of_ads_3', 'is_popular', "is_negative", "min_proportion"]


df = train_impute_n[imp_features + ["episode_length_minutes" , "listening_time_minutes"]]
df.head()


X_impute_existing = df[df.episode_length_minutes.notna()]
X_impute_missing = df[df.episode_length_minutes.isna()]


y_impute = X_impute_existing.pop("listening_time_minutes")


X_impute_existing.info()


X_impute_train, X_impute_test, y_impute_train, y_impute_test = train_test_split(X_impute_existing, y_impute, test_size=0.25, random_state=0)


scaler = StandardScaler().fit(X_impute_train[["episode_length_minutes"]])
X_impute_train.loc[:, "episode_length_minutes_scaled"] = scaler.transform(X_impute_train[["episode_length_minutes"]])


X_impute_train.columns


from statsmodels.stats.outliers_influence import variance_inflation_factor


# vif_features = ['genre_Comedy', 'genre_Education', 'genre_Health', 'genre_Lifestyle',
#        'genre_Music', 'genre_News', 'genre_Sports', 'genre_Technology',
#        'genre_True_Crime', 'episode_sentiment_Neutral',
#        'episode_sentiment_Positive', 'number_of_ads_1', 'number_of_ads_2',
#        'number_of_ads_3', 'episode_length_minutes', 'negativity',\
#         'host_popularity_percentage', 'host_popularity_percentage_i', 'number_of_ads_i',
#        'episode_sentiment_f_i', 'negativity_i', 'episode_length_i']
X_vif = X_impute_existing.copy()
vif = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Create a DataFrame with the VIF results for the column names in X.
df_vif = pd.DataFrame(vif, index= X_vif.columns, columns = ['VIF'])

# Display the VIF results.
df_vif


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error


lr = LinearRegression()
lr.fit(X_impute_train.drop("episode_length_minutes", axis=1), y_impute_train)


pd.DataFrame(lr.coef_.ravel(), index= imp_features + ['episode_length_minutes_scaled'])


x_imp_eval = X_impute_train.drop("episode_length_minutes", axis=1)

# Evaluate the model performance on the training data
r_sq = lr.score(x_imp_eval, y_impute_train)
print('Coefficient of determination:', r_sq)
y_pred_impute_train = lr.predict(x_imp_eval)
print('R^2:', r2_score(y_impute_train, y_pred_impute_train))
print('MAE:', mean_absolute_error(y_impute_train, y_pred_impute_train))
print('MSE:', mean_squared_error(y_impute_train, y_pred_impute_train))
print('RMSE:',np.sqrt(mean_squared_error(y_impute_train, y_pred_impute_train)))


X_impute_test.loc[:, "episode_length_minutes_scaled"] = scaler.transform(X_impute_test[["episode_length_minutes"]])


X_impute_test_eval = X_impute_test.drop("episode_length_minutes", axis=1)
y_pred_impute_test = lr.predict(X_impute_test_eval)
print('R^2:', r2_score(y_impute_test, y_pred_impute_test))
print('MAE:', mean_absolute_error(y_impute_test, y_pred_impute_test))
print('MSE:', mean_squared_error(y_impute_test, y_pred_impute_test))
print('RMSE:',np.sqrt(mean_squared_error(y_impute_train, y_pred_impute_train)))


# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_impute_test,
                             'predicted': y_pred_impute_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()


fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
# Draw an x=y line to show what the results would be if the model were perfect
plt.plot([0,120], [0,140], c='red', linewidth=2)
plt.title('Actual vs. predicted');


# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count');


results['residual'].mean()


import statsmodels.api as sm



sm.qqplot(results['residual'], line='s')
plt.title("Q-Q plot of Residuals")
plt.show()


sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
plt.show()


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc

    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'mean_squared_error': 'mean_test_neg_mean_squared_error',
                   'neg_mean_squared_error': 'mean_test_neg_mean_squared_error'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    neg_mean_squared_error = best_estimator_results.mean_test_neg_mean_squared_error


    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'rmse': [np.sqrt(-neg_mean_squared_error)],
                          'best_params': [best_estimator_results.params]
                        })

    return table


def get_val_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    mae = mean_absolute_error(y_test_data, preds)
    mse = mean_squared_error(y_test_data, preds)
    rmse = np.sqrt(mse)
    table = pd.DataFrame({'model': [model_name],
                          'mae': [mae],
                          'rmse': [rmse]
                         })

    return table


#encode ads as category, even if they are int. Leave them as they are for ordinal
train_tr = label_encode(train_df)
test_tr = label_encode(test_df)


train_tr.info()


test_tr.info()


features_tr_ord = ['episode_sentiment', 'number_of_ads']
features_tr_nom = ['genre']


# train_tr_n
train_tr_n = create_num_inputs(train_tr, None, linear_regression = False, features_nom=features_tr_nom, features_ord=features_tr_ord)
# test_tr_n
test_tr_n = create_num_inputs(test_tr, None, linear_regression = False, features_nom=features_tr_nom, features_ord=features_tr_ord)

train_tr_n.dtypes


train_tr_n = train_tr_n.merge(train_tr[["id", "episode_length_minutes", "episode_title", "host_popularity_percentage", "guest_popularity_percentage", "podcast_name_genre", "listening_time_minutes","date_time","duration_proportion_listened"]], on="id")
test_tr_n = test_tr_n.merge(test_tr[["id", "episode_length_minutes", "episode_title", "host_popularity_percentage",  "guest_popularity_percentage", "podcast_name_genre", "date_time"]], on="id")


train_tr_n.info()


test_tr_n.info()


train_tr_n, test_tr_n = create_features(train_tr_n, test_tr_n, linear_regression = False)


train_tr_n.info()


test_tr_n.info()


encoder= CrossFoldEncoder(QuantileEncoder, m=5, quantile=0.5)
#X_encoded = encoder.fit_transform(train_lg_n, train_lg_n.listening_time_minutes, cols=["podcast_name_genre", "date_time"])
# try with duration_proportion_listened
cols = ["id","podcast_name_genre","date_time"]
X_encoded = encoder.fit_transform(train_tr_n.loc[:, cols], train_tr_n.listening_time_minutes, cols=["podcast_name_genre", "date_time"])
X_test_encoded = encoder.transform(test_tr_n.loc[:, cols])


train_tr_n = train_tr_n.join(X_encoded)
test_tr_n = test_tr_n.join(X_test_encoded)


tr_features = ['genre_f', 'episode_sentiment_f', 'number_of_ads_f', 'episode_length_minutes', 'episode_title',\
               'host_popularity_percentage',\
              'popularity', 'negativity','podcast_name_genre_encoded', 'date_time_encoded', 'is_long']


X_tr = train_tr_n[tr_features + ['listening_time_minutes']]
y_tr = X_tr.pop("listening_time_minutes")


X_tr_train, X_tr_test, y_tr_train, y_tr_test = train_test_split(X_tr , y_tr, test_size=0.30, random_state=0)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


!pip install xgboost


# This is the classifier
from xgboost import XGBRegressor

# This is the function that helps plot feature importance
from xgboost import plot_importance


# REF: https://xgboost.readthedocs.io/en/latest/python/python_api.html
# Xy = xgb.DMatrix(X_tr_train, y_tr_train, enable_categorical=True)
# booster = xgb.train({"tree_method": "hist", "max_cat_to_onehot": 5}, Xy)
# SHAP = booster.predict(Xy, pred_interactions=True)

# categorical features are listed as "c"
# print(booster.feature_types)



# alternative 
# XGBoost Setup

# seed1 = 42
# cv = KFold(7, random_state=seed1, shuffle=True)
# pred_test = np.zeros((250000,))

# params = {
#     'objective': 'reg:squarederror', # booster default = 'gbTree', tree_method default = 'hist'
#     'eval_metric': 'rmse',
#     'seed': seed1,
#     'max_depth': 18,
#     'learning_rate': 0.035,
#     'min_child_weight': 60,
#     'reg_alpha': 4,
#     'reg_lambda': 2,
#     'subsample': 0.85,
#     'colsample_bytree': 0.7,
#     'colsample_bynode': 0.5,
#     'device': "cuda"
# }

# def lr_decay(epoch):
#     if epoch < 100:
#         return 0.035
#     else:
#         return 0.015

# callbacks = xgb.callback.LearningRateScheduler(lr_decay)

# # Save learning curves
# train_rmses, valid_rmses = [], []

# for idx_train, idx_valid in cv.split(df_train):
#     X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
#     X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
#     X_test = df_test[X_train.columns].copy()

#     features = df_train.columns
#     encoder1 = TargetEncoder(n_folds=5, seed=seed1, stat="mean")

#     for col in tqdm(features[:20]):
#         X_train[col+'_te1'] = encoder1.fit_transform(X_train[[col]], y_train)
#         X_valid[col+'_te1'] = encoder1.transform(X_valid[[col]])
#         X_test[col+'_te1'] = encoder1.transform(X_test[[col]])

#     for col in tqdm(features[20:]):
#         X_train[col] = encoder1.fit_transform(X_train[[col]], y_train)
#         X_valid[col] = encoder1.transform(X_valid[[col]])
#         X_test[col] = encoder1.transform(X_test[[col]])

#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dval = xgb.DMatrix(X_valid, label=y_valid)
#     dtest = xgb.DMatrix(X_test)
    
 
#     # Before model training
#     evals_result = {}
    
#     model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=1000000,
#     evals=[(dtrain, 'train'), (dval, 'validation')],
#     early_stopping_rounds=30,
#     verbose_eval=500,
#     callbacks=[callbacks],
#     evals_result=evals_result  # <<< Important
# )


#     train_rmse_list = evals_result['train']['rmse']
#     valid_rmse_list = evals_result['validation']['rmse']

#     best_iter = model.best_iteration


#     train_rmses.append(train_rmse_list[best_iter])
#     valid_rmses.append(valid_rmse_list[best_iter])




#     pred_test += np.maximum(0, np.minimum(120, model.predict(dtest)))

#     print("----------------------------------------------------------------")

# pred_test /= 7


xgb = XGBRegressor(objective='reg:squarederror', booster = 'gbtree', random_state = 0)
cv_params = {'device': ['cuda'],
             'max_depth': [4,5,6,7,8],
             'min_child_weight': [1,2,3,4,5],
             'learning_rate': [0.1, 0.2, 0.3],
             'n_estimators': [100, 200],
             'enable_categorical': [True if c == 'number_of_ads' else False for c in X_tr_train.columns ],
             'monotone_constraints': [1 if c == 'episode_length_time' else 0 for c in X_tr_train.columns]
             }
scoring = ['neg_mean_squared_error']

xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='neg_mean_squared_error')


xgb_cv.fit(X_tr_train, y_tr_train)


import pickle
file_name = 'xgb_cv_model.pkl' 


with open(file_name, 'wb') as f:
    pickle.dump(xgb_cv, f)


with open(file_name, 'rb') as f_r:
    xgb_cv = pickle_load(f_r)


xgb_cv.fit(X_train, y_train)



print('Best score XGB CV: ', xgb_cv.best_score_)


# Create xgb model results table
xgb_cv_results = make_results('XGBoost CV', xgb_cv)
xgb_cv_results


get_val_scores(model_name = "XGBoost CV", xgb_cv, X_tr_test, y_tr_test):


 y_preds = xgb_cv.best_estimator_.predict(test_tr_n)

