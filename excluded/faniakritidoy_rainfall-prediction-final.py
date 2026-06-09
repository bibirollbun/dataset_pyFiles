# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install category_encoders
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sklearn.metrics as metrics

from category_encoders import MEstimateEncoder

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
input_files = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        input_files[filename.split(".")[0]] = os.path.join(dirname, filename)
print(input_files)


random.seed(0)


train_df = pd.read_csv(input_files["train"], index_col="id", header=0)
test_df = pd.read_csv(input_files["test"], index_col="id", header=0)


train_df.head()


test_df.head()


train_df.info()


test_df.info()


# sanity check the indices
print('error in index:',train_df.index.isin(test_df.index).sum())
print('missing train set index:',[i for i in range(0,2190) if i not in train_df.index])
print('missing test set index:', [i for i in range(2190,2920) if i not in test_df.index])


# initial check for duplicates
print('total duplicates:',pd.concat([train_df, test_df]).duplicated().sum())


def clean(df):
  df = df.rename(columns={"temparature": "temperature"})
  return df

def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(df[name].mean())
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df
    
def load_data():
  train_df = pd.read_csv(input_files["train"], index_col="id", header=0)
  test_df = pd.read_csv(input_files["test"], index_col="id", header=0)
  #df = pd.concat([train_df, test_df], axis = 0)
  train_df = clean(train_df)
  test_df = clean(test_df)
  test_df = impute(test_df)
  # # check duplicated
  # # df[df.duplicated()].sum()  
  # train_df = remove_outliers(train_df)
  # test_df = remove_outliers(test_df)
  return train_df, test_df

train_df, test_df = load_data()


train_df.info()


test_df.info()


train_df.rainfall.value_counts(dropna=False, normalize = True)

# the target variable is inbalanced but not extremely so. We will not resample


train_df.describe(include='all')


# 4. check which days are represented in the sets
print("missing train days:",[i for i in range(1,366) if i not in train_df.day.unique()])
print("missing test days:", [i for i in range(1,366) if i not in train_df.day.unique()])


plt.figure(figsize=(5,6))
# Assigning a single numeric variable shows its univariate distribution with points randomly “jittered” on the
ax = sns.stripplot(data=train_df, x='day')
plt.show()


X = train_df.copy()
#y = X.pop("rainfall")



#fig, ax = plt.subplots(8,1, sharey=True)
cols = ["day","cloud","humidity","sunshine","pressure","temperature","windspeed", "dewpoint","mintemp"]
for i,col in enumerate(cols):
    axs = sns.catplot(x="rainfall", y=col, data = X, kind="box")    
plt.tight_layout()
plt.show()

# NOTES:
# there are outliers where there is rain with low cloud and rain with high sunshnine
# there are 3 variables that have different distribution for each targe class: cloud, humidity, sunshine


 # check outlier values
train_df_rain = train_df[train_df.rainfall == 1]
train_df_rain[(train_df_rain.sunshine > 10) & (train_df_rain.cloud < 60)]


outlier_df = train_df_rain[(train_df_rain.sunshine > 10) & (train_df_rain.cloud < 60)]
outlier_months = pd.to_datetime(outlier_df.day, format="%j").dt.month.value_counts()
outlier_months.index.rename("Outlier Month Count", inplace=True)
outlier_months


df_month = train_df.copy()
df_month['month'] = pd.to_datetime(df_month['day'], format='%j').dt.month
fig, ax = plt.subplots(figsize=(12, 4))
average_day_rainfall = df_month.groupby(["month"])["rainfall"].mean()
average_day_rainfall.plot(ax=ax)
_ = ax.set(
    title="mean monthly rainfall",
    xticks=[i  for i in range(1,13)],
    #xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="month of year",
    ylabel="mean of rainfall",
)


df_month.groupby(["month"])["rainfall"].sum(), df_month.value_counts("month")


humidity_q1 = train_df.humidity.quantile(0.25)
humidity_q3 = train_df.humidity.quantile(0.75)
humidity_mean = train_df.humidity.quantile(0.5)
humidity_iqr = humidity_q3 - humidity_q1

humidity_lower_bound = humidity_q1 - (1.5 * humidity_iqr)
humidity_upper_bound = humidity_q3 + (1.5 * humidity_iqr)
print(humidity_q1, humidity_mean, humidity_lower_bound,  humidity_upper_bound)
#train_df[(train_df.humidity < humidity_lower_bound) | (train_df.humidity > humidity_upper_bound)].shape


pressure_q1 = train_df.pressure.quantile(0.25)
pressure_q3 = train_df.pressure.quantile(0.74)
pressure_iqr = pressure_q3 - pressure_q1
pressure_lower_bound = pressure_q1 - (1.5 * pressure_iqr)
pressure_upper_bound = pressure_q3 + (1.5 * pressure_iqr)
print(pressure_lower_bound, pressure_upper_bound)


# # mark outliers
def mark_outliers(X):
    df = pd.DataFrame()
    outliers = X[(X.sunshine > 10) & (X.cloud < 60)]
    df["outlier"] = X.index.isin(outliers.index).astype(int)
    # # sanity check
    # train_df.loc[~train_df.index.isin(outliers.index), :].outlier.sum()
    return df


X = train_df.copy()
y = X.pop("rainfall")


# check correlations
X.corrwith(y)
# humidity, cloud, sunshine and windspeed have the strongest correlations


# check pairplots of highest correlations
top_features = ["humidity", "cloud", "sunshine", "windspeed"]
x_top_features = X[top_features].join(y)
sns.pairplot(data=x_top_features, hue="rainfall")
#_ = ax.set(title="Pairplots of highest correlations")
# plt.tight_layout()
plt.show()


sns.lmplot(x="temperature", y="pressure", data=X.join(y), hue="rainfall", col="rainfall")


# check all correlations
plt.figure(figsize=(10,10))
sns.heatmap(train_df.corr().map(round,  ndigits=4), vmin=-1, vmax=1, annot=True)


X.dtypes


discrete_features = X.select_dtypes(include=['int64'])
discrete_features.head()


def get_mutual_info_scores(X, y, discrete_features):
  x_scaled = X.select_dtypes(["float"])
  #x_scaled = StandardScaler().fit_transform(x_scaled)
  x_scaled = (x_scaled - x_scaled.mean(axis = 0)) / x_scaled.std(axis = 0)
  df = pd.concat([x_scaled, discrete_features], axis=1)
  scores = mutual_info_classif(X, y, discrete_features=X.columns.isin(discrete_features.columns))
  return scores

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]


mi_scores = get_mutual_info_scores(X,y,discrete_features)
mi_scores_df = pd.DataFrame(mi_scores, index=X.columns, columns=['mutual_info_score'])
mi_scores_df = mi_scores_df.sort_values(by = 'mutual_info_score', ascending=False)
mi_scores_df


plt.figure(figsize=(8,8))
ax = sns.barplot(y = mi_scores_df.index, x = mi_scores_df.mutual_info_score, orient="h")
ax.set_title("Mutual Info Scores")
plt.show()


def cluster_labels(df, n_clusters = 5):
  X = df.copy()
  kmeans = KMeans(n_clusters=n_clusters, n_init=50)
  x_scaled = X #X.select_dtypes(include=['float'])
  x_scaled = (x_scaled - x_scaled.mean(axis = 0)) / x_scaled.std(axis = 0) # StandardScaler().fit_trasnform(x_scaled)
  X_new = pd.DataFrame()
  X_new.loc[:, "cluster"] = kmeans.fit_predict(x_scaled)
  X_new["cluster"] = X_new["cluster"].astype("category")
  #############
  # add also the distances from the cluester as a new column
  X_cd = kmeans.fit_transform(x_scaled)
  # Label features and join to dataset
  X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])], index = df.index)
  ###########
  return X_new, X_cd, kmeans.inertia_


inertia = []
for i in range(2,11):
  X = train_df.copy() 
  X.drop(columns=["day"], inplace=True)
  y = X.pop('rainfall')
  X_new, X_cd, inertia_ = cluster_labels(X, i)
  inertia.append(inertia_)
plt.plot(range(2,11), inertia)
plt.show()


sil_score = []
for i in range(2,11):
  X = train_df.copy() 
  X.drop(columns=["day"], inplace=True)
  y = X.pop('rainfall')
  X_new, X_cd, inertia_ = cluster_labels(X, i)
  sil_score.append(metrics.silhouette_score(X, X_new.cluster.values))
plt.plot(range(2,11), sil_score)
plt.show()


X = train_df.copy()
X.drop(columns=["day"], inplace=True)
y = X.pop('rainfall')
X_new, X_cd, inertia = cluster_labels(X, 5)
X = X.join([X_new, X_cd])


sns.catplot(data=pd.concat([X,y], axis=1), x="cluster", y="rainfall", hue="cluster", kind="bar")


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            #fit on 1 fold
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            #transform the other fold
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


def target_encode(df,col,target):
  X = df.copy()
  #X['month'] = pd.to_datetime(X['day'], format='%j').dt.month
  y = X.pop(target)
  encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
  X_encoded = encoder.fit_transform(X, y, cols=[col])
  return X_encoded


def day_to_Month(df):
  dt = pd.DataFrame()
  dt['month'] = pd.to_datetime(df['day'], format='%j').dt.month
  return dt


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
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })

    return table


def get_scores(model_name:str, model, X_test_data, y_test_data):
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
    preds_proba = model.best_estimator_.predict_proba(X_test_data)[:,1]
    
    auc = metrics.roc_auc_score(y_test_data, preds_proba) #use preds_proba for roc_auc_score
    
    accuracy = metrics.accuracy_score(y_test_data, preds)
    precision = metrics.precision_score(y_test_data, preds)
    recall = metrics.recall_score(y_test_data, preds)
    f1 = metrics.f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


def mathematical_transforms(X):
    df = pd.DataFrame()
    return df


def interactions(X):
    df = pd.DataFrame()
    df["dewpoint_temperature"] = (X.dewpoint + X.temperature) * X.humidity
    return df


def create_features(df, df_test = None):
    X = df.copy()
    y = X.pop("rainfall")
    discrete_features = X.select_dtypes(include=["int64"])
    mi_scores = get_mutual_info_scores(X, y, discrete_features)

    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        #X_test.pop("rainfall") # there is no response variable in the test set
        X = pd.concat([X, X_test])

    # Mutual Information
    X = drop_uninformative(X, mi_scores)
    X = X.drop(columns=["mintemp", "maxtemp", "winddirection"], errors="ignore")

    # Transformations
    # X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    # X = X.join(counts(X))
    # # X = X.join(break_down(X))
    # X = X.join(group_transforms(X))
    #X = X.join(day_to_Month(X))
    X = X.join(mark_outliers(X))

    #  Clustering
    clusters_labels, cluster_distances, inertia =  cluster_labels(X, n_clusters = 5)
    #X = X.join(clusters_labels)
    X = X.join(cluster_distances)

    # Lesson 5 - PCA
    # X = X.join(pca_inspired(X))
    # X = X.join(pca_components(X, pca_features)) #add the loadings as columns
    # X = X.join(indicate_outliers(X))

    #X = label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["day"]))
    X.drop(columns=["day", "month"], inplace=True, errors="ignore")

    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))
        X_test.drop(columns=["day", "month"], inplace=True, errors = "ignore")

    if df_test is not None:
        return X, X_test
    else:
        return X



df_train, df_test = load_data()
X_train = create_features(df_train)
y_train = df_train.loc[:, "rainfall"]


X_train.head()


tr_x, vl_x, tr_y, vl_y = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify = y_train)


sns.regplot(x="humidity", y="rainfall", data = train_df, logistic = True, ci=None)


sns.regplot(x="cloud", y="rainfall", data = train_df, logistic = True, ci=None)


sns.regplot(x="sunshine", y="rainfall", data = train_df, logistic = True, ci=None)


#sns.regplot(x="dewpoint_temperature", y="rainfall", data = X_train.join(y_train), logistic=True, ci=False)


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from patsy import dmatrices


# Define the predictor variables
test1 = X_train.copy()[["cloud", "humidity", "sunshine", "day_encoded", "dewpoint_temperature"]]
# Add a constant to the model (intercept)
test1 = add_constant(test1)

# Calculate VIF for each feature
datacamp_vif_data = pd.DataFrame()
datacamp_vif_data['Feature'] = test1.columns
datacamp_vif_data['VIF'] = [variance_inflation_factor(test1.values, i) for i in range(test1.shape[1])]
print(datacamp_vif_data)


y1,X1 = dmatrices('rainfall ~ cloud + humidity + sunshine + day_encoded + dewpoint_temperature', data = tr_x.join(tr_y), return_type = 'dataframe')
mod = sm.Logit(y1,X1)
res = mod.fit()
print(res.summary())


y2,X2 = dmatrices('rainfall ~ cloud + sunshine + day_encoded + humidity', data = tr_x.join(tr_y), return_type = 'dataframe')
mod2 = sm.Logit(y2,X2)
res2 = mod2.fit()
print(res2.summary())


res2.params


fitted_values = X2@res2.params
fitted_values[0:5], res2.fittedvalues[0:5]


fig, ax = plt.subplots(1,4, figsize=(12,6))
sns.regplot(x=X2.sunshine, y=res2.fittedvalues, ax=ax[0])
sns.regplot(x=X2.cloud, y=res2.fittedvalues, ax = ax[1])
#sns.regplot(x=X1.humidity, y=res.fittedvalues, ax= ax[2])
#sns.regplot(x=X2.dewpoint_temperature, y=res2.fittedvalues, ax= ax[3]) # this is not linear!!!

plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression


features = ["cloud", "sunshine", "day_encoded", "dewpoint_temperature", "humidity"]


lr = LogisticRegression(penalty = 'l1', max_iter=5000, solver="liblinear",random_state=0)
clf = lr.fit(tr_x[features], tr_y)


pd.DataFrame(clf.coef_, columns = features)


pd.Series([clf.score(tr_x[features], tr_y)], name = "mean accuracy", index=["clf"])


y_pred = clf.predict(vl_x[features])
metrics.roc_auc_score(y_pred, vl_y)


metrics.RocCurveDisplay.from_estimator(clf, vl_x[features], vl_y)


clf_conf_m = metrics.confusion_matrix(vl_y, y_pred, labels=clf.classes_)
disp = metrics.ConfusionMatrixDisplay(clf_conf_m, display_labels = clf.classes_)
disp.plot(values_format='')


clf2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000,  random_state = 0)
scoring = scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

x_tr_scaled = tr_x.copy()
x_tr_scaled = (x_tr_scaled - x_tr_scaled.mean(axis = 0)) / x_tr_scaled.std(axis = 0) # StandardScaler().fit_trasnform(x_scaled)

clf2_cv = GridSearchCV(clf2, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, scoring=scoring, cv=5, refit="roc_auc")
clf2_cv.fit(x_tr_scaled, tr_y)


clf2_cv.best_score_


pd.DataFrame(clf2_cv.best_estimator_.coef_[0], clf2_cv.feature_names_in_, columns=["coef"])


make_results('logistic regression cv', clf2_cv, 'auc')


x_vl_scaled = vl_x.copy()
x_vl_scaled = (x_vl_scaled - x_vl_scaled.mean(axis = 0)) / x_vl_scaled.std(axis = 0) # StandardScaler().fit_trasnform(x_scaled)
get_scores('logistic_regression validation scores', clf2_cv, x_vl_scaled, vl_y)


metrics.RocCurveDisplay.from_estimator(clf2_cv.best_estimator_, x_vl_scaled, vl_y)


lr_conf_m = metrics.confusion_matrix(vl_y, clf2_cv.best_estimator_.predict(x_vl_scaled), labels = clf2_cv.best_estimator_.classes_)
disp = metrics.ConfusionMatrixDisplay(lr_conf_m, display_labels = clf2_cv.best_estimator_.classes_)
disp.plot(values_format='')


FPR = 17/(74+17)
TPR = 313/(313+17)
TPR,FPR 


target_names = ['Predicted would not rain', 'Predicted would rain']
print(metrics.classification_report(vl_y, clf2_cv.best_estimator_.predict(x_vl_scaled), target_names=target_names))


df_train, df_test = load_data()
X_train = create_features(df_train)
y_train = df_train.loc[:, "rainfall"]


tr_x, vl_x, tr_y, vl_y = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)


tr_x.head()


from sklearn.tree import DecisionTreeClassifier


tree1 = DecisionTreeClassifier(random_state=0)
cv_params = {'max_depth':[4, 6, 8, 12, None],
             'min_samples_leaf': [2, 5, 6],
             'min_samples_split': [2, 4, 6]
             }
scoring = scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv1 = GridSearchCV(tree1, cv_params, scoring=scoring, refit='roc_auc', cv=5)


cv1.fit(tr_x, tr_y)


cv1.best_score_


make_results('decision tree fit', cv1, 'auc')


pd.DataFrame(cv1.best_estimator_.feature_importances_, cv1.best_estimator_.feature_names_in_, columns=["importance"]).sort_values(by="importance", ascending = False)


get_scores('decision tree validation scores', cv1, vl_x, vl_y)


metrics.RocCurveDisplay.from_estimator(cv1.best_estimator_,vl_x, vl_y)


cv1_conf_m = metrics.confusion_matrix(vl_y, cv1.best_estimator_.predict(vl_x), labels = cv1.best_estimator_.classes_)
disp = metrics.ConfusionMatrixDisplay(cv1_conf_m, display_labels = cv1.best_estimator_.classes_)
disp.plot(values_format='')


from sklearn.ensemble import RandomForestClassifier


rf1 = RandomForestClassifier(random_state=0)
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }
cv2 = GridSearchCV(rf1, cv_params, scoring=scoring, refit='roc_auc', cv=5, n_jobs=-1)


cv2.fit(tr_x, tr_y)


cv2.best_estimator_, cv2.best_score_


pd.DataFrame(cv2.best_estimator_.feature_importances_, cv2.best_estimator_.feature_names_in_, columns=["importance"]).sort_values(by="importance", ascending=False)


make_results('random forest fit', cv2, "auc")


get_scores("random forest validation scores", cv2, vl_x, vl_y)


metrics.RocCurveDisplay.from_estimator(cv2.best_estimator_, vl_x, vl_y)


cv2_conf_m = metrics.confusion_matrix(vl_y, cv2.best_estimator_.predict(vl_x), labels = cv2.best_estimator_.classes_)
disp = metrics.ConfusionMatrixDisplay(cv2_conf_m, display_labels = cv2.best_estimator_.classes_)
disp.plot(values_format = '')


from xgboost import XGBClassifier, plot_importance


xgb = XGBClassifier(random_state=0, objective='binary:logistic')


cv_params = {
    'max_depth': [1,2,4],
    'min_child_weight': [3],
    'learning_rate': [0.01],
    'n_estimators': [500],
    'subsample': [0.7],
    'colsample_bytree': [0.7]
}
cv3 = GridSearchCV(xgb, cv_params, scoring=scoring, refit='roc_auc', cv=5)


xgb_cv = cv3.fit(tr_x, tr_y)


xgb_cv.best_estimator_, xgb_cv.best_score_


make_results('xgboost fit', xgb_cv, 'auc')


pd.DataFrame(xgb_cv.best_estimator_.feature_importances_, xgb_cv.best_estimator_.feature_names_in_, columns=["importance"]).sort_values(by="importance", ascending=False)


get_scores('xgboost validation scores', xgb_cv, vl_x, vl_y)


xgb_cv_onf_m = metrics.confusion_matrix(vl_y, xgb_cv.best_estimator_.predict(vl_x), labels=xgb_cv.best_estimator_.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=xgb_cv_onf_m, display_labels=xgb_cv.best_estimator_.classes_)
disp.plot(values_format='')
plt.show()


metrics.RocCurveDisplay.from_estimator(
   xgb_cv.best_estimator_, vl_x, vl_y)


plot_importance(xgb_cv.best_estimator_)


df_train, df_test = load_data()
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "rainfall"]


cv_params = {
    'max_depth': [1,2,4],
    'min_child_weight': [3],
    'learning_rate': [0.01],
    'n_estimators': [500],
    'subsample': [0.7],
    'colsample_bytree': [0.7]
}
scoring = scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

xgb_fin = XGBClassifier(random_state=42, objective='binary:logistic')


cv_fin = GridSearchCV(xgb_fin, cv_params, scoring=scoring, refit='roc_auc', cv=5)


xgb_cv_fin = cv_fin.fit(X_train, y_train)


make_results('xgboost', xgb_cv_fin, 'auc')


y_pred = xgb_cv_fin.best_estimator_.predict(X_test)
y_pred_proba = xgb_cv_fin.best_estimator_.predict_proba(X_test)[:, 1]
submission_df = pd.DataFrame({'id': X_test.index.astype(int), 'rainfall': y_pred_proba})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

