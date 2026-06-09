!pip install category_encoders


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import typing


from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from sklearn import metrics


import gc
import torch


warnings.filterwarnings('ignore')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        paths[filename] = path
        print(path)
paths
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!head -n 10 {paths['train.csv']}


!head -n 10 {paths['test.csv']}


train_df = pd.read_csv(paths['train.csv'], index_col="id", header = 0)
test_df = pd.read_csv(paths['test.csv'], index_col = "id", header = 0)


train_df.info()


test_df.info()


train_df.index.is_unique, test_df.index.is_unique


train_df['Fertilizer Name'].value_counts(dropna=False, normalize=True).apply(lambda x : np.round(x,3))


train_df.duplicated().sum(), test_df.duplicated().sum()


train_df.describe(include="object")


test_df.describe(include="object")


train_df.describe()


test_df.describe()


plt.figure(figsize=(8,5))
sns.countplot(x = "Fertilizer Name", hue = "Fertilizer Name", data = train_df,
              order = train_df["Fertilizer Name"].value_counts().index)
plt.gca().get_legend().remove()
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 10))
sns.histplot(x = "Crop Type", hue="Fertilizer Name", discrete = True,
             multiple = "stack", data = train_df)
plt.xticks(rotation=90)
plt.show()


crop_fertilizer_heat_table = train_df.groupby("Crop Type")["Fertilizer Name"]\
.value_counts(normalize=True, dropna= False).apply(lambda x: np.round(x,3))
plt.figure(figsize=(10, 8))
sns.heatmap(crop_fertilizer_heat_table.unstack(), annot=True, fmt=".3f",
            cmap="Blues")
plt.show()


plt.figure(figsize = (10, 5))
soil_fertilizer_heat_table = train_df.groupby("Soil Type")["Fertilizer Name"].\
value_counts(normalize = True, dropna = False).apply(lambda x: np.round(x,3))
sns.heatmap(soil_fertilizer_heat_table.unstack(), annot = True, fmt = ".3f", cmap = "Blues")
plt.show()


ct = pd.crosstab( # default aggregation = frequency table
    index = [train_df["Soil Type"], train_df["Crop Type"]],
    columns = train_df["Fertilizer Name"],
    normalize = "index" # normalize by each row !!!
)
#ct


# plt.figure(figsize=(12, 15))
# sns.heatmap(ct, annot=False, cmap="Blues")
# plt.show()


box_n = sns.catplot(data = train_df, x = "Soil Type", y = "Nitrogen", kind = "box",
                   col = "Fertilizer Name", col_wrap=3)
for ax in box_n.axes.flat:
  fertilizer_by_col = ax.title.get_text().split("=")[1].strip()
  ax.plot(train_df[train_df["Fertilizer Name"] == fertilizer_by_col]
          .groupby("Soil Type")["Nitrogen"].mean(), 'ro')
  # overall statistics without the fertilizer grouping
  ax.plot(train_df.groupby("Soil Type")["Nitrogen"].mean(), 'bo')
  ax.plot(train_df.groupby("Soil Type")["Nitrogen"].median(), 'yo')
plt.show()


boxes_p = sns.catplot(data = train_df, x = "Soil Type", y = "Potassium", kind = "box", col = "Fertilizer Name", col_wrap=3)
for ax in boxes_p.axes.flat:
  fertilizer_by_col = ax.title.get_text().split("=")[1].strip()
  ax.plot(train_df[train_df["Fertilizer Name"] == fertilizer_by_col]
          .groupby("Soil Type")["Potassium"].mean(), 'ro')
  # overall statistics without the fertilizer grouping
  ax.plot(train_df.groupby("Soil Type")["Potassium"].mean(), 'bo')
  ax.plot(train_df.groupby("Soil Type")["Potassium"].median(), 'yo')


boxes_ph = sns.catplot(data = train_df, x = "Soil Type", y = "Phosphorous", kind = "box", col = "Fertilizer Name", col_wrap=3)
for ax in boxes_ph.axes.flat:
  fertilizer_by_col = ax.title.get_text().split("=")[1].strip()
  ax.plot(train_df[train_df["Fertilizer Name"] == fertilizer_by_col]
          .groupby("Soil Type")["Phosphorous"].mean(), 'ro')
  # overall statistics without the fertilizer grouping
  ax.plot(train_df.groupby("Soil Type")["Phosphorous"].mean(), 'bo')
  ax.plot(train_df.groupby("Soil Type")["Phosphorous"].median(), 'yo')


plt.figure(figsize=(6,4))
sns.boxplot(x="Fertilizer Name", y="Temparature", hue="Fertilizer Name", data=train_df)
plt.gca().get_legend().remove()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="Fertilizer Name", y="Humidity", hue="Fertilizer Name", data=train_df)
plt.gca().get_legend().remove()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="Fertilizer Name", y="Moisture", hue="Fertilizer Name", data=train_df)
plt.gca().get_legend().remove()
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 10))
sns.histplot(x = "Temparature", hue = "Fertilizer Name", discrete = True,
             multiple = "stack",
             kde = True, data = train_df)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 10))
sns.histplot(x = "Moisture", hue = "Fertilizer Name", discrete = True,
             multiple = "stack",
             kde = True, data = train_df)
plt.tight_layout()
plt.show()


from sklearn.feature_selection import mutual_info_classif
from pandas.api.types import is_integer_dtype


def make_mi_scores(df, y):
    X = df.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


X_mi = train_df.copy()
y_mi = X_mi.pop("Fertilizer Name") # unsupervised feature selection
mi_scores = make_mi_scores(X_mi, y_mi)
mi_scores


# by default join uses the index of "other"
# features_l = train_df.select_dtypes(include="int64").join(train_df["Fertilizer Name"])
# sns.pairplot(features_l, hue="Fertilizer Name")
# plt.gca().get_legend().remove()
# plt.tight_layout()
# plt.show()


def clean(df: pd.DataFrame) \
-> typing.Tuple[pd.DataFrame]:
    col_corrector: function = \
    lambda x : 'temperature' if x.strip() == 'Temparature' else x.strip()
    col_renamer: function = \
    lambda x : "_".join(col_corrector(x).split(" ")).lower()
    df.rename(columns = col_renamer, inplace = True)
    return df


features_nom = ["soil_type", "crop_type", "fertilizer_name"]
features_ord = {}
ordered_levels = {key: ["None"] + value for key, value in
                  features_ord.items()}


from pandas.api.types import CategoricalDtype

def encode(df: pd.DataFrame, features_nom = features_nom,
           ordered_levels = ordered_levels) -> pd.DataFrame:
  # nominal features
  for name in features_nom:
    if name not in df.columns:
      continue
    df[name] = df[name].astype("category")
    # if "None" not in df[name].cat.categories:
    #   df[name] = df[name].cat.add_categories("None")
    # ordinal features
  for name, levels in ordered_levels.items():
    if name not in df.columns:
      continue
    df[name] = df[name].astype(CategoricalDtype(categories = levels,
                                                ordered = True))
  return df


def load() -> typing.Tuple[pd.DataFrame]:
    train = pd.read_csv(paths['train.csv'], index_col = "id", header = 0)
    test = pd.read_csv(paths['test.csv'], index_col = "id", header = 0)
    target = (train.pop('Fertilizer Name')).to_frame()
    df = pd.concat([train, test], axis = 0)
    df = clean(df)
    target = clean(target)
    df = encode(df)
    target = encode(target)
    return df.iloc[train.index, :].join(target), df.iloc[test.index,:]


train, test = load()


def sort_labels_by_prob(probs: typing.List[float], labels: np.array) -> \
typing.List:
  # Pair the lists and sort based on the ordering_list values (descending)
  paired_lists = zip(labels, probs)
  sorted_pairs = sorted(paired_lists, key=lambda pair: pair[1], reverse=True)
  # Extract the ordered list
  ordered_list = [pair[0] for pair in sorted_pairs]
  return ordered_list


def get_predicted_labels(probs:np.array, classes: np.array)\
 -> np.array:
  predicted_labels = []
  for i in range(probs.shape[0]):
    predicted_labels.append(sort_labels_by_prob(probs[i,:], classes)[0:3])
  return np.array(predicted_labels)


def get_map3_score(true_vals: pd.Series, probs:np.array,
                   classes: np.array) -> int:
  predicted_labels = get_predicted_labels(probs, classes)
  scores = []
  for index, result in enumerate(predicted_labels):
    true_val = true_vals.values[index]
    marked_labels = np.where(result == true_val, 1, 0)
    scores.append(marked_labels @ np.array([1, 1/2, 1/3]))
  return np.mean(scores)


def make_submission_file(probs:np.array, test_df:pd.DataFrame, \
                         classes: np.array):
  # get the id column back
  # test.reset_index(inplace = True)
  y_test = get_predicted_labels(probs, classes)
  y_test_s = [" ".join(item) for item in y_test]
  pd.DataFrame({"id": test_df.index, "Fertilizer Name": y_test_s})\
  .to_csv("submission.csv", index = False)


def get_cv_score(train_df: pd.DataFrame, target:pd.Series,
                 test: pd.DataFrame, clf:typing.Any, best_params: dict = None) \
 -> typing.Tuple[np.array, np.array, typing.List, typing.List]:

  k_folds = 5

  model_label_encoder = create_label_encoder_instance()
  y_enc = model_label_encoder.fit_transform(target)
  encoder_classes = model_label_encoder.classes_

  oof = np.zeros((train_df.shape[0], encoder_classes.shape[0]))
  test_preds = np.zeros((test.shape[0], encoder_classes.shape[0]))
  scores = []
  models = []

  skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

  for train_indx, val_indx in skf.split(train_df, y_enc):
    X_train_cv, X_val_cv = train_df.iloc[train_indx], train_df.iloc[val_indx]
    y_train_cv, y_val_cv = y_enc[train_indx], y_enc[val_indx]
    #model = XGBClassifier(**xgb_cl_params)
    model = clone(clf)
    if best_params is not None:
        model.set_params(**best_params, early_stopping_rounds=100)
    model.fit(X_train_cv, y_train_cv,
              eval_set=[(X_train_cv, y_train_cv),(X_val_cv, y_val_cv)],
             )
    oof[val_indx] = model.predict_proba(X_val_cv)
    scores.append(get_map3_score(target.iloc[val_indx], oof[val_indx],\
                                 encoder_classes))
    test_preds += (model.predict_proba(test) / k_folds)
    models.append(model)
    del model, X_train_cv, y_train_cv, X_val_cv, y_val_cv
    gc.collect()

  return oof, test_preds, scores, encoder_classes, models


mp3_scorer = metrics.make_scorer(metrics.top_k_accuracy_score, k=3)


def create_label_encoder_instance() -> LabelEncoder:
  return LabelEncoder()


def split_df_by_index(df: pd.DataFrame, train_indx:pd.Series, test_indx: pd.Series) -> \
typing.Tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[train_indx, :], df.iloc[test_indx, :]


X_train = train.copy()
X_test = test.copy()


X_test.head()


# Label encoding is good for XGBoost and RandomForest, but one-hot
# would be better for models like Lasso or Ridge
def label_encode(df: pd.DataFrame, test:pd.DataFrame = None, \
                 linear = False, drop_target = None) -> typing.Tuple[
                     pd.DataFrame, pd.DataFrame]:
    X = df.copy()
    if drop_target is not None:
        X.pop(drop_target)
    if test is not None:
        X = pd.concat([X, test], axis = 0)
    else:
      test = pd.DataFrame()
    if linear == True:
        df_dummy = pd.get_dummies(X.select_dtypes("category"), dtype = "int",
                              drop_first = True)
        return df_dummy.iloc[df.index, :], df_dummy.iloc[test.index]
    for colname in X.select_dtypes(["category"]):
       # instead of col.factorize to preserve order
        X[colname] = X[colname].cat.codes
    return X.iloc[df.index, :], X.iloc[test.index]


from sklearn import naive_bayes


bayes_features = ['temperature', 'humidity', 'moisture', 'soil_type',
                  'crop_type', 'nitrogen', 'potassium', 'phosphorous',
                  'fertilizer_name']


X_train_bayes = X_train[bayes_features].copy()
y_bayes = X_train_bayes.pop('fertilizer_name')
X_test_bayes = X_test[bayes_features[:-1]].copy()


X_train_bayes, X_test_bayes =  label_encode(X_train_bayes, test = X_test_bayes)


X_test_bayes.info()


X_bayes_train, X_bayes_val, y_bayes_train, y_bayes_val = \
train_test_split(X_train_bayes, y_bayes, test_size = 0.3, stratify = y_bayes, random_state = 0)


nb = naive_bayes.MultinomialNB()
nb.fit(X_bayes_train, y_bayes_train)


y_bayes_pred_train = nb.predict_proba(X_bayes_train)
get_map3_score(y_bayes_train, y_bayes_pred_train, nb.classes_)


y_bayes_pred_val = nb.predict_proba(X_bayes_val)
get_map3_score(y_bayes_val, y_bayes_pred_val, nb.classes_)


plt.figure(figsize=(15,8))
val_preds = get_predicted_labels(y_bayes_pred_val, nb.classes_)
cm_preds = []
for index,prediction in enumerate(val_preds):
    true_val = y_bayes_val.values[index]
    if true_val in prediction:
        cm_preds.append(true_val)
    else:
        cm_preds.append(prediction[0]) # or whichever other value
cm = metrics.confusion_matrix(y_bayes_val.values, cm_preds, labels = nb.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = nb.classes_)
disp.plot()
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression


lr_features = ['temperature', 'humidity', 'moisture', 'soil_type',
                'crop_type', 'nitrogen', 'potassium', 'phosphorous', 'fertilizer_name']


X_train_lr = X_train[lr_features].copy()
X_test_lr = X_test[lr_features[:-1]].copy()


y_lr = X_train_lr.pop("fertilizer_name")
 # we will need the classes so use LabelEncoder
lr_label_encoder = create_label_encoder_instance()
y_lr_enc = lr_label_encoder.fit_transform(y_lr)


X_test_lr.info()


# Import variance_inflation_factor from statsmodels.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a subset of the data with the continous independent variables.
X = X_train_lr[['temperature','humidity', 'moisture', 'nitrogen', 'phosphorous',
                'potassium']]

# Calculate the variance inflation factor for each variable.
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Create a DataFrame with the VIF results for the column names in X.
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])

# Display the VIF results.
df_vif


# df_lr = pd.concat([X_train_lr, X_test_lr], axis = 0)
# df_lr = encode(df_lr, features_nom = df_lr.columns)
# X_train_lrc = df_lr.iloc[X_train_lr.index, :]
# X_test_lrc = df_lr.iloc[X_test_lr.index, :]
# X_train_lrc, X_test_lrc = label_encode(X_train_lrc, test = X_test_lrc,
#                                      linear = True)


X_train_lr, X_test_lr = label_encode(X_train_lr, test = X_test_lr)


X_train_lr.info()


scaler = StandardScaler()
X_train_lr_scaled = scaler.fit_transform(X_train_lr)
X_test_lr_scaled = scaler.transform(X_test_lr)


lr_model = LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    class_weight = "balanced",
    random_state = 42,
    multi_class = 'multinomial')
param_grid = {
    "max_iter": [500, 5000, 10000],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}
mp3_scorer = metrics.make_scorer(metrics.top_k_accuracy_score,k=3)
scoring = {'top-k': mp3_scorer}
# for integer/None values of cv, if the estimator is a classifier,
# then stratifiedKFold is used. Otherwise kFold is used.
cv = GridSearchCV(lr_model, param_grid=param_grid, scoring = scoring, cv=5,
                  refit = 'top-k', n_jobs=-1) #-1 means using all processors


%%time
cv.fit(X_train_lr_scaled, y_lr_enc)
# cv.fit(X_train_lrc, y_lr)


lr_best_estimator = cv.best_estimator_


cv.best_params_


#lr_model.set_params(**cv.best_params_)


# oof_probs, test_probs, scores, encoder_classes, models = get_cv_score(
#     pd.DataFrame(X_train_lr_scaled, columns = X_train_lr.columns), y_lr,
#     pd.DataFrame(X_test_lr_scaled, columns = X_test_lr.columns),
#     lr_model, cv.best_params_)
oof_probs, test_probs, scores, encoder_classes, models = \
get_cv_score(X_train_lr_scaled, y_lr, X_test_lr_scaled, lr_model, cv.best_params_)
overall_Score = get_map3_score(pd.Series(y_lr), oof_probs, encoder_classes)
overall_Score, np.mean(scores)


make_submission_file(est_probs, X_test_lr, lr_label_encoder.classes_)


!pip install XGBoost


from xgboost import XGBClassifier, plot_importance, plot_tree, to_graphviz,\
DMatrix, train
from sklearn.pipeline import Pipeline
import shap


values_to_remove = [
  'id',
  # 'npk_formula',
  # 'fertilizer_nitrogen',
  # 'fertilizer_phosphorous',
  # 'fertilizer_potassium',
  # 'soil_type-crop_type'
 ]
xgb_features = [f for f in X_train.columns.tolist()
 if f not in values_to_remove]


X_train_xgb = X_train[xgb_features].copy()
y_xgb = X_train_xgb.pop("fertilizer_name")
X_test_xgb = X_test[X_train_xgb.columns].copy()


xgb_cl_params = dict({
    "objective": "multi:softprob",
    "num_class": y_xgb.unique().shape[0],
    'device': 'gpu' if torch.cuda.is_available() else 'cpu',
    'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
    'enable_categorical': True,
    'random_state': 42,
    'n_jobs':-1
  })


!pip install optuna


import optuna
from functools import partial


def objective(X:pd.DataFrame, y: pd.Series, useDmatrix: bool,
              trial:optuna.Trial):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        subsample=trial.suggest_float("subsample", 0.2, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    )

    xgb_label_encoder = create_label_encoder_instance()
    y_enc = xgb_label_encoder.fit_transform(y)

    y_pred_train = np.zeros((X.shape[0], xgb_label_encoder.classes_.shape[0]))

    if useDmatrix:
      Xy = DMatrix(X, label = y_enc, enable_categorical=True)
      booster = train({**xgb_cl_params,**xgb_params,
                       "max_cat_to_onehot": 30,
                      },
                      dtrain=Xy)
      y_pred_train = booster.predict(DMatrix(X, enable_categorical=True))
    else:
      xgb = XGBClassifier(**xgb_cl_params)
      xgb.set_params(**xgb_params)
      xgb.fit(X, y_enc)
      #xgb = xgb_pipe.steps[-1][-1]
      y_pred_train = xgb.predict_proba(X)

    return get_map3_score(pd.Series(y), y_pred_train, \
                          xgb_label_encoder.classes_)


def get_optuna_best_params (X:pd.DataFrame, y:pd.Series, useDmatrix = False)\
 -> dict:
  # we want to maximize the objective function
  objective_func = partial(objective, X, y, useDmatrix)
  study = optuna.create_study(direction="maximize")
  study.optimize(objective_func, n_trials=20)
  return study.best_params


xgb_params = get_optuna_best_params (X_train_xgb, y_xgb)
xgb_params


# xgb_params = {'max_depth': 10, 'learning_rate': 0.011356253705016433, 'n_estimators': 264, 
#               'min_child_weight': 1, 'colsample_bytree': 0.8176564521937253, 
#               'subsample': 0.9624719804470158, 'reg_alpha': 0.0034014693734025396, 
#               'reg_lambda': 0.00010490920491283931}


#xgb_model = XGBClassifier(**xgb_cl_params)
oof_probs, test_probs, scores, encoder_classes, models = get_cv_score(
    X_train_xgb, y_xgb,X_test_xgb,XGBClassifier(**xgb_cl_params), xgb_params)
overall_Score = get_map3_score(pd.Series(y_xgb), oof_probs, encoder_classes)
overall_Score, np.mean(scores)


best_model = models[np.argmax(scores)]


# Loss curves for the last fold
results = best_model.evals_result()
plt.plot(results['validation_0']['mlogloss'], label='Train')
plt.plot(results['validation_1']['mlogloss'], label='Val')
plt.legend()
plt.show()


plt.figure(figsize=(20,8))
plot_importance(best_model)
plt.tight_layout()
plt.show()


make_submission_file(test_probs, test, encoder_classes)


X_train_partition = X_train.copy()
X_test_partition = X_test.copy()


y_partition = X_train_partition.pop("fertilizer_name")
partition_endoder = create_label_encoder_instance()
y_partition_enc = partition_endoder.fit_transform(y_partition)


features_nom = ['temperature', 'humidity', 'moisture', 'nitrogen',
                'phosphorous', 'potassium']


df_partition = encode(pd.concat([X_train_partition, X_test_partition], axis = 0))
X_train_partition = df_partition.iloc[X_train_partition.index, :]
X_test_partition = df_partition.iloc[X_test_partition.index, :]


X_train_partition.info()


X_test_partition.info()


xgb_params = get_optuna_best_params (X_train_partition, y_partition, True)
xgb_params


xgb_params = {'max_depth': 10,
              'learning_rate': 0.0007885738370116495,
              'n_estimators': 169,
              'min_child_weight': 3,
              'colsample_bytree': 0.5698160826409262,
              'subsample': 0.6061724590130845,
              'reg_alpha': 0.0031632270331219212,
              'reg_lambda': 0.00014338836784782313
              }


def get_cv_score_with_booster(train_df: pd.DataFrame, target:pd.Series,
                 test: pd.DataFrame, best_params: dict = None) -> \
typing.Tuple[np.array, np.array, typing.List, typing.List, typing.List]:
  k_folds = 5
  model_label_encoder = create_label_encoder_instance()
  y_enc = model_label_encoder.fit_transform(target)
  encoder_classes = model_label_encoder.classes_

  oof = np.zeros((train_df.shape[0], encoder_classes.shape[0]))
  test_preds = np.zeros((test.shape[0], encoder_classes.shape[0]))
  scores = []
  SHAP = []
  models = []
  skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

  for train_indx, val_indx in skf.split(train_df, y_enc):
    X_train_cv, X_val_cv = train_df.iloc[train_indx],\
    train_df.iloc[val_indx]
    y_train_cv, y_val_cv = y_enc[train_indx], y_enc[val_indx]
    #model = XGBClassifier(**xgb_cl_params)

    Xy = DMatrix(X_train_cv, label = y_train_cv, enable_categorical=True)
    booster = train(params = {**xgb_cl_params, **xgb_params,
                              "max_cat_to_onehot": 30}, dtrain = Xy)

    oof[val_indx] = booster.predict(DMatrix(X_val_cv, enable_categorical=True))
    scores.append(get_map3_score(target.iloc[val_indx], oof[val_indx],\
                                  encoder_classes))
    # SHAP.append(booster.predict(DMatrix(X_val_cv, enable_categorical=True),
    #                        pred_contribs=True))

    test_preds += (booster.predict(
        DMatrix(test, enable_categorical= True)) / k_folds)
    models.append(booster)
    #del booster, X_train_cv, y_train_cv, X_val_cv, y_val_cv
    #gc.collect()
  return oof, test_preds, scores, encoder_classes, models


oof, test_preds, map3_scores, encoder_classes, models = get_cv_score_with_booster(
    X_train_partition, y_partition, X_test_partition, xgb_params)
overall_Score = get_map3_score(pd.Series(y_partition), oof, encoder_classes)
overall_Score, np.mean(map3_scores)


best_model = models[np.argmax(map3_scores)]


plot_importance(best_model)
plt.tight_layout()
plt.show()


# Assuming you have a single booster model from your cross-validation loop
# Use the best performing model
explainer = shap.TreeExplainer(best_model)

# Select a sample of the validation data for plotting to avoid long computation
sample_X_val_cv = X_train_partition.sample(n=1000, random_state=42)

# Convert categorical columns to integer codes for SHAP
for col in sample_X_val_cv.select_dtypes(include='category').columns:
    sample_X_val_cv[col] = sample_X_val_cv[col].cat.codes

# Calculate SHAP values for the sample
shap_values = explainer.shap_values(sample_X_val_cv)

# Visualize the feature importance across all instances in the sample
shap.summary_plot(shap_values, sample_X_val_cv, plot_type="bar",
                  class_names=encoder_classes)

# You can also visualize the contribution for a single instance
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:],
#/sample_X_val_cv.iloc[0,:])


predicted_clases = get_predicted_labels(oof, encoder_classes)
predicted_clases


for i in range(3):
  print(f"Rank: {i+1}")
  plt.figure(figsize=(18,6))

  ranked = predicted_clases[:, i]
  cm = metrics.confusion_matrix(y_partition, ranked, labels = encoder_classes)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, \
                                        display_labels=encoder_classes)
  disp.plot(values_format='')
  plt.tight_layout()
  plt.show()



make_submission_file(test_preds, test, encoder_classes)


pd.read_csv('./submission.csv')

