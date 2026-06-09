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


# Loading train & test
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col="id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col="id")
df_original = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
# Shapes
print(f"Train shape: {df_train.shape}.")
print(f"Test shape: {df_test.shape}.")
print(f"Original shape: {df_original.shape}.")


df_train.describe()


df_train.info()


df_original.info()


# Adding more data.
df_train = pd.concat([df_train, df_original], axis=0, ignore_index=True)
# Drop duplicated.
df_train = df_train.drop_duplicates()
# Drop na.
df_train = df_train.dropna(subset=["Listening_Time_minutes"])


categorical_column_names = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment",
]
numerical_column_names = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]
column_name_to_predict = "Listening_Time_minutes"


# Numerical columns.
import matplotlib.pyplot as plt
import seaborn as sns

for column_name in numerical_column_names:
    if column_name in ["Podcast_Name", "Episode_Title"]:
        continue
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df_train, x=column_name, bins=30)
    plt.title(f"Histogram of {column_name} with 30 bins.")
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_train, x=column_name)
    plt.title(f"Boxplot of {column_name}.")
    plt.tight_layout()
    plt.show()


df_total = pd.concat([df_train, df_test])
for column_name in df_total.columns:
    if column_name == column_name_to_predict:
        continue
    if df_total[column_name].dtype == "object":
        df_total[column_name] = df_total[column_name].astype("category")
    elif df_total[column_name].dtype == "float64":
        df_total[column_name] = df_total[column_name].astype("float32")
    elif df_total[column_name].dtype == "float32":
        continue
    else:
      raise TypeError(f"{column_name}: {df_total[column_name].dtype}.")
df_total["Number_of_Ads"] = df_total["Number_of_Ads"].clip(0, 4)
df_total["Host_Popularity_percentage"] = df_total["Host_Popularity_percentage"].clip(0, 100)
df_total["Guest_Popularity_percentage"] = df_total["Guest_Popularity_percentage"].clip(0, 100)
df_total["Episode_Title"] = df_total.Episode_Title.str.extract('(\d+)').astype("int32")
df_total["Publication_Day"] = df_total.Publication_Day.map(
    {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
).astype("int32")
df_total["Publication_Time"] = df_total.Publication_Time.map(
    {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2,
        "Night": 3,
    }
).astype("int32")
df_total.info()


COMBO_NANs = ["NaNs"]
df_total["NaNs"] = np.float32(0)

for index, column_name in enumerate(categorical_column_names):
    df_total["NaNs"] += df_total[column_name].isna()*2**index

    n = f"{column_name}_nan_elm"
    df_total[n] = (
        df_total[column_name].isna()*100
        + df_total["Episode_Length_minutes"]
    )
    COMBO_NANs.append(n)

df_total["NaNs"] = df_total["NaNs"].astype("float32")


def fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Filling na values.

    :param df:
    :return pd.DataFrame:
    """
    # Copying.
    df_filled = df.copy()
    # Iterating over columns.
    for column_name in df_filled.columns:
        if column_name in [column_name_to_predict, "Podcast_Name", "Episode_Title"]:
            continue
        if column_name == "Episode_Length_minutes":
            value = 60
        elif df_filled[column_name].dtype == "category":
            value = df_filled[column_name].mode()[0]
        elif df_filled[column_name].dtype in ["float32", "float64"]:
            value = df_filled[column_name].mean()
        elif df_filled[column_name].dtype == "int32":
            value = df_filled[column_name].mode()[0]
        else:
            raise TypeError(f"{column_name}: {df_filled[column_name].dtype}.")
        df_filled[column_name] = df_filled[column_name].fillna(value=value)
    # Returning filled df.
    return df_filled

df_total_filled = fillna(df=df_total)
df_total_filled.info()


# Making a coipy for safety.
df_total_extended = df_total_filled.copy()
# Generating feature interact
feature_sum_interact = []
tot_features = categorical_column_names + numerical_column_names
for index_inner, column_name1 in enumerate(tot_features):
    if not column_name1 in [
        "Episode_Length_minutes",
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
    ]:
        continue
    for index_outer, column_name2 in enumerate(tot_features):
        n = f"{column_name1}_{column_name2}"
        if (
            (column_name1 == column_name2)
            or (n in feature_sum_interact)
            or (f"{column_name2}_{column_name1}" in feature_sum_interact)
        ):
            continue
        df_total_extended[n] = (
            df_total_extended[column_name1].astype("str")
            + '_'
            + df_total_extended[column_name2].astype("str")
        )
        feature_sum_interact.append(n)

print(f"There are {len(feature_sum_interact)} interaction features:")
print(feature_sum_interact)


# Making a coipy for safety.
df_total_extended2 = df_total_extended.copy()
# Generating feature interact
feature_product_interact = []
for index_inner, column_name1 in enumerate(numerical_column_names):
    for index_outer, column_name2 in enumerate(numerical_column_names[index_inner+1:]):
        n = f"{column_name1}_product_{column_name2}"
        df_total_extended2[n] = (
            df_total_extended2[column_name1]
            * df_total_extended2[column_name2]
        )
        feature_product_interact.append(n)

print(f"There are {len(feature_product_interact)} interaction features:")
print(feature_product_interact)


df_total_extended2.info()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generating new variables.

    :param df:
    :return pd.DataFrame:
    """
    # Copying dataset for safety.
    df_fe = df.copy()
    # Functions
    def get_episode_length_class(x):
        if x < 0:
            raise ValueError("")
        elif (x>=0) & (x<30):
            return 0 # "Short"
        elif (x>=30) & (x<60):
            return 1 # "Medium"
        elif (x>=60) & (x<100):
            return 2 # "Long"
        else:
            return 3 # "Super Long"
    def host_guest_popularity_class(x):
        if (x < 0) or (x > 100):
            raise ValueError("")
        elif (x>=0) and (x<25):
            return 0 # "Unknown"
        elif (x>=25) and (x<50):
            return 1 # "Normal"
        elif (x>=50) and (x<75):
            return 2 # "Famous"
        else:
            return 3 # "Star"
    # Generating new variables.
    df_fe["cos_day"] = np.cos(2 * np.pi * df_fe.Publication_Day / 7).astype("float32")
    df_fe["sin_day"] = np.sin(2 * np.pi * df_fe.Publication_Day / 7).astype("float32")
    df_fe["cos_time"] = np.cos(2 * np.pi * df_fe.Publication_Time / 4).astype("float32")
    df_fe["sin_time"] = np.sin(2 * np.pi * df_fe.Publication_Time / 4).astype("float32")
    df_fe["cos_elm"] = np.cos(2 * np.pi * df_fe.Episode_Length_minutes / 60).astype("float32")
    df_fe["sin_elm"] = np.sin(2 * np.pi * df_fe.Episode_Length_minutes / 60).astype("float32")
    df_fe["day_time"] = (
        df_fe.Publication_Day.astype("str")
        + "_"
        + df_fe.Publication_Time.astype("str")
    ).astype("category")
    df_fe["episode_length_class"] = (
        df_fe.Episode_Length_minutes
        .apply(lambda x: get_episode_length_class(x))
        .astype("float32")
    )
    df_fe["host_popularity_class"] = (
        df_fe.Host_Popularity_percentage
        .apply(lambda x: host_guest_popularity_class(x))
        .astype("float32")
    )
    df_fe["guest_popularity_class"] = (
        df_fe.Guest_Popularity_percentage
        .apply(lambda x: host_guest_popularity_class(x))
        .astype("float32")
    )
    df_fe = df_fe.merge(
        (
            df_fe
            .groupby("Podcast_Name")["Episode_Title"]
            .count()
            .to_frame()
            .reset_index()
            .rename(columns={"Episode_Title": "number_of_episodes"})
        ),
        on="Podcast_Name",
        how="left",
    )
    # Returning
    return df_fe, [
        "cos_day", "sin_day", "cos_time", "sin_time",
        "cos_elm", "sin_elm",
        "episode_length_class", "number_of_episodes",
        "day_time", "host_popularity_class", "guest_popularity_class",
    ] 

df_fe, new_cols = feature_engineering(df=df_total_extended2)
feature_column_names = categorical_column_names + numerical_column_names
df_fe.info()


from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
)

def scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Scaling columns.

    :param df:
    :return pd.DataFrame:
    """
    # Copying.
    df_scaled = df.copy()
    # Iterating over columns
    for column_name in df_scaled.columns:
        if column_name in [column_name_to_predict, "cos_day", "sin_day", "cos_time", "sin_time", "cos_elm", "sin_elm"]:
            continue
        if df_scaled[column_name].dtype in ["object", "category"]:
            encoder = LabelEncoder()
            df_scaled[column_name] = (
                encoder
                .fit_transform(df_scaled[column_name])
                .astype("int32")
            )
        elif df_scaled[column_name].dtype in ["float32", "float64"]:
            encoder = StandardScaler()
            df_scaled[column_name] = (
                encoder
                .fit_transform(df_scaled[column_name].values.reshape(-1, 1))
                .astype("float32")
            )
        elif df_scaled[column_name].dtype == "int64":
            encoder = StandardScaler()
            df_scaled[column_name] = (
                encoder
                .fit_transform(df_scaled[column_name].values.reshape(-1, 1))
                .astype("float32")
            )
        elif df_scaled[column_name].dtype == "int32":
            continue
        else:
            raise TypeError(f"{column_name}: {df_scaled[column_name].dtype}.")
    # Return
    return df_scaled

df_total_scaled = scaling(df=df_fe)
df_total_scaled.info()


for index, column_name in enumerate(categorical_column_names):
    n = f"{column_name}_elm"
    df_total_scaled[n] = df_total_scaled[column_name]*100 + df_total_scaled["Episode_Length_minutes"]
    COMBO_NANs.append(n)

# NEW FEATURE - Episode_Length_minutes USING ROUNDING
for k in range(7,10):
    n = f"round{k}"
    df_total_scaled[n] = df_total_scaled["Episode_Length_minutes"].round(k)
    COMBO_NANs.append(n)
    tmp = df_total_scaled.groupby(n).Listening_Time_minutes.mean()
    tmp.name = f"ltm_r{k}"
    df_total_scaled = df_total_scaled.merge(tmp, on=n, how="left")
    COMBO_NANs.append(f"ltm_r{k}")


# NEW FEATURE - DIGIT EXTRACTION FROM WEIGHT CAPACITY
for k in range(1,10):
    df_total_scaled[f'digit{k}'] = (
        ((df_total_scaled["Episode_Length_minutes"] * 10**k) % 10)
        .fillna(-1)
        .astype("int8")
    )
COMBO_NANs += [f"digit{k}" for k in range(1,10)]

# NEW FEATURE - COMBINATIONS OF DIGITS 
for i in range(4):
    for j in range(i+1,5):
        n = f"digit_{i+1}_{j+1}"
        df_total_scaled[n] = (
            (
                (
                    (df_total_scaled[f'digit{i+1}']+1)*11
                ) + (
                    df_total_scaled[f'digit{j+1}']+1
                )
            ).astype("int8")
        )
        COMBO_NANs.append(n)

print(f"There are {len(COMBO_NANs)} Nans features created.")
print(COMBO_NANs)


df_train_prep = df_total_scaled[:len(df_train)]
df_test_prep = df_total_scaled[len(df_train):].drop(columns=[column_name_to_predict])
df_test_prep.info()


# XGB parameters.
xgb_params = {
    "device": "cuda",
    "random_state": 42,
    "max_depth": 10,
    "alpha": 0.9,
    "colsample_bytree": 0.6,
    "subsample": 0.9,
    "n_estimators": 800,
    "learning_rate": 0.1,
    "eval_metric": "rmse",
    "early_stopping_rounds": 100,
    "enable_categorical": True,
}


# Import.
import random
import datetime as dt
from sklearn.model_selection import (
    KFold,
    GroupKFold,
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Trying to select features
ADD_XGB = []
best_auc_xgb = 100
best_oof_xgb = None
best_pred_xgb = None
columns_to_test = (
    feature_product_interact.copy()
    # + feature_sum_interact.copy()
    + COMBO_NANs.copy()
    + new_cols.copy()
)
random.shuffle(columns_to_test)
FOLDS = len(df_train_prep.Publication_Day.unique())
print(f"There is {len(columns_to_test) + 1} models to train.")

# FORWARD FEATURE SELECTION 
for k, col in enumerate(['baseline'] + columns_to_test):
    kf = KFold(n_splits=FOLDS)

    oof_xgb = np.zeros(len(df_train_prep))
    pred_xgb = np.zeros(len(df_test_prep))

    if col!='baseline':
        ADD_XGB.append(col)

    start_time = dt.datetime.now()
    # GROUP K FOLD USING YEAR AS GROUP
    for i, (train_index, test_index) in enumerate(kf.split(df_train_prep)):
        # TRAIN AND VALID DATA
        x_train = df_train_prep.loc[train_index, feature_column_names + ADD_XGB].copy()
        y_train = df_train_prep.loc[train_index, column_name_to_predict]
        x_valid = df_train_prep.loc[test_index, feature_column_names + ADD_XGB].copy()
        y_valid = df_train_prep.loc[test_index, column_name_to_predict]
        x_test = df_test_prep[feature_column_names + ADD_XGB].copy()

        # TRAIN XGB MODEL
        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    
        # INFER OOF
        oof_xgb[test_index] = model_xgb.predict(x_valid)
        pred_xgb += model_xgb.predict(x_test)
        
    # COMPUTE AVERAGE TEST PREDS
    pred_xgb /= FOLDS

    # COMPUTE CV VALIDATION AUC SCORE
    true = df_train_prep[column_name_to_predict].values
    m_xgb = np.sqrt(mean_squared_error(y_true=true, y_pred=oof_xgb))
    run_time = (dt.datetime.now() - start_time).seconds
   
    # XGB
    if m_xgb < best_auc_xgb:
        print(f"{k+1}/{len(columns_to_test)+1} - XGB: BEST with {col} at {m_xgb:.2f} - {run_time / 60:.2f} mins.")
        best_auc_xgb = m_xgb
        best_oof_xgb = oof_xgb.copy()
        best_pred_xgb = pred_xgb.copy()
    else:
        ADD_XGB.remove(col)

print(f"\n\n\tXGB: Best with columns: {ADD_XGB} - score is: {best_auc_xgb:.3f} - to beat is 12.82%.")  # Start 12.82


# Import.
import datetime as dt
from sklearn.model_selection import (
    KFold,
    GroupKFold,
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Setting KFold.
FOLDS = 7
kf = KFold(n_splits=FOLDS)

# Preparing outputs.
oof = np.zeros(len(df_train_prep))
pred = np.zeros(len(df_test_prep))

# Iterating over groups.
for i, (train_index, test_index) in enumerate(kf.split(df_train_prep)):
    start_time = dt.datetime.now()
    X_train = df_train_prep.loc[train_index, feature_column_names + ADD_XGB].reset_index(drop=True).copy()
    y_train = df_train_prep.loc[train_index, column_name_to_predict]

    X_valid = df_train_prep.loc[test_index, feature_column_names + ADD_XGB].reset_index(drop=True).copy()
    y_valid = df_train_prep.loc[test_index, column_name_to_predict]

    X_test = df_test_prep[feature_column_names + ADD_XGB].reset_index(drop=True).copy()

    model = XGBRegressor(**xgb_params)

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    predict_train = model.predict(X_valid)
    oof[test_index] += predict_train
    pred += model.predict(X_test)
    
    run_time = (dt.datetime.now() - start_time).seconds
    score = np.sqrt(mean_squared_error(y_true=y_valid, y_pred=predict_train))

    print(f"# Fold {i+1}. training took: {run_time / 60:.2f} minutes - rmse: {score:.2f}%.")

pred /= FOLDS
final_score = np.sqrt(mean_squared_error(y_true=df_train_prep[column_name_to_predict], y_pred=oof))
print(f"Final rmse with {FOLDS} folds is: {final_score:.2f}%. - to beat is 12.83%.")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(13,9))
abs_feature_importance = abs(model.feature_importances_.reshape(-1))
sns.barplot(
    x="feature_importance",
    y="feature_names",
    data=pd.DataFrame({
        "feature_importance": abs_feature_importance,
        "feature_names": feature_column_names + ADD_XGB,
    }).sort_values(by=["feature_importance"], ascending=False),
)
plt.title("Feature Importance from XGB")
plt.show()


df_test[column_name_to_predict] = pred
df_test[[column_name_to_predict]].to_csv("submission.csv", index=True, sep=",")


!head -5 submission.csv




