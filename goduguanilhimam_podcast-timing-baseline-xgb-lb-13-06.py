import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

TARGET_COLUMN = "Listening_Time_minutes"


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train_df.info()


train_df.head(10)


train_df.isna().sum()


def initial_imputing(df, impute_model):
    df["Episode_Title"] = df["Episode_Title"].apply(lambda row: int(row.split()[-1]))
    target = df.pop("Listening_Time_minutes")
    
    num_cols = df.select_dtypes(include=["number"]).columns
    print(num_cols)
    df_num = df[num_cols].copy()
    filled_df = pd.DataFrame(
        impute_model.fit_transform(df_num), columns=df_num.columns, index=df_num.index
    )

    df[num_cols] = filled_df
    return (df, target)

impute_model = SimpleImputer(strategy="median")
df, target = initial_imputing(train_df, impute_model)
df.info()


sns.displot(x=target)


print("Unique Podcast Names:\n", df["Podcast_Name"].unique(), end="\n\n")
print("No of Unique Podcasts:\n", df["Podcast_Name"].nunique())


print("Unique Podcast Genre:\n", df["Genre"].unique(), end="\n\n")
print("No of Unique Podcast Genre:\n", df["Genre"].nunique())


sns.displot(df, x="Genre")
plt.xticks(rotation=90)
plt.show()


print("Unique Podcast Publication Days:\n", df["Publication_Day"].unique(), end="\n\n")
print("No of Unique Podcast Publication Days:\n", df["Publication_Day"].nunique())


sns.displot(df, x="Publication_Day")
plt.xticks(rotation=90)
plt.show()


print("Unique Podcast Publication Time of Day:\n", df["Publication_Time"].unique(), end="\n\n")
print("No of Unique Podcast Publication Times of Day:\n", df["Publication_Time"].nunique())


sns.displot(df, x="Publication_Time")
plt.xticks(rotation=90)
plt.show()


sns.displot(df, x="Episode_Sentiment")
plt.xticks(rotation=90)
plt.show()


sns.displot(df, x="Episode_Length_minutes")


sns.displot(df, x="Host_Popularity_percentage")


sns.displot(df, x="Guest_Popularity_percentage")


df["Number_of_Ads"].unique()


sns.displot(df, x="Episode_Title")


df = df.drop("id", axis=1)
df.info()


cat_columns = df.select_dtypes("object")
cat_columns


ord_enc = OrdinalEncoder()

def baseline_cat_tfx(df):
    cat_df = pd.DataFrame(
        ord_enc.fit_transform(df), columns=df.columns
    )

    return cat_df

cat_df = baseline_cat_tfx(cat_columns)
cat_df


num_df = df.select_dtypes(exclude="object")
num_df


combined_df = pd.concat(
    [num_df, cat_df], axis=1
)
combined_df


corr_mat = pd.concat([combined_df, target], axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdYlBu")


def mi_scores(df, target, cat_df):
    discrete_features = ~df.columns.isin(cat_df.columns)
    
    mi_scores = mutual_info_regression(
        df, target, discrete_features=discrete_features
    )
    mi_scores = pd.Series(
        mi_scores, index=df.columns
    )
    
    mi_scores = mi_scores.sort_values(ascending=True)
    width = np.arange(len(mi_scores))

    return (mi_scores, width)

mi_scores, width = mi_scores(combined_df, target, cat_df)

plt.figure(figsize=(8, 6))
plt.barh(width, mi_scores.values)
plt.yticks(width, list(mi_scores.index))
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    combined_df, target, test_size=0.12, shuffle=True
)

print(X_train.shape)
print(y_train.shape)


std_sc = StandardScaler()
mm_sc = MinMaxScaler()

def combined_tfx(df, target, test=False):
    if test:
        sc_df = pd.DataFrame(
            std_sc.transform(df), columns=df.columns
        )
        sc_target = mm_sc.transform(target.to_numpy().reshape(-1, 1))
    else:
        sc_df = pd.DataFrame(
            std_sc.fit_transform(df), columns=df.columns
        )
        sc_target = mm_sc.fit_transform(target.to_numpy().reshape(-1, 1))

    return (sc_df, sc_target)

X_train_sc, y_train_sc = combined_tfx(X_train, y_train, test=False)
X_test_sc, y_test_sc = combined_tfx(X_test, y_test, test=True)

print(X_train_sc.shape)
print(y_train_sc.shape)


from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error


def find_param_grid(model):
    if isinstance(model, XGBRegressor):
        param_grid = {
            'n_estimators': [200, 300],       # Increased n_estimators
            'learning_rate': [0.03, 0.07],    # Slightly lower learning rates, more rounds
            'max_depth': [4, 7],              # Slightly deeper trees
            'min_child_weight': [1, 2],       # Fine-tune min_child_weight
            'subsample': [0.7, 0.9],          # Adjust subsample slightly
            'colsample_bytree': [0.7, 0.9],   # Adjust colsample_bytree slightly
            'reg_alpha': [0, 0.1],            # L1 regularization - start with small values
            'reg_lambda': [1, 1.5],           # L2 regularization - start around 1
            'tree_method': ['gpu_hist'],
            'predictor': ['gpu_predictor']
        }
    elif isinstance(model, LGBMRegressor):
        param_grid = {
            'n_estimators': [200, 300],     # Increased n_estimators
            'learning_rate': [0.03, 0.07],  # Slightly lower learning rates
            'max_depth': [6, 9],            # Slightly deeper trees
            'num_leaves': [40, 70],         # Adjust num_leaves accordingly
            'min_child_samples': [20, 30],  # Fine-tune min_child_samples
            'subsample': [0.7, 0.9],        # Adjust subsample
            'colsample_bytree': [0.7, 0.9], # Adjust colsample_bytree
            'reg_alpha': [0, 0.1],          # L1 regularization
            'reg_lambda': [1, 1.5],         # L2 regularization
            'device': ["gpu"]
        }

    return param_grid


def regression_report(y_true, y_pred):
    print("Performance on the Test Set:\n")
    print("Mean Absolute Error: ", mean_absolute_error(y_true, y_pred))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_true, y_pred)))

def model_predictions(model, X_train, y_train, X_test, y_test):
    estimator = model()
    param_grid = find_param_grid(estimator)
    grid_search = GridSearchCV(
        estimator, param_grid=param_grid, n_jobs=-1, verbose=1,
        scoring="neg_root_mean_squared_error", cv=5 
    )
    grid_search.fit(X_train_sc, y_train_sc.ravel())

    best_estimator = grid_search.best_estimator_
    print("Performance on the Training Set:\n")
    print("Best Params:\n", grid_search.best_params_, end="\n\n")
    print("Best Score:\n", grid_search.best_score_, end="\n\n")
    
    y_pred = best_estimator.predict(X_test_sc)
    regression_report(y_test_sc, y_pred)
    return best_estimator


# XG Boost
xgb = model_predictions(
    XGBRegressor, X_train_sc, y_train_sc, X_test_sc, y_test_sc
)


# # LGBM Regressor
# lgbm = model_predictions(
#     LGBMRegressor, X_train_sc, y_train_sc, X_test_sc, y_test_sc
# )


test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_df.info()


def impute_test(df):
    test_df["Episode_Title"] = test_df["Episode_Title"].apply(lambda row: int(row.split()[-1]))
    
    num_cols = test_df.select_dtypes(include=["number"]).columns
    num_df = df[num_cols].copy()
    filled_df = pd.DataFrame(
        impute_model.transform(num_df), columns=num_cols, index=num_df.index
    )
    
    df[num_cols] = filled_df
    return df

test_df = impute_test(test_df)
ids = test_df.pop("id")
test_df.info()


def baseline_test_tfx(df):
    cat_columns = df.select_dtypes("object").columns
    cat_df = df[cat_columns].copy()
    cat_tfx = pd.DataFrame(
        ord_enc.transform(cat_df), columns=cat_columns, index=cat_df.index
    )
    
    num_columns = df.select_dtypes(include=["number"]).columns
    num_df = df[num_columns].copy()
    
    combined_test = pd.concat([num_df, cat_tfx], axis=1)
    
    sc_df = pd.DataFrame(
        std_sc.transform(combined_test), columns=combined_test.columns
    )
    return sc_df

sc_df = baseline_test_tfx(test_df)
sc_df


def view_results(best_estimator, model_name):
    model = best_estimator
    
    pred = model.predict(sc_df)
    pred_rescaled = mm_sc.inverse_transform(pred.reshape(-1, 1))
    pred_df = pd.DataFrame(
        np.c_[ids.to_numpy(), pred_rescaled], 
        columns=["id", "Listening_Time_minutes"]
    )
    
    fig = sns.displot(pred_df, x="Listening_Time_minutes")
    plt.title(model_name)

    pred_df.to_csv(f"SUBMISSION_{model_name}.csv", index=False)


view_results(xgb, "XGB")




