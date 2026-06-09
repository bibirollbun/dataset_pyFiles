# import libraries.
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm.notebook import tqdm
import optuna


# Preprocessing the dataset.
def data_processing(filepath):
    """
    Process the dataset.

    --------
    Parameters:
        filepath: path to the dataset.

    -------
    Return:
        DataFrame.
    """
    df = pd.read_csv(filepath, index_col="id")

    df2 = df[["cloud", "humidity"]].shift([1, 2]).bfill()

    df = pd.concat([df, df2], axis=1)
    
    df["cloud_1"] = np.sin(np.pi * 2 * df["humidity_1"]/365)

    df = df.drop(columns=["mintemp", "temparature", "maxtemp", "winddirection", "humidity_1"])

    return df


train_filepath = "/kaggle/input/playground-series-s5e3/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e3/test.csv"


df = data_processing(train_filepath)
df.head()


df.describe()


sns.heatmap(df.drop(columns="rainfall").corr());


abs(df.corr()["rainfall"]).sort_values(ascending=True).plot(kind="barh")
plt.title("Feature Correlation withe the Target(rainfall)")
plt.ylabel("Features")
plt.xlabel("Correlation Coefficient");


sns.boxplot(df, x="rainfall", y="cloud");


sns.boxplot(df, x="rainfall", y="sunshine");


target = "rainfall"
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, y_train.shape


test_df = data_processing(test_filepath)


# Create a function that takes in a list of algortithm and then make the prediction then save the prediction.
def score_predict(algorithms, X_train, X_test, y_train, y_test, test_df=None, save=True): 

    test_data_dict = {}

    # loop throught the algorithms.
    for algorithm in tqdm(algorithms, desc="Starting"):
        # make the pipeline
        model = make_pipeline(
            MinMaxScaler(),
            algorithm
        )
        # Fitting the model.
        alg_name = list(model.named_steps.keys())[-1]
        print(f"Fitting the {alg_name}")
        model.fit(X_train, y_train)
        
        kfold= KFold(n_splits=10)
        score = cross_val_score(model, X, y, cv=kfold, scoring="roc_auc").mean()
        # Get the score.
        print("Making Prediction on the test dataset")
        y_test_pred = model.predict_proba(X_test)[:, 1]
        test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
        test_data_dict[alg_name] = [test_roc_auc_score, score]
        
        if save == True:
            y_pred = model.predict_proba(test_df)[:, 1]
            pd.DataFrame({"rainfall": y_pred}, index=test_df.index).to_csv(f"{alg_name}.csv")

    test_score_df = pd.DataFrame(test_data_dict, index=["Test_ROC_AUC", "CV_Score"]).T.sort_values(by="Test_ROC_AUC", ascending=False)
    
    return test_score_df


# lgb_params= {'learning_rate': 0.013404480931044339,
#  'max_depth': 2,
#  'n_estimators': 913,
#  'colsample_bytree': 0.9225549105850854,
#  'reg_alpha': 0.00727232867980159,
#  'reg_lambda': 0.004041825210663614}
lgb_params = {'learning_rate': 0.0711427728871425,
 'max_depth': 2,
 'n_estimators': 253,
 'colsample_bytree': 0.19739276131827962,
 'reg_alpha': 0.0011069844888468342,
 'reg_lambda': 0.08725400368504219}
# xgb_params = {'learning_rate': 0.014956810065122068,
#  'max_depth': 4,
#  'n_estimators': 589,
#  'colsample_bytree': 0.7295368018926482,
#  'colsample_bylevel': 0.14809885789657845,
#  'colsample_bynode': 0.38890619077474303,
#  'reg_alpha': 0.009360396514881238,
#  'reg_lambda': 0.0023547419665849124}
xgb_params = {'learning_rate': 0.047043030343516784,
 'max_depth': 2,
 'n_estimators': 254,
 'colsample_bylevel': 0.5526211047601903,
 'colsample_bytree': 0.7194225188907712,
 'colsample_bynode': 0.9067636822944657,
 'reg_alpha': 0.039084050180955364,
 'reg_lambda': 0.07373115657838296}
cat_params = {'learning_rate': 0.0318014293989377,
 'max_depth': 2,
 'n_estimators': 688,
 'colsample_bylevel': 0.3997104608510243,
 'l2_leaf_reg': 0.8230684007601325}
algorithms = [LGBMClassifier(**lgb_params, verbose=-1, random_state=42),
             XGBClassifier(**xgb_params, random_state=42),
             CatBoostClassifier(**cat_params, verbose=0, random_state=42),
             ]
score = score_predict(algorithms, X_train, X_test, y_train, y_test, test_df)
score


estimators = [("lgb", algorithms[0]), ("xgb", algorithms[1])]
vote = VotingClassifier(estimators=estimators, weights=[1, 2], voting='soft')
vote.fit(X_train, y_train)
y_test_pred = vote.predict_proba(X_test)[:, 1]
test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
print(test_roc_auc_score)
y_pred = vote.predict_proba(test_df)[:, 1]
pd.DataFrame({"rainfall": y_pred}, index=test_df.index).to_csv(f"submission.csv")


# def objective(trials):
#     params = {
#         "learning_rate": trials.suggest_float("learning_rate", 0.001, 0.1),
#         "max_depth": trials.suggest_int("max_depth", 2, 16),
#         "n_estimators": trials.suggest_int("n_estimators", 200, 1000),
#         "colsample_bylevel": trials.suggest_float("colsample_bylevel", 0.0, 1.0),
#         "l2_leaf_reg": trials.suggest_float("l2_leaf_reg", 0.00, 1.00),
#             }
#     model = make_pipeline(
#             MinMaxScaler(),
#             CatBoostClassifier(**params, random_state=42, verbose=0)
#     )
#     # kfold = KFold(n_splits=10)
#     # score = cross_val_score(model, X_train, y_train, cv=kfold, scoring="roc_auc").mean()

#     model.fit(X_train, y_train)
    
#     y_test_pred = model.predict_proba(X_test)[:, 1]
#     test_roc_auc_score = roc_auc_score(y_test, y_test_pred)
    
#     return test_roc_auc_score


# study = optuna.create_study(study_name="CAT", direction="maximize")
# study.optimize(objective, n_trials=100, show_progress_bar=True)


# study.best_params




