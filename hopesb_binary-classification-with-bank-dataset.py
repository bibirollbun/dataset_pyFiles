import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import OneHotEncoder, TargetEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tqdm.notebook import tqdm

import optuna
import warnings
warnings.simplefilter("ignore")


# File path to the datasets.
filepath = "/kaggle/input/playground-series-s5e8/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e8/test.csv"


def wrangle(filepath):
    """
    This helps to process the dataset.

    ----------
    Parameter:
        filepath: str
    Return:
        Dataframe
    """
    # Read the csv filepath.
    df = pd.read_csv(filepath, index_col="id")

     # Creating new features.
    df["questions"]= df["default"] + " " + df["housing"] + " " + df["contact"]
    df["status"] = df["job"] + " " + df["education"] + " " + df["marital"]
    df["intellect"] = df["job"] + " " + df["education"]
    df["min_duration_sin"] = np.sin(df["duration"] / 60)
    df["min_duration_cos"] = np.cos(df["duration"] / 60)
    df["date"] = df["day"].astype(str) + " " + df["month"]
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    
    df = df.drop(columns="pdays")
    
    # print the shape of the dataset.
    rows, cols = df.shape
    print(f"""
    The Dataset Contains:
    Rows: {rows}
    Columns: {cols}
    """)
    # print the number of columns.
    columns = df.columns.tolist()
    print(f"""
    The Dataset contains the following columns:
    {columns}
    """)
    # print the info in the dataset as well as the number of unique values.
    print(f"""
    Dataset Information:""")
    df.info()
    
    print(f"""
    Number of Unique Values:
    {df.nunique().sort_values(ascending=False)}
    """)

    return df


df = wrangle(filepath)
test_df = wrangle(test_filepath)
df.head()


# Distribution of the target.
n_target = df["y"].value_counts(normalize=True)
plt.bar(n_target.index.astype(str), n_target.values)
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.title("Target Distribution");


# Checking for Multicollinearity in the Numerical Dataset
sns.heatmap(df.select_dtypes("number").drop(columns= "y").corr());


abs(df.select_dtypes("number").corr()["y"]).sort_values().plot(kind="barh")
plt.title("Correlation Coefficient")
plt.ylabel("Columns")
plt.xlabel("Coefficient");


target = "y"
X = df.drop(columns=target)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(X_train.shape, y_train.shape)


def score_sub(models, X_train, X_test, y_train, y_test, test_df):
    score_dict = {}
    
    for model in tqdm(models, desc="Processing"):
        model = make_pipeline(
                    OneHotEncoder(use_cat_names=True),
                    StandardScaler(),
                    model
                )
        model.fit(X_train, y_train)
        alg_name = list(model.named_steps.keys())[-1]
        pred = model.predict_proba(X_test)[:, 1]
        # score_acc = accuracy_score(y_test, pred)
        score_roc_auc = roc_auc_score(y_test, pred)

        classes = ['0', '1']
        predi = model.predict(X_test)
        print(classification_report(y_test, predi, target_names=classes))
        cm = confusion_matrix(y_test, predi, labels=[0, 1])
        dis = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[0, 1])
        dis.plot(cmap= plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=15, pad=20)
        plt.xlabel('Prediction', fontsize=11)
        plt.ylabel('Actual', fontsize=11)
        plt.show()
        # Submission.
        test_pred = model.predict_proba(test_df)[:, 1]
        test_pred = test_pred
        sub_df = pd.DataFrame({"y": test_pred}, index=test_df.index)
        # sub_df["y"] = sub_df["Personality"].replace({1: "Extrovert", 0: "Introvert"})
        sub_df.to_csv(f"{alg_name}.csv")
        print(f"\nSubmission File for {alg_name} Created.")
        score_dict[alg_name]= [score_roc_auc]
    df = pd.DataFrame(score_dict, index=["Score"])
    return df.T.sort_values("Score", ascending=False)


xgb_params = {
    'n_estimators': 2805, 
    'learning_rate': 0.03034552103623421, 
    'max_depth': 10, 
    'subsample': 0.9191200410873819, 
    'colsample_bytree': 0.5007886498006493}
lgb_params = {'n_estimators': 965,
 'learning_rate': 0.06218639379404327,
 'num_leaves': 244,
 'max_depth': 10,
 'is_unbalance': True}
cat_params = {'iterations': 976,
 'learning_rate': 0.09409948639683283,
 'depth': 10,
 'bagging_temperature': 2.7935440039897337,
 'auto_class_weights': None}
algorithms = [
             CatBoostClassifier(**cat_params, random_state=42, verbose=0),
             XGBClassifier(**xgb_params, random_state=42),
             LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
]
score_sub(algorithms, X_train, X_test, y_train, y_test, test_df)


# feature importances.
xgb = make_pipeline(
            OneHotEncoder(use_cat_names=True),
            StandardScaler(),
            XGBClassifier(**xgb_params, random_state=42)
        )
xgb.fit(X_train, y_train)
imp = xgb.named_steps["xgbclassifier"].feature_importances_
feat = xgb.named_steps["onehotencoder"].feature_names_out_
feat_imp = pd.Series(imp, index= feat)
feat_imp.sort_values().tail().plot(kind="barh")
plt.ylabel("feature")
plt.xlabel("importance");


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = make_pipeline(
                    OneHotEncoder(use_cat_names=True),
                    StandardScaler(),
                    XGBClassifier(**xgb_params, random_state=42)
                )
cat = make_pipeline(
                OneHotEncoder(use_cat_names=True),
                StandardScaler(),
                CatBoostClassifier(**cat_params, random_state=42, verbose=0)
            )
lgb = make_pipeline(
                OneHotEncoder(use_cat_names=True),
                StandardScaler(),
                LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
            )


estimators = [("xgb", xgb), ("cat", cat)]
weights = [3, 2]

model = VotingClassifier(estimators=estimators, voting='soft', weights= weights)
# Use StratifiedKFold for cross-validation to ensure balanced folds
model.fit(X_train, y_train)

classes = ['0', '1']
predi = model.predict(X_test)
print(classification_report(y_test, predi, target_names=classes))
cm = confusion_matrix(y_test, predi, labels=[0, 1])
dis = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[0, 1])
dis.plot(cmap= plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.show()
# Submission.
test_pred = model.predict_proba(test_df)[:, 1]
test_pred = test_pred
sub_df = pd.DataFrame({"y": test_pred}, index=test_df.index)
alg_name = "voteclassifier"
sub_df.to_csv(f"{alg_name}.csv")
print(f"\nSubmission File for {alg_name} Created.")

