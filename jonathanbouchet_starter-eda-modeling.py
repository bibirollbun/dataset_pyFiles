import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import time

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/binary-smoke-detector/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/binary-smoke-detector/test.csv", sep=",")


df_train.head()


df_train.dtypes


# null values ?
res = pd.concat(
    [pd.DataFrame(df_train.isna().sum()), 
     pd.DataFrame(df_test.isna().sum())], axis=1)
res.columns = ["number of null values(train)","number of null values(test)"]
res


# checking age variable
df_train["age"].unique()


# convert to integers
df_train["age"] = df_train["age"].astype("int")
df_test["age"] = df_test["age"].astype("int")
df_train["smoking"] = df_train["smoking"].astype("int")


def plot_feature(feature: str, df= pd.DataFrame, showfliers=True) -> None:
    """plot a :
            boxplot of feature "feature" breakdown by age and target variable
            density plot of feature "feature" breakdown by target variable

    Args:
        feature (str): _description_
        df (pd.DataFrame): _description_
    """
    fig, ax = plt.subplots(1,2, figsize=(16,2), width_ratios=[3, 1])
    sns.boxplot(data=df, x="age", y=feature, hue="smoking", ax=ax[0], showfliers=showfliers)
    sns.kdeplot(data=df, x=feature, ax=ax[1], hue="smoking")
    plt.tight_layout()
    plt.show()


for c in [c for c in df_train.columns if c not in ["id", "smoking", "age"]]:
    plot_feature(feature=c, df=df_train, showfliers=False)


df_train["age"].value_counts().sort_index()


# convert to integer
for c in ['dental caries', 'Urine protein', 'hearing(left)','hearing(right)']:
    df_train[c] = df_train[c].astype("int")
    df_test[c] = df_test[c].astype("int")


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif


# checking imbalance
df_train["smoking"].value_counts()


# select features to be 1hot encoded
cat_cols = ['dental caries', 'Urine protein', 'hearing(left)','hearing(right)', 'age']
# select numerical features
num_cols: list[str] = [c for c in df_train.columns if c not in ["id", "smoking"] + cat_cols]
target: str = "smoking"
print(f"categorical columns: {cat_cols}\nnumerical columns: {num_cols}")

X = df_train[num_cols + cat_cols]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


# pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), 
           ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder="passthrough"
)

preprocessor


def get_confusion_matrix(pipeline, X:list, true:list, labels:list):
    """plot the confusion matrix for a given datatype
    Args: true, pred
    Return: confusion matrix
    """
    pred = pipeline.predict(X)
    return confusion_matrix(true, pred, labels=labels)

clfs = {'RandomForest':RandomForestClassifier(), 
        'KNNClassifier': KNeighborsClassifier(),
        'Adaboost':AdaBoostClassifier(), 
        'xgb':xgb.XGBClassifier(objective="binary:logistic", random_state=42),
        'LR': LogisticRegression()}


labels = sorted(list(y.unique()))
metrics = ['accuracy','balanced_accuracy','f1_macro']
cross_validate_res = []

fig, ax = plt.subplots(2,5, figsize=(18,7))
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 'small'})

for cnt, (clf_name, clf) in enumerate(clfs.items()):
    
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            (clf_name, clf)]
    )
    
    print(f"processing {clf_name}")
    res = cross_validate(pipe, X_train, y_train, cv=3, return_train_score=True, scoring=metrics)
    pipe.fit(X_train, y_train)

    # compute confusion matrix
    cm_train = get_confusion_matrix(pipe, X_train, y_train, labels=labels)
    cm_test = get_confusion_matrix(pipe, X_test, y_test, labels=labels)
    
    # plot CM
    disp_tr = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels)
    disp_te = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)

    disp_tr.plot(ax=ax[0, cnt])
    disp_te.plot(ax=ax[1, cnt])
    ax[0,cnt].grid(False)
    ax[1,cnt].grid(False)
    ax[0,cnt].set_title(f"CM (train) for {clf_name}", fontsize=8)
    ax[1,cnt].set_title(f"CM (test) for {clf_name}", fontsize=8)
    
    res_df = pd.DataFrame(res).mean()
    res_df = pd.DataFrame(res_df).apply(pd.to_numeric).transpose()
    res_df['Classifier'] = clf_name
    cross_validate_res.append(res_df)
    
plt.tight_layout()
plt.show()


pd.concat(cross_validate_res, ignore_index=True).style.background_gradient(cmap="viridis")


# tuning a single classifier
# Create a pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ('selectkbest', SelectKBest()),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'selectkbest__k': [5, 10, 15, 20],
    'classifier__max_depth': [4, 6, 8],
    'classifier__n_estimators':[50, 100],
    'classifier__max_features': ['sqrt','log2'],
    'classifier__criterion':['gini', 'entropy', 'log_loss'],
}


# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()


# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)


from sklearn.metrics import roc_curve, RocCurveDisplay, auc

fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(3,3))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(loc="best", fontsize=10)
plt.show()


preds = best_model.predict(df_test)
preds_probs = best_model.predict_proba(df_test)[:,1]
sample = pd.read_csv("/kaggle/input/binary-smoke-detector/sample_submission.csv", sep=",")
sample["smoking"] = preds_probs
# sample.to_csv("sample_solution_binary_smoke_detector_baseline.csv", index=False)




