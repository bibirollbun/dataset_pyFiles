import os

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


print(*os.walk('/kaggle/input'), sep='\n')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("submission_data shape :",submission_data.shape)


train_data.head()


# test_data.shape, submission_data.shape
print(test_data.head(), submission_data.head(), sep='\n')



train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)
# cheat: merge original and competition dataset :D
train_data_new = pd.concat([train_data, original_data], ignore_index=True)
train_data_new = train_data_new.drop_duplicates()
print("shape of the data :",train_data_new.shape)


'''
Objective:
To select best fertilizer [Fetilizer Name]

Categorical features:
- soil type
- crop type
- fertilizer name
'''


def plot_pie_bar_feature(dataset, col_name, title_prefix=""):
    counts = train_data[col_name].value_counts()
    labels = counts.index  # unique cats
    values = counts.values  # count of each cat
    colors = plt.cm.tab20_r.colors[:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title(f"{title_prefix}{col_name} Distribution (Bar Chart)")
    axes[0].set_xlabel(col_name)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    axes[1].set_title(f"{title_prefix}{col_name} Distribution (Pie Chart)")
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(dataset: pd.DataFrame, numeric_cols=None):
    if numeric_cols is None:
        df = dataset.select_dtypes(include='number')
    else:
        df = df.loc[numeric_cols]
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)


def plot_numeric_features_dist(datasets, numeric_cols=None, labels=None):
    # distribution of numerical features among different datasets
    if numeric_cols is None:
        numeric_cols = datasets[0].select_dtypes(include='number').columns.tolist()

    num_datasets = len(datasets)
    if labels is None:
        labels = [f"dataset_{i}" for i in range(num_datasets)]
    
    # print(numeric_cols)
    dfs = [df[numeric_cols] for df in datasets]
    num_datasets = len(datasets)
    if labels is None:
        labels = [f"dataset_{i}" for i in range(num_datasets)]
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        for i, df in enumerate(datasets):
            sns.kdeplot(df[col].dropna(), label=labels[i], fill=True, alpha=0.4)
        
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()


train_data_new.info()


plot_numeric_features_dist([train_data, test_data, train_data_new, original_data], 
                           labels=['train', 'test', 'train_merged', 'original'])


plot_pie_bar_feature(train_data_new, 'Soil Type')


plot_pie_bar_feature(train_data_new, 'Crop Type')


plot_pie_bar_feature(train_data_new, 'Fertilizer Name')


plot_correlation_heatmap(train_data_new)








def pd_one_hot(dataset, col):
    one_hot = pd.get_dummies(dataset[col])
    dataset = dataset.drop(col,axis = 1)
    dataset = dataset.join(one_hot)
    return dataset


def transform_dataset(dataset, cat_cols: list[str]):
    # categorical to onehot
    for col in cat_cols:
        dataset = pd_one_hot(dataset, col)
    
    # normalize (no need for decision trees)
    
    return dataset


print(train_data_new.shape)
train_data_new.head()


num_cols_train = list(train_data_new.select_dtypes(include='number')\
    .columns\
    .difference(['Fertilizer Name']))
cat_cols_train = list(train_data_new.select_dtypes(exclude='number')\
    .columns\
    .difference(['Fertilizer Name']))
num_cols_test = list(test_data.select_dtypes(include='number')\
    .columns)
cat_cols_test = list(test_data.select_dtypes(exclude='number')\
    .columns)

num_cols_train, cat_cols_train, num_cols_test, cat_cols_test


train_data_copy = train_data_new.copy()
test_data_copy = test_data.copy()

cat_cols = test_data.select_dtypes(exclude='number').columns.tolist()
feature_les = {col: LabelEncoder() for col in cat_cols}  # encoder for each categorical feature
target_le = LabelEncoder()


for col in cat_cols:
    train_data_copy[col] = feature_les[col].fit_transform(train_data_copy[col])
    test_data_copy[col] = feature_les[col].transform(test_data_copy[col])

train_data_copy['Fertilizer Name'] = target_le.fit_transform(train_data_copy['Fertilizer Name'])


train_data_copy.head()[cat_cols]


num_classes = len(target_le.classes_)
num_classes


from sklearn.preprocessing import label_binarize
import numpy as np


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    """
    predicted_top5[i] is a length‐3 array of labels (strings/ints) that your model thinks
    are most likely for sample i, ordered from most confident 3rd most confident.
    y_true_fold[i] is the single true label for sample i.
    We give credit = 1/rank if the true label is at position 'rank' in that top‐3 list;
    otherwise 0. Then we average over all i.
    """
    print(type(predicted_top3), type(y_true_fold))
    
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


X = train_data_copy.drop('Fertilizer Name',axis = 1)
y = train_data_copy["Fertilizer Name"]
# test = test_data_copy.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


def xgboost_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'eta': trial.suggest_float('eta', 0, 1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        # x_train_fold = X_train[train_index]
        # x_val_fold = X_train[val_index]
        # y_train_fold = y_train[train_index]
        # y_val_fold = y_train[val_index]

        x_train_fold = X_train.iloc[train_index]
        x_val_fold = X_train.iloc[val_index]
        
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        model = XGBClassifier(
            **params,
            verbosity=0,
            objective='multi:softprob',
            enable_categorical=True,
            tree_method="gpu_hist",
            gpu_id=0, 
            predictor="gpu_predictor",
            n_jobs=-1,
            random_seed=42
        )
        
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              early_stopping_rounds=50, verbose=False)

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        fold_map3 = map3_score(top3_labs, y_val_fold.to_numpy())
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

    return mean_map3


# study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
# study.optimize(xgboost_objective, n_trials=50, n_jobs=1)
# print("Best trial:")
# print(study.best_trial.params)


test_data_copy.head()


# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# Best trial:
# {'learning_rate': 0.04035529891870569, 'max_depth': 20, 'min_child_weight': 5.533830209405815, 'gamma': 0.2845805417802597, 'alpha': 2.9500411716472144, 'subsample': 0.5998720852200778, 'colsample_bytree': 0.4193001268301755, 'eta': 0.5271936074396966, 'n_estimators': 893, 'lambda': 0.01059616433916218}

params = {
    'learning_rate': 0.04035529891870569, 'max_depth': 20, 
    'min_child_weight': 5.533830209405815, 'gamma': 0.2845805417802597, 
    'alpha': 2.9500411716472144, 'subsample': 0.5998720852200778, 
    'colsample_bytree': 0.4193001268301755, 'eta': 0.5271936074396966, 
    'n_estimators': 893, 'lambda': 0.01059616433916218
}

xgb_classifier = XGBClassifier(
    **params, 
    verbosity=0,
    objective='multi:softprob',
    enable_categorical=True,
    tree_method="gpu_hist",
    gpu_id=0, 
    predictor="gpu_predictor",
    n_jobs=-1,
    random_seed=42
)

xgb_classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
      early_stopping_rounds=50, verbose=False)


test_data_copy.head().to_numpy()


pred_proba = xgb_classifier.predict_proba(test_data_copy.to_numpy())
top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labs = xgb_classifier.classes_
top3_labs = class_labs[top3_index]


top3_result_strings = np.array(list(map(
    lambda x: ' '.join(target_le.inverse_transform(x)), top3_labs)))


top3_result_strings.shape, test_data.shape, submission_data.shape


submission = pd.DataFrame({
    'id': submission_data['id'].values,
    'Fertilizer Name': top3_result_strings
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission

