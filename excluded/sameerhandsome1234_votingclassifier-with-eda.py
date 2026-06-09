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


import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns    


train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train_df.head()


cat_feats = ['Soil Type', 'Crop Type']

for col in cat_feats:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=train_df,
        x=col,
        order=train_df[col].value_counts().index,
        palette='Set2',
        edgecolor='black'
    )
    plt.title(f'{col} Distribution', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f'\nðŸ“Š Proportion of Each Category in "{col}":\n')
    print(train_df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


def plot_correlation_heatmap(dataset: pd.DataFrame, numeric_cols=None):
    if numeric_cols is None:
        df = dataset.select_dtypes(include='number')
    else:
        df = df.loc[numeric_cols]
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)


plot_correlation_heatmap(train_df)


def new_column(df):
    df['Total_Compound']= df['Nitrogen']+df['Potassium']+df['Phosphorous']
    df['Nitrogen_Prop']= df['Nitrogen']/df['Total_Compound']
    df['Potassium_Prop']= df['Potassium']/df['Total_Compound']
    df['Phosphorous_Prop']= df['Phosphorous']/df['Total_Compound']
    df['Temp_Humidity']= df['Temparature']*df['Humidity']
    df['Temp_Moisture']= df['Temparature']*df['Moisture']
    df['Humi_Moisture']= df['Humidity']*df['Moisture']

new_column(train_df)
new_column(test_df)


cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])
    test_df[col] = LabelEncoder().fit_transform(test_df[col])


le = LabelEncoder()
train_df["Fertilizer Name"] = le.fit_transform(train_df["Fertilizer Name"])
target_classes = le.classes_


X=train_df.drop(columns=['id','Fertilizer Name'])
y=train_df['Fertilizer Name']
X_test=test_df.drop(columns='id')


numeric_cols=X.select_dtypes(include=np.number).columns.tolist()


scaler=StandardScaler()
scaler.fit(X[numeric_cols])
X[numeric_cols]=scaler.transform(X[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break 
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


import optuna

def objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga", "newton-cg"]),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000)
    }

    model = LogisticRegression(
        **params,
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)

        top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
        map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
        scores.append(map3)

    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best hyperparameters:", study.best_params)



params= {'C': 0.0013077771143426063, 'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 278}
model = LogisticRegression(
    **params,
    multi_class='multinomial',
    random_state=42,
    n_jobs=-1
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]

    map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
    scores.append(map3)

print(f"Cross-Validation MAP@3 scores: {scores}")
print(f"Mean MAP@3: {np.mean(scores):.4f}")


global2 = None
scores = []
desired_fold = 2  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)
        scores.append(map3_score)

        y_proba_test = model.predict_proba(X_test)
        global2 = y_proba_test

print(f"Fold {desired_fold} MAP@3 score: {scores}")


from IPython.display import FileLink

top_3_preds = np.argsort(global2, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission3.csv', index=False)
FileLink("fsubmission3.csv")



param={'n_estimators': 387, 'max_depth': 7, 'learning_rate': 0.08191929192304931, 'subsample': 0.9133364364348068, 'colsample_bytree': 0.9980024005431026, 'gamma': 0.13046712134556082, 'min_child_weight': 10}
     
model = XGBClassifier(
    **param,
    objective='multi:softprob',  
    num_class=7,  
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device="cuda" 
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]

    map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
    scores.append(map3)

print(f"Cross-Validation MAP@3 scores: {scores}")
print(f"Mean MAP@3: {np.mean(scores):.4f}")


global3 = None
desired_fold = 3  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)

        y_proba_test = model.predict_proba(X_test)
        global3 = y_proba_test


from IPython.display import FileLink

top_3_preds = np.argsort(global3, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission4.csv', index=False)
FileLink("fsubmission4.csv")



from lightgbm import LGBMClassifier

params1={'n_estimators': 421, 'max_depth': 12, 'learning_rate': 0.08209154690911798, 'num_leaves': 47, 'subsample': 0.5652850984207674, 'colsample_bytree': 0.540501980372391, 'min_child_samples': 29, 'reg_alpha': 3.0294381123965324, 'reg_lambda': 0.35114687815583745}
model = LGBMClassifier(
    **params1,
    objective='multiclass',
    num_class=7,
    random_state=42,
    n_jobs=-1,
    device='gpu', 
    verbose=-1
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]

    map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
    scores.append(map3)

print(f"Cross-Validation MAP@3 scores: {scores}")
print(f"Mean MAP@3: {np.mean(scores):.4f}")



global4 = None
desired_fold = 2  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_proba_val = model.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)

        y_proba_test = model.predict_proba(X_test)
        global4 = y_proba_test


from IPython.display import FileLink

top_3_preds = np.argsort(global4, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission5.csv', index=False)
FileLink("fsubmission5.csv")



from sklearn.ensemble import VotingClassifier


logreg_params = {
    'C': 0.0013077771143426063,
    'solver': 'newton-cg',
    'penalty': 'l2',
    'max_iter': 278,
    'multi_class': 'multinomial',
    'random_state': 42,
    'n_jobs': -1
}

xgb_params = {
    'n_estimators': 387,
    'max_depth': 7,
    'learning_rate': 0.08191929192304931,
    'subsample': 0.9133364364348068,
    'colsample_bytree': 0.9980024005431026,
    'gamma': 0.13046712134556082,
    'min_child_weight': 10,
    'objective': 'multi:softprob',
    'num_class': 7,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'
}

lgbm_params = {
    'n_estimators': 421,
    'max_depth': 12,
    'learning_rate': 0.08209154690911798,
    'num_leaves': 47,
    'subsample': 0.5652850984207674,
    'colsample_bytree': 0.540501980372391,
    'min_child_samples': 29,
    'reg_alpha': 3.0294381123965324,
    'reg_lambda': 0.35114687815583745,
    'objective': 'multiclass',
    'num_class': 7,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',
    'verbose': -1
}

logreg = LogisticRegression(**logreg_params)
xgb = XGBClassifier(**xgb_params)
lgbm = LGBMClassifier(**lgbm_params)

voting_clf = VotingClassifier(
    estimators=[
        ('logreg', logreg),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    voting='soft',  
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    voting_clf.fit(X_train, y_train)
    y_proba = voting_clf.predict_proba(X_val)

    top_3_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :3]
    map3 = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)
    scores.append(map3)

print(f"Cross-Validation MAP@3 scores: {scores}")
print(f"Mean MAP@3: {np.mean(scores):.4f}")



global5 = None
desired_fold = 2  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    if fold == desired_fold:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        voting_clf.fit(X_train, y_train)
        y_proba_val = voting_clf.predict_proba(X_val)
        top_3_preds_val = np.argsort(y_proba_val, axis=1)[:, ::-1][:, :3]
        map3_score = mapk(y_val.tolist(), top_3_preds_val.tolist(), k=3)

        y_proba_test = voting_clf.predict_proba(X_test)
        global5 = y_proba_test


from IPython.display import FileLink

top_3_preds = np.argsort(global5, axis=1)[:, -3:][:, ::-1] 

flat_preds = top_3_preds.ravel()
top_3_labels = le.inverse_transform(flat_preds).reshape(top_3_preds.shape)

submission_df['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]


submission_df.to_csv('fsubmission6.csv', index=False)
FileLink("fsubmission6.csv")


