import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import optuna
import warnings

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


train = pd.read_csv('../input/playground-series-s5e7/train.csv').drop(columns=['id'])
test = pd.read_csv('../input/playground-series-s5e7/test.csv').drop(columns=['id'])
sub = pd.read_csv('../input/playground-series-s5e7/sample_submission.csv')


df = pd.concat([train, test])
df['Personality'] = df['Personality'].fillna('Extrovert')
df = df.drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
df


df.shape


df.info()


NUM_COL = df.select_dtypes(include='float64').columns
CAT_COL = df.select_dtypes(include='object').columns

NUM_COL, CAT_COL


plt.figure(figsize=(5, 4))
plt.title('Numerical Features Relationship', fontsize=16)
sns.heatmap(df[NUM_COL].corr(), annot=True, cmap='coolwarm')
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(18, 5))
axs = axs.flatten()

for idx, col in enumerate(CAT_COL):
    ax = axs[idx]
    sns.countplot(x=col, data=df, ax=ax, palette='viridis')
    
    ax.set_xlabel(f'{col}') 
    ax.set_title(f'{col} Data Distributions')


miss_val_cnt = df.isnull().sum()
miss_val_percent = round(100 * miss_val_cnt / len(df), 2)
miss_val = pd.concat([miss_val_cnt, miss_val_percent], axis=1).rename(columns={0: 'Count', 1: 'Percentage'})
miss_val = miss_val[miss_val['Count'] > 0].sort_values(by='Percentage', ascending=False)
miss_val


TARGET = 'Personality'


imputer = SimpleImputer(strategy='most_frequent')
knn_imp = KNNImputer(n_neighbors=4)

CAT_COL = df.select_dtypes(include='object').drop(columns=TARGET).columns
for col in CAT_COL:
    # df.loc[:, col] = imputer.fit_transform(df[[col]])
    df[col] = df[col].fillna('missing')

NUM_COL = df.select_dtypes(include='float64').columns
for col in NUM_COL:
    df.loc[:, col] = knn_imp.fit_transform(df[[col]])    

df.isnull().sum()


df = pd.get_dummies(df, columns=CAT_COL, prefix=['Stage', 'Drained'])
df.head(5)


le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
df[TARGET] = le.fit_transform(df[TARGET])

df.head(5)


plt.figure(figsize=(10, 6))
sns.boxplot(data=df[NUM_COL])


sns.pairplot(df, vars=NUM_COL)


for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')


df.dtypes


train_df = df.iloc[:train.shape[0]]
test_df = df.iloc[train.shape[0]:]

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
X_test = test_df.drop(columns=TARGET).copy()


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_proba = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        oof_preds_proba[val_idx] = model.predict(dval)

    fold_logloss = log_loss(y, oof_preds_proba)
    
    return fold_logloss

study = optuna.create_study(direction='minimize', study_name='XGBoost CV Tuning')

study.optimize(objective, n_trials=50)


best_params = study.best_params
print(f"Best Parameters: {best_params}")

best_logloss = study.best_value
print(f"Best Score: {best_logloss}")


best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'logloss'
best_params['random_state'] = 42

print("Parameter terbaik yang akan digunakan:")
print(best_params)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_proba = np.zeros(len(X))
test_preds_proba = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y)), ncols=500):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        best_params, dtrain, num_boost_round=1000,
        evals=[(dval, "valid")],
        early_stopping_rounds=50, verbose_eval=False
    )
    
    oof_preds_proba[val_idx] = model.predict(dval)
    test_preds_proba += model.predict(dtest) / skf.n_splits

final_cv_logloss = log_loss(y, oof_preds_proba)
oof_preds_binary = (oof_preds_proba > 0.5).astype(int)
final_cv_accuracy = accuracy_score(y, oof_preds_binary)

print(f"\nSkor Logloss Cross-Validation Final: {final_cv_logloss:.4f}")
print(f"Skor Akurasi Cross-Validation Final: {final_cv_accuracy:.4f}")

final_test_preds_binary = (test_preds_proba > 0.5).astype(int)


sub["Personality"] = le.inverse_transform(final_test_preds_binary)
sub['Personality'] = sub['Personality'].replace({'Yes': 'Introvert', 'No': 'Extrovert'})
sub.to_csv("submission.csv", index=False)
sub

