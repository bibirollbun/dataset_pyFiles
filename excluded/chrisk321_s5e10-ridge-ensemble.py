import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print(f"This dataset has {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"There are {train.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(train.duplicated().sum())} duplicates in the dataset.")

# quick look at the data
train.head(3)


unique_id = 'id'
target = 'accident_risk'
categorical_columns = ['num_lanes', 'speed_limit', 'road_type', 'lighting',  'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']
numerical_columns = [ 'curvature', 'num_reported_accidents']


print('The unique identifier is unique.') if train[unique_id].nunique() == train.shape[0] else print('The unique identifier is not unique.')


skewness_threshold = .5 # can tune / experiment with this value
skewed_cols = [col for col in numerical_columns if train[col].skew() > skewness_threshold]

print(f'There are {len(skewed_cols)} skewed columns: {str(skewed_cols)}')


imbalance_threshold = 3.33
imbalanced_cols = [col for col in categorical_columns if (train[col].count() / train[col].value_counts().values.min()) > imbalance_threshold]

print(f'There are {len(imbalanced_cols)} imbalanced columns: {str(imbalanced_cols)}')


train_unique_cols = [x for x in train.drop([target],axis=1).columns if x not in test.columns]
print('All columns in train exist in test.') if not train_unique_cols else print(f'The following train columns are not in test: {train_unique_cols}')

test_unique_cols = [x for x in test.columns if x not in train.columns]
print('All columns in test exist in train.') if not test_unique_cols else print(f'The following test columns are not in train: {test_unique_cols}')


for col in numerical_columns:
    _, axes = plt.subplots(1,2,figsize=(10,5),sharex=False,sharey=False)
    
    sns.histplot(data=train, x=col,bins=20,ax=axes[0])
    axes[0].set_title(f'Distribution of {col}')

    sns.boxplot(data=train,x=col,ax=axes[1],showfliers=False)
    axes[1].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()


for col in categorical_columns:  
    _, axes = plt.subplots(1,1,figsize=(8,5),sharex=False,sharey=False)

    sns.barplot(data=train, x=col,y=target)
    axes.set_title(f'{col}')

    plt.tight_layout()
    plt.show()


print(f"This dataset has {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"There are {test.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(test.duplicated().sum())} duplicates in the dataset.")
# quick look at the data
test.head(3)


_, axes = plt.subplots(1,2,figsize=(10,5),sharex=False,sharey=False)

sns.histplot(data=train, x=target,bins=20,ax=axes[0])
axes[0].set_title(f'Distribution of {target}')

sns.boxplot(data=train,x=target,ax=axes[1],showfliers=False)
axes[1].set_title(f'{target}')

plt.tight_layout()
plt.show()


def pre_process(df,categorical_columns, numerical_columns, unique_id, target):

    for col in categorical_columns:
        df[col] = df[col].fillna('NA')
        df[col] = df[col].astype('category')

    return df

train_orig = pre_process(train.copy(),categorical_columns, numerical_columns, unique_id, target)
test_orig = pre_process(test.copy(),categorical_columns, numerical_columns, unique_id, target)


X = train_orig.drop([target, unique_id],axis=1)
y = train_orig[target]

test_ids = test_orig[unique_id]
X_test = test_orig.drop(unique_id,axis=1).copy()

cv_method = KFold(n_splits=5, shuffle=True, random_state=1)


baseline_models = []
# Initialize test predictions for ensembling
xgb_test_preds = np.zeros(test.shape[0])
lgbm_test_preds = np.zeros(test.shape[0])
cb_test_preds = np.zeros(test.shape[0])


scores = []
NFOLDS=5
xgb_oof = np.zeros((len(y)))

params = {'lambda': 1.154546885839723e-07, 'alpha': 0.23395911682704026, 'max_depth': 8, 'min_child_weight': 8, 'subsample': 0.9403238878070037, 'colsample_bytree': 0.884020682853199, 'eta': 0.08094832564214673}
boosters = ['gbtree']
for booster in boosters:
        for fold, (idx_tr, idx_va) in enumerate(cv_method.split(X, y), start=0):
            X_tr = X.iloc[idx_tr]
            X_va = X.iloc[idx_va]
            y_tr = y.iloc[idx_tr]
            y_va = y.iloc[idx_va]

            xgb_model = xgb.XGBRegressor(**params, booster=booster, device='cuda', enable_categorical=True, random_state=1, missing=np.inf)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],verbose=0)
            y_pred = xgb_model.predict(X_va)

            score = np.sqrt(mean_squared_error(y_va, y_pred))
            print(f"# Fold {fold}: {score=:.5f}")
            scores.append(score)
            xgb_oof[idx_va] = y_pred
            xgb_test_preds += xgb_model.predict(X_test) / NFOLDS

        print(f"# {booster} Overall score: {np.mean(scores)}: +/- {np.std(scores)}")
        baseline_models.append(('xgboost_'+ booster, str(np.mean(scores))))



scores = []
lgbm_oof = np.zeros((len(y)))

boosting_types = ['gbdt']
for booster in boosting_types:
    for fold, (idx_tr, idx_va) in enumerate(cv_method.split(X, y), start=1):
        X_tr = X.iloc[idx_tr]
        X_va = X.iloc[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]

        lgbm_model = lgb.LGBMRegressor(**params, boosting=booster, device='gpu', random_state=1,verbose=-1)

        lgbm_model.fit(X_tr, y_tr.values.ravel())
        y_pred = lgbm_model.predict(X_va)

        score = np.sqrt(mean_squared_error(y_va, y_pred))
        print(f"# Fold {fold}: {score=:.5f}")
        scores.append(score)
        lgbm_oof[idx_va] = y_pred
        lgbm_test_preds += lgbm_model.predict(X_test) / NFOLDS

    print(f"#{booster} Overall score: {np.mean(scores)}: +/- {np.std(scores)}")
    baseline_models.append(('lightgbm_'+ booster, str(np.mean(scores))))


scores = []
cb_oof = np.zeros((len(y)))

boosting_types = ['Plain']
params = {'learning_rate': 0.06918350235411831, 'depth': 8, 'l2_leaf_reg': 3.5607892331643987, 'min_data_in_leaf': 42, 'random_strength': 0.00038634802501875184, 'bagging_temperature': 0.005659763664000533}

for booster in boosting_types:
    for fold, (idx_tr, idx_va) in enumerate(cv_method.split(X, y), start=1):
        X_tr = X.iloc[idx_tr]
        X_va = X.iloc[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]

        cb_model = cb.CatBoostRegressor(**params, boosting_type=booster, task_type='GPU', random_state=1,cat_features = categorical_columns)
        cb_model.fit(X_tr, y_tr,verbose=False)
        y_pred = cb_model.predict(X_va)

        score = np.sqrt(mean_squared_error(y_va, y_pred))
        print(f"# Fold {fold}: {score=:.5f}")
        scores.append(score)
        cb_oof[idx_va] = y_pred
        cb_test_preds += cb_model.predict(X_test) / NFOLDS

    print(f"#{booster} Overall score: {np.mean(scores)}: +/- {np.std(scores)}")
    baseline_models.append(('catboost_'+ booster, str(np.mean(scores))))



baselines = pd.DataFrame(baseline_models, columns=['model_name', 'model_score']).sort_values(by='model_score',ascending=True) # Create DataFrame
baselines['model_score'] = baselines['model_score'].astype('float')

_, ax = plt.subplots(1,1,figsize=(8,5),sharex=False,sharey=False)
sns.barplot(data=baselines, x='model_name', y='model_score',color='steelblue')
plt.xticks(rotation=45, ha='right')
ax.set_title(f'Baseline Model RMSE Comparison')

for container in ax.containers:
    ax.bar_label(container)
plt.show()
plt.tight_layout()


import optuna
from sklearn.linear_model import Ridge
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
X_oof = pd.DataFrame({'xgb':xgb_oof,'lgb':lgbm_oof,'cb':cb_oof})
scores = []
def objective(trial, X_oof, y):
    params = {
        "alpha": trial.suggest_float("alpha", 0, 100),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2)
    }

    NFOLDS = 10
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_splits = cv.split(X_oof, y)
    scores = []

    ens_model = Ridge(**params,random_state=1)
    for train_idx, val_idx in cv_splits:
        X_train_fold, X_val_fold = X_oof.iloc[train_idx], X_oof.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        ens_model.fit(X_train_fold, y_train_fold)
        preds = ens_model.predict(X_val_fold)

        score = np.sqrt(mean_squared_error(y_val_fold, preds))
        scores.append(score)
    mean_score = np.mean(scores)
    return mean_score

study = optuna.create_study(sampler=TPESampler(n_startup_trials=30, seed=1),direction='minimize')

study.optimize(lambda trial: objective(trial, X_oof, y), n_trials=250)
print(study.best_value, study.best_params)


params = {'alpha': 0.4586653082709201, 'tol': 0.006005699739141118} # These are the best parameters for our Ridge model

NFOLDS = 5
scores = []
cv = KFold(n_splits=5, shuffle=True, random_state=1)
cv_splits = cv.split(X_oof, y)
X_test = pd.DataFrame({'xgb':xgb_test_preds,'lgb':lgbm_test_preds,'cb':cb_test_preds})
test_preds = np.zeros(X_test.shape[0])
model = Ridge(**params, random_state=1)
for train_idx, val_idx in cv_splits:
    X_train_fold, X_val_fold = X_oof.iloc[train_idx], X_oof.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train_fold, y_train_fold)
    preds = model.predict(X_val_fold)
    test_preds += model.predict(X_test) / NFOLDS

    score = np.sqrt(mean_squared_error(y_val_fold, preds))
    scores.append(score)
    print(f"# Fold {fold}: {score=:.5f}")

mean_score = np.mean(scores)
print(f"Mean  Score = {mean_score:.5f}")


submission = pd.DataFrame({unique_id:test[unique_id],target:test_preds})
submission.head(3)


submission.to_csv("submission.csv", index=False)

