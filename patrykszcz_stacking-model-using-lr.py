import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb 
import matplotlib.pyplot as plt 
import seaborn as sns 
import optuna
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcolors
import optuna.visualization as vis
import warnings
import logging
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(logging.WARNING)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col = 'id')
origin = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')


target = 'Fertilizer Name'
cat_columns = [i for i in train.columns if train[i].dtype == np.object_][:-1]
num_columns = [i for i in train.columns if i not in cat_columns][:-1]
print(f'Target column:       {target}')
print(f'Categorical columns: {cat_columns}')
print(f'Numeric columns:     {num_columns}')


## FOR kde line
def darker(color, amount=0.5):
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = color 

    return tuple(max(min(x * amount, 1), 0) for x in c)

colors = sns.color_palette('Set3', 3)

fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(12, len(num_columns) * 3))
axs = np.atleast_2d(axs)

for i, feature in enumerate(num_columns):
    # histogram + ciemniejsza linia KDE dla train
    sns.histplot(train[feature], bins=20, kde=False, color=colors[0], ax=axs[i, 0], label='Train', stat='density')
    sns.kdeplot(train[feature], color=darker(colors[0], 0.5), lw=3, ax=axs[i, 0])
    axs[i, 0].set_title(f'Histogram - Train - {feature}')
    axs[i, 0].grid(color='gray', linestyle=':', linewidth=1)

    # histogram + ciemniejsza linia KDE dla origin
    sns.histplot(origin[feature], bins=20, kde=False, color=colors[2], ax=axs[i, 1], label='Original', stat='density')
    sns.kdeplot(origin[feature], color=darker(colors[2], 0.5), lw=3, ax=axs[i, 1])
    axs[i, 1].set_title(f'Histogram - Original - {feature}')
    axs[i, 1].grid(color='gray', linestyle=':', linewidth=1)

    # histogram + ciemniejsza linia KDE dla test
    sns.histplot(test[feature], bins=20, kde=False, color=colors[1], ax=axs[i, 2], label='Test', stat='density')
    sns.kdeplot(test[feature], color=darker(colors[1], 0.5), lw=3, ax=axs[i, 2])
    axs[i, 2].set_title(f'Histogram - Test - {feature}')
    axs[i, 2].grid(color='gray', linestyle=':', linewidth=1)

plt.tight_layout()
plt.show()


plt.figure(figsize = (8,7))
sns.heatmap(train.corr(numeric_only=True),annot=True)
plt.tight_layout()
plt.title('Coreletion numeric features')
plt.show()


## Label and ordinal encoding
label_enc = LabelEncoder()
ordinal_enc = OrdinalEncoder(handle_unknown='error')

train[cat_columns] = ordinal_enc.fit_transform(train[cat_columns])
test[cat_columns] = ordinal_enc.transform(test[cat_columns])
origin[cat_columns] = ordinal_enc.transform(origin[cat_columns])

train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])
origin['Fertilizer Name'] = label_enc.transform(origin['Fertilizer Name'])

train['const'] = 1
test['const'] =1
origin['const'] =1

train = train.astype('category')
test = test.astype('category')
origin = origin.astype('category')
print(f'Preprocessing data - DONE')


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



X = train.drop(['Fertilizer Name'],axis = 1)
y = train["Fertilizer Name"]
X_origin = origin.drop(['Fertilizer Name'],axis = 1)
y_origin = origin["Fertilizer Name"]


FOLDS =5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_1 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_1 = np.zeros(shape = (len(test),y.nunique()))


params = {
    'max_depth': 7,
    'colsample_bytree': 0.36078847485284476,
    'subsample': 0.7812245246658889,
    'learning_rate': 0.058219706172072075,
    'alpha': 6.820101448722005,
    'reg_lambda': 5.480081652527049,
    'min_child_weight': 2,
    'max_bin': 128,
    'device': "cpu",
    'tree_method': 'hist',
    'eval_metric': "mlogloss",
    'objective': 'multi:softprob',
    'num_class': 7,
    
}

print('Starting training XGB models...')
for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15,'FOLD:', i+1, '#' *15)

    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]
    x_test = test.copy()
    
    ## EXTRA DATA
    X_origin_expanded = X_origin.copy()
    y_origin_expanded = y_origin.copy()
    
    for _ in range(5):  
        X_origin_expanded = pd.concat([X_origin_expanded, X_origin.copy()], axis=0, ignore_index=True)
        y_origin_expanded = pd.concat([y_origin_expanded, y_origin.copy()], axis=0, ignore_index=True)


    x_train = pd.concat([x_train, X_origin_expanded], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, y_origin_expanded], axis=0, ignore_index=True)

    print(x_train.shape)
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=500
    )

    actual = [[label] for label in y_valid]
    oof_1[valid_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    pred_prob_1 += model.predict(dtest, iteration_range=(0, model.best_iteration + 1))/ FOLDS

    top_3_preds = np.argsort(oof_1[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3 MODEL_2 Score: {map3_score:.5f}")
    

actual = [[label] for label in y]

top_3_preds = np.argsort(oof_1, axis=1)[:, -3:][:, ::-1]  
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final MODEL_2 MAP@3 Score: {map3_score:.5f}')


oof_2 = np.load('/kaggle/input/model-xgb-240625/oof_240625_1.npy')  ## LB: 0.37997
oof_3 = np.load('/kaggle/input/models-xgb-250625/oof_250625_1.npy')
oof_4 = np.load('/kaggle/input/models-xgb-250625/oof_250625_2.npy')
oof_lgb_5 = np.load('/kaggle/input/models-lgb/lgb_oof.npy')
oof_lgbgoss_6 = np.load('/kaggle/input/models-lgb/lgb_goss_oof.npy')


pred_prob_2 = np.load('/kaggle/input/model-xgb-240625/pred_240625_1.npy')
pred_prob_3 = np.load('/kaggle/input/models-xgb-250625/pred_250625_1.npy')
pred_prob_4 = np.load('/kaggle/input/models-xgb-250625/pred_250625_2.npy')
pred_prob_lgb_5 = np.load('/kaggle/input/models-lgb/lgb_test.npy')
pred_prob_lgbgoss_6 = np.load('/kaggle/input/models-lgb/lgb_goss_test.npy')


X_meta = np.column_stack([oof_1,oof_2,oof_3,oof_4,oof_lgb_5,oof_lgbgoss_6])
x_test = np.column_stack([pred_prob_1,pred_prob_2,pred_prob_3,pred_prob_4,pred_prob_lgb_5,pred_prob_lgbgoss_6])


def objective(trial):

    C = trial.suggest_float('C', 1e-3, 3.0)
    tol = trial.suggest_float('tol', 1e-5, 1e-1)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs', 'sag', 'newton-cg', 'newton-cholesky'])
    max_iter = trial.suggest_int('max_iter', 800, 2000)

    # COMBINATION solverâ€“penalty
    valid_combinations = {
        'liblinear': ['l1', 'l2'],
        'saga': ['l1', 'l2', 'elasticnet'],
        'lbfgs': ['l2'],
        'sag': ['l2'],
        'newton-cg': ['l2'],
        'newton-cholesky': ['l2']
    }

    if penalty not in valid_combinations[solver]:
        raise optuna.exceptions.TrialPruned()

    # elasticnet - l1_ratio
    l1_ratio = None
    if penalty == 'elasticnet' and solver == 'saga':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    elif penalty == 'elasticnet':
        raise optuna.exceptions.TrialPruned()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X_meta), y.nunique()))

    for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
        print('#' * 15, f" FOLD {i+1} ", '#' * 15)
        x_train, x_valid = X_meta[train_idx], X_meta[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        try:
            model_params = {
                'C': C,
                'tol': tol,
                'penalty': penalty,
                'solver': solver,
                'max_iter': max_iter,
                'fit_intercept':True
            }

            if l1_ratio is not None:
                model_params['l1_ratio'] = l1_ratio

            model = LogisticRegression(**model_params)
            model.fit(x_train, y_train)

            oof[valid_idx] = model.predict_proba(x_valid)
            actual = [[label] for label in y_valid]
            top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
            map3_score = mapk(actual, top_3_preds)
            print(f"âœ… FOLD {i+1}: MAP@3  Score: {map3_score:.5f}")
        except:
            return 0.0  

    actual = [[label] for label in y]
    top_3_preds = np.argsort(oof, axis=1)[:, -3:][:, ::-1]
    score = mapk(actual, top_3_preds)
    print(f'SCORE: {score}')
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial:")
best_params = study.best_trial.params
print(best_params)


plt.figure(figsize=(18,10))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout()
plt.show()


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_stack = np.zeros(shape = (len(train), y.nunique()))
preds_folds = []
print(f"Starting 10-Folds LR training...")
for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
    lr_model = LogisticRegression(**best_params,
        fit_intercept=True
    )

    x_train,x_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train,y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    lr_model.fit(x_train, y_train)

    oof_stack[valid_idx] = lr_model.predict_proba(x_valid)
    pred_prob_stack =lr_model.predict_proba(x_test) 
    preds_folds.append(pred_prob_stack)
    actual = [[label] for label in y_valid]
    top_3_preds = np.argsort(oof_stack[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3  Score: {map3_score:.5f}")


actual = [[label] for label in y]

top_3_preds_1 = np.argsort(oof_stack, axis=1)[:, -3:][:, ::-1]  
map3_score_1 = mapk(actual, top_3_preds_1)
print(f'âœ… Final MAP@3 Score: {map3_score_1:.5f}')


top_3_preds = np.argsort(np.mean(preds_folds,axis=0), axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()

