import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.special import logit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import shutil
import optuna
import json
import os
import gc

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s4e11/train.csv'
    test_path = '/kaggle/input/playground-series-s4e11/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s4e11/sample_submission.csv'
    
    original_data_path = '/kaggle/input/depression-surveydataset-for-analysis/final_depression_dataset_1.csv'
    
    target = 'Depression'
    n_folds = 5
    seed = 42


_train = pd.read_csv(CFG.train_path, index_col='id')
_test = pd.read_csv(CFG.test_path, index_col='id')


depression_counts = _train.Depression.value_counts(normalize=True).reset_index()
depression_counts.columns = ['Depression', 'Proportion']
palette = {0: sns.color_palette("Set1")[1], 1: sns.color_palette("Set1")[0]}
class_mapping = {0: 'Not Depressed', 1: 'Depressed'}

plt.figure(figsize=(10, 6))
sns.barplot(x='Depression', y='Proportion', data=depression_counts, palette=palette)

plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.ylim(0, 1)

for index, row in depression_counts.iterrows():
    plt.text(row.name, row.Proportion + 0.02, f'{class_mapping[row.name]} ({row.Proportion:.2%})', ha='center', va='bottom', fontsize=12)

sns.despine()
plt.tight_layout()
plt.show()


cat_cols = _train.select_dtypes(include='object').columns.tolist()
palette = {0: sns.color_palette("Set1")[1], 1: sns.color_palette("Set1")[0]}

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(data=_train, x=col, hue=CFG.target, ax=axes[i], palette=palette)
    axes[i].set_title(f'{col} distribution')
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(
    x=_train[cat_cols].nunique().sort_values(ascending=False).values,
    y=_train[cat_cols].nunique().sort_values(ascending=False).index,
    ax=ax,
    color="grey"
)
ax.set_title('Unique values in categorical columns')

for i, value in enumerate(_train[cat_cols].nunique().sort_values(ascending=False).values):
    ax.text(value, i, f'{value}', va='center')

sns.despine()
plt.show()


train_null_values = _train.drop(columns=CFG.target).isnull().T
test_null_values = _test.isnull().T

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

sns.heatmap(train_null_values, cbar=False, ax=axes[0], cmap=sns.light_palette('grey', as_cmap=True))
axes[0].set_title('Train')
axes[0].set_xticks([]) 
axes[0].set_xlabel('')

sns.heatmap(test_null_values, cbar=False, ax=axes[1], cmap=sns.light_palette('grey', as_cmap=True))
axes[1].set_title('Test')
axes[1].set_xticks([]) 
axes[1].set_xlabel('')

plt.tight_layout()
plt.show()


def get_data(model):
    train = pd.read_csv(CFG.train_path, index_col='id')
    test = pd.read_csv(CFG.test_path, index_col='id')
    original = pd.read_csv(CFG.original_data_path)
    original[CFG.target] = original[CFG.target].map({'Yes': 1, 'No': 0})
    
    if model == 'cb':
        cat_cols = test.columns.tolist()
    else:
        cat_cols = test.select_dtypes(include='object').columns.tolist()
        
    train[cat_cols] = train[cat_cols].astype(str).astype('category')
    test[cat_cols] = test[cat_cols].astype(str).astype('category')
    original[cat_cols] = original[cat_cols].astype(str).astype('category')
        
    for col in cat_cols:
        common_categories = train[col].cat.categories.union(original[col].cat.categories)
        train[col] = train[col].cat.set_categories(common_categories)
        original[col] = original[col].cat.set_categories(common_categories)
    
    for col in train.columns:
        original[col] = original[col].astype(train[col].dtype)

    if model in ['gb', 'adb']:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train[cat_cols] = encoder.fit_transform(train[cat_cols])
        test[cat_cols] = encoder.transform(test[cat_cols])
        original[cat_cols] = encoder.transform(original[cat_cols])
    
        cols_with_missing_values = train.columns[train.isnull().mean() > 0]
        imputer = SimpleImputer(strategy='mean')
        train[cols_with_missing_values] = imputer.fit_transform(train[cols_with_missing_values])
        test[cols_with_missing_values] = imputer.transform(test[cols_with_missing_values])
        original[cols_with_missing_values] = imputer.transform(original[cols_with_missing_values])
    
    X = train.drop(CFG.target, axis=1)
    y = train[CFG.target]
    X_test = test
    
    X_original = original.drop(CFG.target, axis=1)
    y_original = original[CFG.target]
    
    return X, y, X_test, X_original, y_original


class Trainer:
    def __init__(self, model, config=CFG, is_ensemble_model=False):
        self.model = model
        self.config = config
        self.is_ensemble_model = is_ensemble_model

    def fit_predict(self, X, y, X_test, X_original=None, y_original=None, threshold=0.5):
        print(f'Training {self.model.__class__.__name__}\n')
        
        scores = []        
        coeffs = np.zeros((1, X.shape[1]))
        oof_pred_probs = np.zeros(X.shape[0])
        test_pred_probs = np.zeros(X_test.shape[0])
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if not self.is_ensemble_model:                                
                X_train = pd.concat([X_train, X_original], ignore_index=True)
                y_train = pd.concat([y_train, y_original], ignore_index=True)
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            if self.is_ensemble_model:
                coeffs += model.coef_ / self.config.n_folds
                if isinstance(self.model, LogisticRegression):
                    n_iters = model.n_iter_[0]
            
            y_pred_probs = model.predict(X_val) if isinstance(self.model, Ridge) else model.predict_proba(X_val)[: ,1]
            oof_pred_probs[val_idx] = y_pred_probs 
            
            temp_test_pred_probs = model.predict(X_test) if isinstance(self.model, Ridge) else model.predict_proba(X_test)[:, 1]
            test_pred_probs += temp_test_pred_probs / self.config.n_folds
            
            score = accuracy_score(y_val, (y_pred_probs > threshold).astype(int))
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
            if self.is_ensemble_model and isinstance(self.model, LogisticRegression):
                print(f'--- Fold {fold_idx + 1} - Accuracy: {score:.6f} ({n_iters} iterations)')
            else:
                print(f'--- Fold {fold_idx + 1} - Accuracy: {score:.6f}')
            
        overall_score = accuracy_score(y, (oof_pred_probs > threshold).astype(int))
            
        print(f'\n------ Overall: {overall_score:.6f} | Average: {np.mean(scores):.6f} ± {np.std(scores):.6f}')
        
        if self.is_ensemble_model:
            return oof_pred_probs, test_pred_probs, scores, coeffs
        else:
            os.makedirs('oof_pred_probs', exist_ok=True)
            os.makedirs('test_pred_probs', exist_ok=True)
            self._save_pred_probs(oof_pred_probs, np.mean(scores), 'oof')
            self._save_pred_probs(test_pred_probs, np.mean(scores), 'test')
            return oof_pred_probs, test_pred_probs, scores
        
    def fit(self, X, y, threshold=0.5):
        scores = []                
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict(X_val) if isinstance(self.model, Ridge) else model.predict_proba(X_val)[:, 1]  
            score = accuracy_score(y_val, (y_pred_probs > threshold).astype(int))
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect() 
        
        return np.mean(scores)
        
    def _save_pred_probs(self, pred_probs, cv_score, name):
        model_name = self.model.__class__.__name__.lower().replace('classifier', '')
        with open(f'{name}_pred_probs/{model_name}_{name}_pred_probs_{cv_score:.6f}.pkl', 'wb') as f:
            pickle.dump(pred_probs, f)


def save_submission(name, test_pred_probs, score, threshold=0.5):
    sub = pd.read_csv(CFG.sample_sub_path)
    sub[CFG.target] = (test_pred_probs > threshold).astype(int)
    sub.to_csv(f'sub_{name}_{score:.6f}.csv', index=False)
    return sub.head()


lr_params = {
    "C": 5.559884346567435,
    "max_iter": 1000,
    "n_jobs": 4,
    "penalty": "l2",
    "random_state": 42,
    "solver": "newton-cg",
    "tol": 0.08313081991676836
}

cb_params = {
    "border_count": 180,
    "colsample_bylevel": 0.7351678905666684,
    "depth": 4,
    "iterations": 2372,
    "l2_leaf_reg": 4.442847441200204,
    "learning_rate": 0.0514109059943355,
    "min_child_samples": 146,
    "random_state": 42,
    "random_strength": 0.18678416655567043,
    "scale_pos_weight": 1.019889465491297,
    "subsample": 0.3511896501762123,
    "verbose": False
}

xgb_params = {
    "colsample_bylevel": 0.25155109886677396,
    "colsample_bynode": 0.5723191165109757,
    "colsample_bytree": 0.18034301813835885,
    "enable_categorical": True,
    "gamma": 3.6392698070258622,
    "max_bins": 26161,
    "max_depth": 16,
    "max_leaves": 67,
    "min_child_weight": 34,
    "n_estimators": 3853,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 7.996080341061729,
    "reg_lambda": 46.83054555763492,
    "scale_pos_weight": 1.2157646356820928,
    "subsample": 0.9117754083869292,
    "verbosity": 0
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.18283018243382332,
    "learning_rate": 0.09945326391012832,
    "max_bins": 36644,
    "min_child_samples": 105,
    "min_child_weight": 0.2083765599710974,
    "n_estimators": 244,
    "n_jobs": -1,
    "num_leaves": 122,
    "random_state": 42,
    "reg_alpha": 8.662578235164972,
    "reg_lambda": 3.5696291074963926,
    "scale_pos_weight": 1.0733293968870794,
    "subsample": 0.5360642841695424,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.11905309670044416,
    "learning_rate": 0.04641567005485582,
    "max_bins": 18501,
    "min_child_samples": 283,
    "min_child_weight": 0.5242575557671028,
    "n_estimators": 966,
    "n_jobs": -1,
    "num_leaves": 159,
    "random_state": 42,
    "reg_alpha": 8.160196501794122,
    "reg_lambda": 9.861767101758469,
    "scale_pos_weight": 0.991416082619142,
    "subsample": 0.9259030065865966,
    "verbose": -1
}

lgbm_dart_params = {
    "boosting_type": "dart",
    "colsample_bytree": 0.1181378019860333,
    "learning_rate": 0.0910272033595692,
    "max_bins": 7386,
    "min_child_samples": 225,
    "min_child_weight": 0.28492487885169293,
    "n_estimators": 1454,
    "n_jobs": -1,
    "num_leaves": 163,
    "random_state": 42,
    "reg_alpha": 2.9686583338116925,
    "reg_lambda": 9.781841345026509,
    "scale_pos_weight": 0.8309734952725212,
    "subsample": 0.511061530253864,
    "verbose": -1
}

gb_params = {
    'learning_rate': 0.129726178306296,
    'max_depth': 82,
    'max_features': 0.9815414831121304,
    'max_leaf_nodes': 45,
    'min_samples_leaf': 0.0015657248550179637,
    'min_samples_split': 0.3385622260186072,
    'min_weight_fraction_leaf': 0.03201568899384577,
    'n_estimators': 1048,
    'random_state': 42,
    'subsample': 0.9550206349439808
}

adb_params = {
    'learning_rate': 1.5174537517219886,
    'n_estimators': 410,
    'random_state': 42
}


scores = {}
oof_pred_probs = {}
test_pred_probs = {}


# Reference: https://www.kaggle.com/competitions/playground-series-s4e11/discussion/543746

X, y, X_test, X_original, y_original = get_data('lr')
lr_model = make_pipeline(
    OneHotEncoder(handle_unknown='ignore', min_frequency=0.001),
    LogisticRegression(**lr_params)
)
lr_trainer = Trainer(lr_model)
oof_pred_probs['LogisticRegression'], test_pred_probs['LogisticRegression'], scores['LogisticRegression'] = lr_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("cb")
cb_model = CatBoostClassifier(**cb_params, cat_features=X.columns.tolist())
cb_trainer = Trainer(cb_model)
oof_pred_probs['CatBoost'], test_pred_probs['CatBoost'], scores['CatBoost'] = cb_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("xgb")
xgb_model = XGBClassifier(**xgb_params)
xgb_trainer = Trainer(xgb_model)
oof_pred_probs['XGBoost'], test_pred_probs['XGBoost'], scores['XGBoost'] = xgb_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("lgbm")
lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_trainer = Trainer(lgbm_model)
oof_pred_probs['LightGBM'], test_pred_probs['LightGBM'], scores['LightGBM'] = lgbm_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("lgbm_goss")
lgbm_goss_model = LGBMClassifier(**lgbm_goss_params)
lgbm_goss_trainer = Trainer(lgbm_goss_model)
oof_pred_probs['LightGBM (goss)'], test_pred_probs['LightGBM (goss)'], scores['LightGBM (goss)'] = lgbm_goss_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("lgbm_dart")
lgbm_dart_model = LGBMClassifier(**lgbm_dart_params)
lgbm_dart_trainer = Trainer(lgbm_dart_model)
oof_pred_probs['LightGBM (dart)'], test_pred_probs['LightGBM (dart)'], scores['LightGBM (dart)'] = lgbm_dart_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("gb")
gb_model = GradientBoostingClassifier(**gb_params)
gb_trainer = Trainer(gb_model)
oof_pred_probs['Gradient Boosting'], test_pred_probs['Gradient Boosting'], scores['Gradient Boosting'] = gb_trainer.fit_predict(X, y, X_test, X_original, y_original)


X, y, X_test, X_original, y_original = get_data("adb")
adb_model = AdaBoostClassifier(**adb_params)
adb_trainer = Trainer(adb_model)
oof_pred_probs['AdaBoost'], test_pred_probs['AdaBoost'], scores['AdaBoost'] = adb_trainer.fit_predict(X, y, X_test, X_original, y_original)


def plot_weights(weights, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(list(oof_pred_probs.keys()))[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.5))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f'{value:.3f}', va='center', ha='left', color='black')
        else:
            ax.text(value, i, f'{value:.3f}', va='center', ha='right', color='black')

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


def objective(trial):
    solver_penalty_options = [
        ('liblinear', 'l1'),
        ('liblinear', 'l2'),
        ('lbfgs', 'l2'),
        ('lbfgs', None),
        ('newton-cg', 'l2'),
        ('newton-cg', None),
        ('newton-cholesky', 'l2'),
        ('newton-cholesky', None)
    ]
    solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_options)
    
    params = {
        'random_state': CFG.seed,
        'max_iter': 500,
        'C': trial.suggest_float('C', 0, 10),
        'tol': trial.suggest_float('tol', 1e-10, 1e-1),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'solver': solver,
        'penalty': penalty
    }
    
    threshold = trial.suggest_float('threshold', 0.4, 0.6, step=0.001)
    
    model = LogisticRegression(**params)
    trainer = Trainer(model, is_ensemble_model=True)
    return trainer.fit(X, y, threshold)

sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, n_jobs=-1)
best_params = study.best_params


l2_oof_pred_probs = {}
l2_test_pred_probs = {}


X = logit(pd.DataFrame(oof_pred_probs).clip(1e-15, 1-1e-15))
X_test = logit(pd.DataFrame(test_pred_probs).clip(1e-15, 1-1e-15))


solver, penalty = best_params['solver_penalty']
lr_params = {
    'random_state': CFG.seed,
    'max_iter': 500,
    'C': best_params['C'],
    'tol': best_params['tol'],
    'fit_intercept': best_params['fit_intercept'],
    'class_weight': best_params['class_weight'],
    'solver': solver,
    'penalty': penalty
}


print(json.dumps(lr_params, indent=2))


best_threshold = study.best_params['threshold']
print(f'Best threshold: {best_threshold:.3f}')


lr_model = LogisticRegression(**lr_params)
lr_trainer = Trainer(lr_model, is_ensemble_model=True)
l2_oof_pred_probs['l2-ensemble-lr'], l2_test_pred_probs['l2-ensemble-lr'], scores['L2 Ensemble LR'], lr_coeffs = lr_trainer.fit_predict(X, y, X_test, None, None, best_threshold)


save_submission('l2-ensemble-lr', l2_test_pred_probs['l2-ensemble-lr'], np.mean(scores['L2 Ensemble LR']), best_threshold)


plot_weights(lr_coeffs, 'LR Coefficients')


X = pd.DataFrame(oof_pred_probs)
X_test = pd.DataFrame(test_pred_probs)


def objective(trial):    
    params = {
        'random_state': CFG.seed,
        'alpha': trial.suggest_float('alpha', 0, 50),
        'tol': trial.suggest_float('tol', 1e-10, 1e-2),
        'positive': trial.suggest_categorical('positive', [True, False]),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
    }
    
    threshold = trial.suggest_float('threshold', 0.4, 0.6, step=0.001)
    
    model = Ridge(**params)
    trainer = Trainer(model, is_ensemble_model=True)
    return trainer.fit(X, y, threshold)


sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, n_jobs=-1)
best_params = study.best_params


ridge_params = {
    'random_state': CFG.seed,
    'alpha': best_params['alpha'],
    'tol': best_params['tol'],
    'positive': best_params['positive'],
    'fit_intercept': best_params['fit_intercept']
}


print(json.dumps(ridge_params, indent=2))


best_threshold = study.best_params['threshold']
print(f'Best threshold: {best_threshold:.3f}')


ridge_model = Ridge(**ridge_params)
ridge_trainer = Trainer(ridge_model, is_ensemble_model=True)
l2_oof_pred_probs['l2-ensemble-ridge'], l2_test_pred_probs['l2-ensemble-ridge'], scores['L2 Ensemble Ridge'], ridge_coeffs = ridge_trainer.fit_predict(X, y, X_test, None, None, best_threshold)


save_submission('l2-ensemble-ridge', l2_test_pred_probs['l2-ensemble-ridge'], np.mean(scores['L2 Ensemble Ridge']), best_threshold)


plot_weights(ridge_coeffs, 'Ridge Coefficients')


def objective(trial):
    weights = np.array([trial.suggest_float(l2_model_name, 0, 1) for l2_model_name in l2_oof_pred_probs.keys()])
    weights /= np.sum(weights)
    
    preds = np.zeros(len(y))
    for model, weight in zip(l2_oof_pred_probs.keys(), weights):
        preds += l2_oof_pred_probs[model] * weight
        
    threshold = trial.suggest_float('threshold', 0.4, 0.6, step=0.001)
            
    return accuracy_score(y, (preds > threshold).astype(int))

sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=-1)


scores['L3 Weighted Ensemble (LR + Ridge)'] = [study.best_value] * CFG.n_folds


best_weights = np.array([study.best_params[l2_model] for l2_model in l2_oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)
print(json.dumps({model: weight for model, weight in zip(l2_oof_pred_probs.keys(), best_weights)}, indent=2))

best_threshold = study.best_params['threshold']
print(f'\nBest threshold: {best_threshold:.3f}')


plt.figure(figsize=(8, 5))
plt.pie(
    [v for k,v in study.best_params.items() if k != "threshold"], 
    labels=[k for k,v in study.best_params.items() if k != "threshold"], 
    autopct='%1.1f%%',
    colors=sns.color_palette('Set2', 2)
)
plt.title("L3 Ensemble Weights")
plt.tight_layout()
plt.show()


weighted_test_preds = np.zeros((X_test.shape[0]))
for model, weight in zip(l2_test_pred_probs.keys(), best_weights):
    weighted_test_preds += l2_test_pred_probs[model] * weight
    
save_submission('l3-weighted-ensemble', weighted_test_preds, np.mean(scores['L3 Weighted Ensemble (LR + Ridge)']), best_threshold)


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.3))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient='h', color='grey')
axs[0].set_title('Fold Accuracy')
axs[0].set_xlabel('')
axs[0].set_ylabel('')

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color='grey')
axs[1].set_title('Average Accuracy')
axs[1].set_xlabel('')
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel('')

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = 'cyan' if 'ensemble' in model.lower() else 'grey'
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va='center')

plt.tight_layout()
plt.show()


shutil.rmtree('catboost_info', ignore_errors=True)

