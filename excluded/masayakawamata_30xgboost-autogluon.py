!pip install -q autogluon.tabular ray==2.10.0


from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings
import shutil
import glob

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e4/train.csv'
    test_path = '/kaggle/input/playground-series-s5e4/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'
    
    oof_path = '/kaggle/input/podcast-oof'
    
    target = 'Listening_Time_minutes'
    n_folds = 5
    seed = 42
    
    time_limit = 3600 * 6


train = pd.read_csv(CFG.train_path, index_col='id')

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]


def get_data(path, X, y, CFG):
    oof_files  = sorted(glob.glob(os.path.join(path, 'oof_preds_*.csv')))
    test_files = sorted(glob.glob(os.path.join(path, 'test_preds_*.csv')))
    assert len(oof_files) == len(test_files), "oof と test の数が一致しません"

    oof_dict   = {}
    test_dict  = {}
    score_dict = {}

    for oof_fp, test_fp in zip(oof_files, test_files):
        version = os.path.basename(oof_fp) \
                     .replace('oof_preds_', '') \
                     .replace('.csv', '')

        oof_vals  = pd.read_csv(oof_fp)['oof'].to_numpy()
        test_vals = pd.read_csv(test_fp)['Listening_Time_minutes'].to_numpy()

        scores = []
        kf = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
        for _, val_idx in kf.split(X):
            y_true = y[val_idx]
            y_pred = oof_vals[val_idx]
            scores.append(mean_squared_error(y_true, y_pred, squared=False))

        oof_dict[version]   = oof_vals
        test_dict[version]  = test_vals
        score_dict[version] = scores

    return oof_dict, test_dict, score_dict

BASE_PATH = CFG.oof_path  # '/kaggle/input/podcast-oof' など

oof_preds, test_preds, scores = get_data(BASE_PATH, X, y, CFG)

for v in sorted(scores):
    print(v, scores[v])


train = pd.DataFrame(oof_preds)
train[CFG.target] = y

test = pd.DataFrame(test_preds)


kf = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = kf.split(train)
for i, (_, val_index) in enumerate(split):
    train.loc[val_index, 'fold'] = i


predictor = TabularPredictor(
    problem_type='regression',
    eval_metric='root_mean_squared_error',
    label=CFG.target,
    groups='fold',
    verbosity=2
)


predictor.fit(
    train_data=train,
    time_limit=CFG.time_limit,
    presets='best_quality',
    ag_args_fit={
        'num_gpus': 2, 
        'num_cpus': 4
    }
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


def get_ensemble_weights(predictor):
    info = predictor.info()
    ensemble_weights = {}
    for model_name, values in info["model_info"].items():
        if "Ensemble" in model_name:
            children_info = values["children_info"]
            ensemble_weights[model_name] = values["children_info"][list(children_info.keys())[0]]["model_weights"]
    return ensemble_weights


ensemble_weights = get_ensemble_weights(predictor)

for key, value in ensemble_weights.items():
    plt.figure(figsize=(8, 8))
    plt.pie(value.values(), labels=value.keys(), autopct='%1.1f%%', colors=sns.color_palette('Set2', len(value)))
    plt.title(key)
    plt.tight_layout()
    plt.show()


split = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True).split(train)
for fold_idx, (train_index, val_index) in enumerate(split):
    for model in predictor.model_names():
        model_oof_preds = predictor.predict_oof(model).values
        fold_score = mean_squared_error(train.loc[val_index, CFG.target], model_oof_preds[val_index], squared=False)
        if model not in scores:
            scores[model] = []
        scores[model].append(fold_score)


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.4))

sns.boxplot(data=scores, order=order, ax=axs[0], orient='h', palette='RdYlGn_r')
axs[0].set_title('Fold RMSE')
axs[0].set_xlabel('')
axs[0].set_ylabel('')

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], palette='RdYlGn_r')
axs[1].set_title('Average RMSE')
axs[1].set_xlabel('')
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel('')

for i, score in enumerate(mean_scores.values):
    barplot.text(score, i, round(score, 6), va='center')

plt.tight_layout()
plt.show()


sub = pd.read_csv(CFG.sample_sub_path)
sub[CFG.target] = predictor.predict(test).values
sub.to_csv(f'sub_autogluon_{np.mean(scores[predictor.model_best]):.6f}.csv', index=False)


sub


shutil.rmtree("AutogluonModels")

