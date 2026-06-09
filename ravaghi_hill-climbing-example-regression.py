!pip install hill-climbing


from hill_climbing import Climber, ClimberCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import glob

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/playground-series-s4e12/train.csv"
    test_path = "/kaggle/input/playground-series-s4e12/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s4e12/sample_submission.csv"

    oof_path = "/kaggle/input/hill-climbing-example-datasets/regression"
    
    target = "Premium Amount"
    n_folds = 10
    seed = 42


train = pd.read_csv(CFG.train_path, index_col="id")
X, y = train.drop(CFG.target, axis=1), np.log1p(train[CFG.target])


scores, oof_preds, test_preds = {}, {}, {}


paths = glob.glob(f"{CFG.oof_path}/*")
for path in paths:
    files = glob.glob(f"{path}/*")
    temp_oof_preds = joblib.load([file for file in files if "oof_preds" in file][0])
    temp_test_preds = joblib.load([file for file in files if "test_preds" in file][0])
    temp_oof_preds = np.log1p(temp_oof_preds)
    temp_test_preds = np.log1p(temp_test_preds)
    model_name = path.split("/")[-1]
        
    temp_scores = []        
    kf = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
    for _, val_idx in kf.split(X, y):
        temp_scores.append(np.sqrt(mean_squared_error(y[val_idx], temp_oof_preds[val_idx])))
    
    scores[model_name] = temp_scores
    oof_preds[model_name] = temp_oof_preds
    test_preds[model_name] = temp_test_preds


X = pd.DataFrame(oof_preds)
X_test = pd.DataFrame(test_preds)


X.head()


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


climber = Climber(
    objective="minimize",
    eval_metric=rmsle,
    allow_negative_weights=True,
    precision=0.001,
    score_decimal_places=6
)


climber.fit(X, y)


scores["climber"] = [climber.best_score] * CFG.n_folds


sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_context("notebook", font_scale=1.2)

def add_annotations(ax, x, y):
    for xi, yi in zip(x, y):
        ax.annotate(
            f'{yi:.6f}', (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            va='bottom',
            fontsize=8
        )

palette = sns.color_palette("deep")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

sns.lineplot(x=climber.history.model[1:], y=climber.history.coef[1:], color=palette[0], marker='x', ax=ax1, linewidth=2)
add_annotations(ax1, climber.history.model[1:], climber.history.coef[1:])
ax1.set_ylabel('Coefficient', fontsize=12)
ax1.spines[['top', 'right']].set_visible(False)

sns.lineplot(x=climber.history.model[1:], y=climber.history.improvement[1:], color=palette[1], marker='o', ax=ax2, linewidth=2)
add_annotations(ax2, climber.history.model[1:], climber.history.improvement[1:])
ax2.set_ylabel('Improvement', fontsize=12)
ax2.spines[['top', 'right']].set_visible(False)

sns.lineplot(x=climber.history.model[1:], y=climber.history.score[1:], color=palette[2], marker='*', ax=ax3, linewidth=2)
add_annotations(ax3, climber.history.model[1:], climber.history.score[1:])
ax3.set_ylabel('Score', fontsize=12)
ax3.set_xlabel('Model', fontsize=12)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.suptitle('Hill Climbing History', y=1.02, fontsize=14, fontweight='bold')

for ax in [ax1, ax2, ax3]:
    ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1)

plt.show()


preds = climber.predict(X_test)
preds[:25]


climber_cv = ClimberCV(
    objective="minimize",
    eval_metric=rmsle,
    allow_negative_weights=True,
    precision=0.001,
    score_decimal_places=6,
    cv=KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True),
)


climber_cv.fit(X, y)


scores["climber-cv"] = climber_cv.fold_scores


fold_1_history = climber_cv.history[climber_cv.history.fold == 1]

palette = sns.color_palette("deep")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

sns.lineplot(x=fold_1_history.model[1:], y=fold_1_history.coef[1:], color=palette[0], marker='x', ax=ax1, linewidth=2)
add_annotations(ax1, fold_1_history.model[1:], fold_1_history.coef[1:])
ax1.set_ylabel('Coefficient', fontsize=12)
ax1.spines[['top', 'right']].set_visible(False)

sns.lineplot(x=fold_1_history.model[1:], y=fold_1_history.train_score[1:], color=palette[1], marker='o', ax=ax2, linewidth=2)
add_annotations(ax2, fold_1_history.model[1:], fold_1_history.train_score[1:])
ax2.set_ylabel('Train Score', fontsize=12)
ax2.spines[['top', 'right']].set_visible(False)

sns.lineplot(x=fold_1_history.model[1:], y=fold_1_history.val_score[1:], color=palette[2], marker='*', ax=ax3, linewidth=2)
add_annotations(ax3, fold_1_history.model[1:], fold_1_history.val_score[1:])
ax3.set_ylabel('Validation Score', fontsize=12)
ax3.set_xlabel('Model', fontsize=12)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.suptitle('Hill Climbing History', y=1.02, fontsize=14, fontweight='bold')

for ax in [ax1, ax2, ax3]:
    ax.tick_params(labelsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1)

plt.show()


preds = climber_cv.predict(X_test)
preds[:25]


sns.reset_defaults()

scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=True)
order = scores.mean().sort_values(ascending=True).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.3))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient='h', color='grey')
axs[0].set_title('Fold Score')
axs[0].set_xlabel('')
axs[0].set_ylabel('')

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color='grey')
axs[1].set_title('Average Score')
axs[1].set_xlabel('')
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel('')

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = 'cyan' if 'climber' in model.lower() else 'grey'
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va='center')

plt.tight_layout()
plt.show()

