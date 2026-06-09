!pip install autogluon.tabular scikit-learn==1.5.2


from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import pickle
import os

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/playground-series-s5e7/train.csv"
    test_path = "/kaggle/input/playground-series-s5e7/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s5e7/sample_submission.csv"

    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    target = "Personality"
    n_folds = 5
    seed = 42
    
    log_path = "/logs"
    time_limit = 3600 * 11


train = pd.read_csv(CFG.train_path, index_col="id")
test = pd.read_csv(CFG.test_path, index_col="id")

# Reference: https://www.kaggle.com/code/paddykb/ps-s5e7-don-t-look-at-me
original = pd.read_csv(CFG.original_path)
original = original.rename(columns={'Personality': 'match_p'})
original = original.drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'])
train = train.merge(original, how='left')
test = test.merge(original, how='left')


skf = StratifiedKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = skf.split(train, train[CFG.target])
for i, (_, val_index) in enumerate(split):
    train.loc[val_index, "fold"] = i


predictor = TabularPredictor(
    path=CFG.log_path,
    problem_type="binary",
    eval_metric="accuracy",
    label=CFG.target,
    groups="fold",
    verbosity=2
)


predictor.fit(
    train_data=train,
    time_limit=CFG.time_limit,
    presets="best_quality"
)


predictor.leaderboard(silent=True).style.background_gradient(subset=["score_val"], cmap="RdYlGn")


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
    plt.pie(value.values(), labels=value.keys(), autopct="%1.1f%%", colors=sns.color_palette("Set2", len(value)))
    plt.title(key)
    plt.tight_layout()
    plt.show()


def save_pred_probs(pred_probs, cv_score, name, type, is_ensemble):
    base_path = "oof_pred_probs" if type == "oof" else "test_pred_probs"
    base_path = "." if is_ensemble else base_path
    with open(f"{base_path}/{name}_{type}_pred_probs_{cv_score:.6f}.pkl", "wb") as f:
        pickle.dump(pred_probs, f)

def save_submission(test, test_preds, score):        
    sub = pd.read_csv(CFG.sample_sub_path)
    sub[CFG.target] = test_preds
    sub.loc[test.match_p == "Extrovert", CFG.target] = "Introvert"
    sub.loc[test.match_p == "Introvert", CFG.target] = "Extrovert"
    sub.to_csv(f"sub_autogluon_{score:.6f}.csv", index=False)
    
os.makedirs("oof_pred_probs", exist_ok=True)
os.makedirs("test_pred_probs", exist_ok=True)


oof_pred_probs = {}
test_pred_probs = {}
oof_preds = {}


best_model = predictor.model_best
_test_pred_probs = predictor.predict_proba_multi(test)
for model in predictor.model_names():
    model_oof_pred_probs = predictor.predict_proba_oof(model).values[:, 1]
    model_oof_preds = predictor.predict_oof(model).values
    model_test_pred_probs = _test_pred_probs[model].values[:, 1]
    
    cv_score = accuracy_score(train[CFG.target], model_oof_preds)
    if model != best_model:
        save_pred_probs(model_oof_pred_probs, cv_score, model, "oof", False)
        save_pred_probs(model_test_pred_probs, cv_score, model, "test", False)
    else:
        save_pred_probs(model_oof_pred_probs, cv_score, model, "oof", True)
        save_pred_probs(model_test_pred_probs, cv_score, model, "test", True)
        save_submission(test, predictor.predict(test).values, cv_score)
        
    oof_pred_probs[model] = model_oof_pred_probs
    test_pred_probs[model] = model_test_pred_probs
    oof_preds[model] = model_oof_preds


scores = {}
split = StratifiedKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True).split(train, train[CFG.target])
for fold_idx, (train_index, val_index) in enumerate(split):
    for model in predictor.model_names():
        fold_score = accuracy_score(train.loc[val_index, CFG.target], oof_preds[model][val_index])
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

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.2))

sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", palette="RdYlGn_r")
axs[0].set_title("Fold Accuracy")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], palette="RdYlGn_r")
axs[1].set_title("Average Accuracy")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, score in enumerate(mean_scores.values):
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()

