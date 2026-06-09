from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold
import pandas as pd
import warnings
import joblib
import shutil

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
    test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
    sample_sub_path = "/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv"

    n_folds = 10
    seed = 42

    time_limit = 3600 * 11
    run_name = "ag_classifier"


train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)


kf = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = kf.split(train, train.efs)

for i, (train_index, val_index) in enumerate(split):
    train.loc[val_index, "fold"] = i


train = train.drop(["ID", "efs_time"], axis=1)
test = test.drop(["ID"], axis=1)


predictor = TabularPredictor(
    path=f"/{CFG.run_name}",
    problem_type="binary",
    eval_metric="roc_auc",
    label="efs",
    groups="fold",
    verbosity=2
)


predictor.fit(
    train_data=train,
    time_limit=CFG.time_limit,
    presets="best_quality",
    excluded_model_types=["KNN"],
    keep_only_best=True
)


predictor.leaderboard(silent=True).style.background_gradient(subset=["score_val"], cmap="RdYlGn")


oof_preds = predictor.predict_oof()
oof_pred_probs = predictor.predict_proba_oof()


joblib.dump(oof_preds, "oof_preds.pkl")
joblib.dump(oof_pred_probs, "oof_pred_probs.pkl")


shutil.make_archive(
    f"/kaggle/working/{CFG.run_name}", 
    "zip", 
    f"/{CFG.run_name}"
)

