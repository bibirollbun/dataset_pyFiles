import numpy as np
import pandas as pd
import xgboost as xgb


from scipy.sparse import hstack
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer


tr = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
ts = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
ss = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


tr.head()


tr.values[0]


# create combine text 
def combine_text(df):
    return (
        "subreddit: " + df['subreddit'] + "\n" +
        "rule: " + df['rule'] + "\n" + 
        "positive_1: " + df['positive_example_1'] + "\n" + 
        "positive_2: " + df['positive_example_2'] + "\n" + 
        "negative_1: " + df['negative_example_1'] + "\n" +
        "negative_2: " + df['negative_example_2'] + "\n" +
        "comment: " + df['body']
     ) 
Xtr_txt=combine_text(tr)
Xts_txt=combine_text(ts)


vectorizer = TfidfVectorizer(
    max_features = 60000,
    ngram_range = (1, 5),
    stop_words = 'english'
)


Xtrv = vectorizer.fit_transform(Xtr_txt)
Xtsv = vectorizer.transform(Xts_txt)
y = tr['rule_violation']


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
vl_auc, models= [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(Xtrv, y)):
    Xtr, Xvl = Xtrv[tr_idx], Xtrv[vl_idx]
    ytr, yvl = y.iloc[tr_idx], y.iloc[vl_idx]

    model = xgb.XGBClassifier(
        n_estimators=10000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaf=31,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        learning_rate=0.03,
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )

    model.fit(Xtr, ytr,
             eval_set=[(Xvl,yvl)],
             early_stopping_rounds=50,
             verbose=False)

    vlp=model.predict_proba(Xvl)[:, 1]
    auc=roc_auc_score(yvl, vlp)
    vl_auc.append(auc)
    models.append(model)
    print(f"Fold {fold + 1} Auc Score: {auc:.4f}")
print(f"\nMean CV Auc Score: {np.mean(vl_auc):.4f}")


fp = 0
for model in models:
    fp += model.predict_proba(Xtsv)[:, 1]

fp = fp / len(models)

submission = pd.DataFrame({
    'row_id': ss['row_id'],
    'rule_violation': fp
})

submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv saved!")

