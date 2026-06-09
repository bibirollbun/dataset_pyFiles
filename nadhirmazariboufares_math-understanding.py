import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 4
EPOCHS = 1
LR = 2e-5
SEED = 42

# =============================
# SEED FIX
# =============================
def seed_everything(seed=SEED):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
display(train.head())
display(test.head())


def preprocess(df):
    df["input_text"] = (
        "Question: " + df["QuestionText"].fillna("") +
        " Answer: " + df["MC_Answer"].astype(str) +
        " Explanation: " + df["StudentExplanation"].fillna("")
    )
    return df

train = preprocess(train)
test = preprocess(test)

# Label encoding for Category
labels = train["Category"].unique().tolist()
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
train["label"] = train["Category"].map(label2id)



from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('/kaggle/input/sentence-transformersall-minilm-l6-v2/other/default/1/all-MiniLM-L6-v2', device=device)


# Example traint text
texts = train['input_text'].tolist()  # combined Question + Answer + Explanation

# Compute embeddings
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)


# Example test text
texts_tests = test['input_text'].tolist()  # combined Question + Answer + Explanation

# Compute embeddings
test_embeddings = model.encode(texts_tests, show_progress_bar=True, batch_size=64)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import f1_score
import joblib

print("train shape", train.shape, "embeddings shape", embeddings.shape)
print("test shape", test.shape, "test_embeddings shape", test_embeddings.shape)


from sklearn.utils.class_weight import compute_class_weight

cat_le = LabelEncoder()
y_cat = cat_le.fit_transform(train['Category'].astype(str))
cat_classes = list(cat_le.classes_)
print("Category classes:", cat_classes)

# Simple LightGBM with Stratified K-Fold
oof_probs = np.zeros((len(train), len(cat_classes)))
test_probs = np.zeros((len(test), len(cat_classes)))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


for fold, (tr_idx, va_idx) in enumerate(skf.split(embeddings, y_cat)):
    X_tr, y_tr = embeddings[tr_idx], y_cat[tr_idx]
    X_va, y_va = embeddings[va_idx], y_cat[va_idx]
    classes = np.unique(y_tr)
    cw = compute_class_weight('balanced', classes=classes, y=y_tr)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
    clf = lgb.LGBMClassifier(
    objective='multiclass',
    device='gpu',
    num_class=len(cat_classes),
    n_estimators=500,        
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    min_child_samples=5,
    class_weight=class_weights,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
)

    clf.fit(X_tr, y_tr,
            eval_set=[(X_va,y_va)])
    oof_probs[va_idx] = clf.predict_proba(X_va)
    test_probs += clf.predict_proba(test_embeddings) / skf.n_splits
    joblib.dump(clf, f"lgb_cat_fold{fold}.pkl")
    
oof_preds = np.argmax(oof_probs, axis=1)


print("Stage1 OOF macro F1:", f1_score(y_cat, oof_preds, average='macro'))


joblib.dump(cat_le, "cat_label_encoder.pkl")
np.save("cat_test_probs.npy", test_probs)


mask = train['Misconception'].notna()
train_mis = train[mask].reset_index(drop=True)
emb_mis = embeddings[mask]


mis_le = LabelEncoder()
y_mis = mis_le.fit_transform(train_mis['Misconception'].astype(str))
mis_classes = list(mis_le.classes_)
print("Number of misconception classes:", len(mis_classes))


oof_probs_mis = np.zeros((len(train_mis), len(mis_classes)))
test_probs_mis = np.zeros((len(test), len(mis_classes)))
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



from lightgbm import early_stopping, log_evaluation

for fold, (tr_idx, va_idx) in enumerate(skf2.split(emb_mis, y_mis)):
    X_tr, y_tr = emb_mis[tr_idx], y_mis[tr_idx]
    X_va, y_va = emb_mis[va_idx], y_mis[va_idx]
    clf_m = lgb.LGBMClassifier(
        device='gpu',
        objective='multiclass',
        num_class=len(mis_classes),
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        verbose=-1,
    )
    clf_m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=50),   # â�¹ï¸� stops if no improvement
            log_evaluation(period=50)              # ğŸ“Š prints every 50 rounds
        ]
    )
    oof_probs_mis[va_idx] = clf_m.predict_proba(X_va)
    test_probs_mis += clf_m.predict_proba(test_embeddings) / skf2.n_splits
    joblib.dump(clf_m, f"lgb_mis_fold{fold}.pkl")


oof_preds_mis = np.argmax(oof_probs_mis, axis=1)
print("Stage2 OOF macro F1 (on mis rows):", f1_score(y_mis, oof_preds_mis, average='macro'))

joblib.dump(mis_le, "mis_label_encoder.pkl")
np.save("mis_test_probs.npy", test_probs_mis)


# Function to produce up to top_k Category:Misconception strings per row
def build_predictions(cat_probs, mis_probs, top_k=3):
    out = []
    for i in range(cat_probs.shape[0]):
        # top categories indices and probs
        top_cat_idxs = np.argsort(-cat_probs[i])[:top_k]
        row_preds = []
        for idx in top_cat_idxs:
            cat_name = cat_le.inverse_transform([idx])[0]
            if 'Misconception' in cat_name:  # e.g., "True_Misconception"
                # get the top misconception(s) from mis_probs
                top_mis_idxs = np.argsort(-mis_probs[i])[:1]  # choose top-1 mis for each cat
                mis_name = mis_le.inverse_transform([top_mis_idxs[0]])[0]
                row_preds.append(f"{cat_name}:{mis_name}")
            else:
                row_preds.append(f"{cat_name}:NA")
        out.append(" ".join(row_preds))
    return out




submission_entries = build_predictions(test_probs, test_probs_mis, top_k=3)


# For test set:
sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = submission_entries
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

