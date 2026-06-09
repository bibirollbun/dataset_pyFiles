import pandas as pd
import numpy as np
import spacy
import re
from tqdm import tqdm

KAGGLE = True
tqdm.pandas()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

if not KAGGLE:
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
else:
    train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
    
train.head()


from sklearn.preprocessing import LabelEncoder

cat_enc = LabelEncoder()
misc_enc = LabelEncoder()


def text_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n+", "", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s_]", "", text).strip()
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)


def dataframe_preprocess(
    data: pd.DataFrame,
    cat_enc: LabelEncoder,
    misc_enc: LabelEncoder,
    is_train : False = True
) -> pd.DataFrame:
    df = data.copy()
    if is_train:
        df.Misconception = df.Misconception.fillna("NA").astype(str)
        df["Category:Misconception"] = df.Category + ":" + df.Misconception
        cat_enc.fit(df.Category)
        misc_enc.fit(df.Misconception)
        
        df["target_cat"] = cat_enc.transform(df.Category)
        df["target_misc"] = misc_enc.transform(df.Misconception)

    df["sentence"] = (
        "Quetion: " + df.QuestionText.astype(str) + 
        " Answer: " + df["MC_Answer"].astype(str) +
        " Explanation: " + df["StudentExplanation"].astype(str)
    )
    df.sentence = df.sentence.progress_apply(text_preprocess)
    return df


train_pre = dataframe_preprocess(train, cat_enc, misc_enc, is_train=True)


test_pre = dataframe_preprocess(test, cat_enc, misc_enc, is_train=False)


train_pre.Category.value_counts()


train_pre.Misconception.value_counts()


train_pre.Category.value_counts(ascending=True).head().plot.barh(title="Top 5 Category")


train_pre.Misconception.value_counts(ascending=False).head().plot.barh(title="Top 5 Misconception")


train_pre["Category:Misconception"].value_counts(ascending=False).head().plot.barh(title="Top 5 Category:Misconception")


train_pre["Category:Misconception"].value_counts()


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", max_df=0.95, min_df=2, stop_words="english")
tfidf.fit(pd.concat([train_pre["sentence"], test_pre["sentence"]]))

train_embeddings = tfidf.transform(train_pre["sentence"])
test_embeddings = tfidf.transform(test_pre["sentence"])

print("Train sparse shape is :", train_embeddings.shape)
print("Test sparse shape is  :", test_embeddings.shape)


def custom_train_test_split(X, y, splitter):
    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        yield X_train, X_test, y_train, y_test


from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
import xgboost as xgb


n_splits = 4
skf = StratifiedKFold(n_splits=n_splits)
test_cat_pred = np.zeros((test.shape[0], train_pre.target_cat.nunique()))

oof_cat_pred = list()
oof_cat_true = list()
params = {
    "eta": 0.025,
    "max_depth": 16,
    "objective": "multi:softprob",
    "subsample": 0.6,
    "lambda": 4,
    "tree_method": "auto",
    "grow_policy": "lossguide",
    "sampling_method": "gradient_based",
    "device": "cuda",
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.5,
    "colsample_bynode": 0.5,
}


for X_train_cat, X_val_cat, y_train_cat, y_val_cat in tqdm(
    custom_train_test_split(train_embeddings, train_pre.target_cat, skf),
    total=n_splits
):
    weight_train = compute_sample_weight("balanced", y_train_cat)
    dtrain = xgb.DMatrix(X_train_cat, y_train_cat, weight=weight_train)
    dval = xgb.DMatrix(X_val_cat, y_val_cat)
    dtest = xgb.DMatrix(test_embeddings)
    
    params.update({"num_class": y_train_cat.nunique()})
    
    model_cat = xgb.train(
        params, 
        dtrain,
        num_boost_round=1000, 
        early_stopping_rounds=100,
        evals=[(dval, "val")],
        verbose_eval=False
    )
    oof_cat_pred.append(model_cat.predict(dval, iteration_range=(1, model_cat.best_iteration)))
    oof_cat_true.append(y_val_cat.values)
    test_cat_pred += model_cat.predict(dtest, iteration_range=(1, model_cat.best_iteration)) / n_splits


oof_cat_pred = np.concatenate(oof_cat_pred)
oof_cat_true = np.concatenate(oof_cat_true)
cat_pred = np.argmax(oof_cat_pred, 1)


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt 


cr = classification_report(oof_cat_true, cat_pred, target_names=cat_enc.classes_)
ConfusionMatrixDisplay.from_predictions(oof_cat_true, cat_pred, display_labels=cat_enc.classes_, xticks_rotation=45, colorbar=False)
plt.title("Model - Category")
print(cr)
print("ROC_AUC:", roc_auc_score(oof_cat_true, oof_cat_pred, multi_class="ovr", average="weighted"))


test_cat = np.argsort(-test_cat_pred, 1)[:, :3]
test_cat


oof_misc_pred = list()
oof_misc_true = list()
test_misc_pred = np.zeros((test.shape[0], train_pre.target_misc.nunique()))


for X_train_misc, X_val_misc, y_train_misc, y_val_misc in tqdm(
    custom_train_test_split(train_embeddings, train_pre.target_misc, skf),
    total=n_splits
):
    weight_train = compute_sample_weight("balanced", y_train_misc)
    dtrain = xgb.DMatrix(X_train_misc, y_train_misc, weight=weight_train)
    dval = xgb.DMatrix(X_val_misc, y_val_misc)
    dtest = xgb.DMatrix(test_embeddings)
    
    params.update({"num_class": y_train_misc.nunique()})
    
    model_misc = xgb.train(
        params, 
        dtrain,
        num_boost_round=1000, 
        early_stopping_rounds=100,
        evals=[(dval, "val")],
        verbose_eval=False
    )
    oof_misc_pred.append(model_misc.predict(dval, iteration_range=(1, model_misc.best_iteration)))
    oof_misc_true.append(y_val_misc.values)
    test_misc_pred += model_misc.predict(dtest, iteration_range=(1, model_misc.best_iteration)) / n_splits


oof_misc_pred = np.concatenate(oof_misc_pred)
oof_misc_true = np.concatenate(oof_misc_true)
misc_pred = np.argmax(oof_misc_pred, 1)


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt 


cr = classification_report(oof_misc_true, cat_pred, target_names=misc_enc.classes_)
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
ConfusionMatrixDisplay.from_predictions(oof_misc_true, cat_pred, xticks_rotation=45, ax=ax, colorbar=False)
plt.title("Model - Misconception")
print(cr)
print("ROC_AUC:", roc_auc_score(oof_misc_true, oof_misc_pred, multi_class="ovr", average="weighted"))


test_misc = np.argsort(-test_misc_pred, 1)[:, :3]
test_misc


oof_misc_pred


predict = []
cv_pred_cat = oof_cat_pred
cv_pred_misc = oof_misc_pred
cv_pred_cat_top_3 = np.argsort(-cv_pred_cat, 1)[:, :3]
cv_pred_misc_top_3 = np.argsort(-cv_pred_misc, 1)[:, :3]

for i in tqdm(list(range(len(cv_pred_cat_top_3)))):
    pred = []
    for j in range(3):
        p1 = cat_enc.inverse_transform([cv_pred_cat_top_3[i, j]])[0]
        p2 = misc_enc.inverse_transform([cv_pred_misc_top_3[i, 0]])[0]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(pred)


true_cat_misc = cat_enc.inverse_transform(oof_cat_true) + ":" + misc_enc.inverse_transform(oof_misc_true)
true_cat_misc


print('ACCURACY_1')
print( np.mean(true_cat_misc == [p[0] for p in predict]) )
print('ACCURACY_2')
print( np.mean(true_cat_misc == [p[1] for p in predict]) )
print('ACCURACY_3')
print( np.mean(true_cat_misc == [p[2] for p in predict]) )


def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score+=1.
        elif t == p[1]:
            score+=1/2
        elif t == p[2]:
            score+=1/3
    return score / len(target_list)
        
print(f"MAP@3: {map3(true_cat_misc.tolist(), predict)}") # 0.892 CV


predict = []
for i in tqdm(list(range(len(test_cat)))):
    pred = []
    for j in range(3):
        p1 = cat_enc.inverse_transform([test_cat[i, j]])[0]
        p2 = misc_enc.inverse_transform([test_misc[i, 0]])[0]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(" ".join(pred))

if not KAGGLE:
    sub = pd.read_csv("data/sample_submission.csv")
else:
    sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
    
sub['Category:Misconception'] = predict
sub.to_csv("submission.csv", index=False)
sub

