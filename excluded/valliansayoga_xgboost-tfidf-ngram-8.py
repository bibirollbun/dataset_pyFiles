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


train_pre.sentence.head()[0]


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(ngram_range=(1, 8), analyzer="char", max_df=0.95, min_df=5, stop_words="english", use_idf=False)
tfidf.fit(pd.concat([train_pre["sentence"], test_pre["sentence"]]))

train_embeddings = tfidf.transform(train_pre["sentence"])
test_embeddings = tfidf.transform(test_pre["sentence"])

print("Train sparse shape is :", train_embeddings.shape)
print("Test sparse shape is  :", test_embeddings.shape)


from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from tqdm.notebook import tqdm
import xgboost as xgb


X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    train_embeddings,
    train_pre.target_cat,
    stratify=train_pre.target_cat, 
    test_size=0.2,
    random_state=42
)

sample_weight = compute_sample_weight("balanced", y_train_cat)


params = {
    "objective": "multi:softprob",
    "num_class": train_pre.target_cat.nunique(),
    "device": "cuda",
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "colsample_bylevel": 0.9,
    "grow_policy": "lossguide",
    "lambda": 0.1,
    "tree_method": "hist",
    "eta": 0.06,
    "max_depth": 10,
}

dtrain = xgb.DMatrix(X_train_cat, y_train_cat, weight=sample_weight)
dtest = xgb.DMatrix(X_test_cat, y_test_cat)
model_cat = xgb.train(params, dtrain, evals=[(dtest, "test")], num_boost_round=500, verbose_eval=50, early_stopping_rounds=50)


cv_pred_cat = model_cat.predict(dtest)
cat_pred = cv_pred_cat.argmax(axis=1)
cat_pred_proba = cv_pred_cat[np.arange(0, cat_pred.shape[0]), cv_pred_cat.argmax(axis=1)]


test_cat = np.argsort(-model_cat.predict(xgb.DMatrix(test_embeddings)), 1)[:, :3]
test_cat


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt 


cr = classification_report(y_test_cat, cat_pred, target_names=cat_enc.classes_)
ConfusionMatrixDisplay.from_predictions(y_test_cat, cat_pred, display_labels=cat_enc.classes_, xticks_rotation=45)
plt.title("Model - Category")
print(cr)
print("ROC_AUC:", roc_auc_score(y_test_cat, cv_pred_cat, multi_class="ovr", average="weighted"))


from sklearn import clone


# model_misc = clone(model_misc)

X_train_misc, X_test_misc, y_train_misc, y_test_misc = train_test_split(
    train_embeddings,
    train_pre.target_misc,
    stratify=train_pre.target_misc, 
    test_size=0.2,
    random_state=42
)
params = {
    "objective": "multi:softprob",
    "num_class": train_pre.target_misc.nunique(),
    "device": "cuda",
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "colsample_bylevel": 0.9,
    "grow_policy": "lossguide",
    "tree_method": "hist",
    "eta": 0.06,
    "lambda": 0.1,
    "max_depth": 10,
}
sample_weight = compute_sample_weight("balanced", y_train_misc)
dtrain = xgb.DMatrix(X_train_misc, y_train_misc, weight=sample_weight)
dtest = xgb.DMatrix(X_test_misc, y_test_misc)
model_misc = xgb.train(params, dtrain, evals=[(dtrain, "train"), (dtest, "test")], num_boost_round=500, verbose_eval=50, early_stopping_rounds=50)


test_misc = np.argsort(-model_misc.predict(xgb.DMatrix(test_embeddings)), 1)[:, :3]
test_misc


cv_pred_misc = model_misc.predict(xgb.DMatrix(X_test_misc))
misc_pred = cv_pred_misc.argmax(axis=1)
misc_pred_proba = cv_pred_misc[np.arange(0, misc_pred.shape[0]), cv_pred_misc.argmax(axis=1)]


cr = classification_report(misc_enc.inverse_transform(y_test_misc), misc_enc.inverse_transform(misc_pred))

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ConfusionMatrixDisplay.from_predictions(y_test_misc, misc_pred, xticks_rotation=45, ax=ax)
plt.title("Model - Misconception")
print(cr)
print("ROC_AUC:", roc_auc_score(y_test_misc, cv_pred_misc, multi_class="ovr", average="weighted"))


predict = []
cv_pred_cat = model_cat.predict(xgb.DMatrix(train_embeddings))
cv_pred_misc = model_misc.predict(xgb.DMatrix(train_embeddings))
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


print('ACCURACY_1')
print( np.mean(train_pre['Category:Misconception'] == [p[0] for p in predict]) )
print('ACCURACY_2')
print( np.mean(train_pre['Category:Misconception'] == [p[1] for p in predict]) )
print('ACCURACY_3')
print( np.mean(train_pre['Category:Misconception'] == [p[2] for p in predict]) )


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
        
print(f"MAP@3: {map3(train_pre['Category:Misconception'].tolist(), predict)}") # 0.892 CV


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

