import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping,log_evaluation,LGBMClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


path="/kaggle/input/wsdm-cup-multilingual-chatbot-arena/"
train = pd.read_parquet(path+"train.parquet")
test = pd.read_parquet(path+"test.parquet")
sub = pd.read_csv(path+"sample_submission.csv")


train.head()


test.head()


# split 20% data as validation data 
train,valid=train_test_split(train,test_size=0.2,stratify=train["winner"],random_state=161194)

# Train set can be inverted (and winner too) to get twice the data from the available training dataset
train_inv=train.copy()
train_inv["response_a"],train_inv["response_b"]=train_inv["response_b"],train_inv["response_a"]
train_inv["winner"]=train_inv["winner"].apply(lambda x: "model_a" if "b" in x else "model_b")


def compute_feats(df):
    for col in ["response_a","response_b","prompt"]:
        # response lenght is a key factor when choosing between two responses
        df[f"{col}_len"]=df[f"{col}"].str.len()

        # Some characters counting features 
        df[f"{col}_spaces"]=df[f"{col}"].str.count("\s")
        df[f"{col}_punct"]=df[f"{col}"].str.count(",|\.|!")
        df[f"{col}_question_mark"]=df[f"{col}"].str.count("\?")
        df[f"{col}_quot"]=df[f"{col}"].str.count("'|\"")
        df[f"{col}_formatting_chars"]=df[f"{col}"].str.count("\*|\_")
        df[f"{col}_math_chars"]=df[f"{col}"].str.count("\-|\+|\=")
        df[f"{col}_curly_open"]=df[f"{col}"].str.count("\{")
        df[f"{col}_curly_close"]=df[f"{col}"].str.count("}")
        df[f"{col}_round_open"]=df[f"{col}"].str.count("\(")
        df[f"{col}_round_close"]=df[f"{col}"].str.count("\)")
        df[f"{col}_accent_chars"]=df[f"{col}"].str.count("è|ò|à|ù|é|ì")
        df[f"{col}_special_chars"]=df[f"{col}"].str.count("\W")
        df[f"{col}_digits"]=df[f"{col}"].str.count("\d")/df[f"{col}_len"]
        df[f"{col}_lower"]=df[f"{col}"].str.count("[a-z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_upper"]=df[f"{col}"].str.count("[A-Z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_chinese"]=df[f"{col}"].str.count(r'[\u4e00-\u9fff]+').astype("float32")/df[f"{col}_len"]
        df[f"{col}_tild"]=df[f"{col}"].str.count("~")>0

        # Feature that show how balanced are curly and round brackets
        df[f"{col}_round_balance"]=df[f"{col}_round_open"]-df[f"{col}_round_close"]
        df[f"{col}_curly_balance"]=df[f"{col}_curly_open"]-df[f"{col}_curly_close"]

        # Feature that tells if the string json is present somewhere (e.g. asking a json response or similar)
        df[f"{col}_json"]=df[f"{col}"].str.lower().str.count("json")
        df[f"{col}_yaml"]=df[f"{col}"].str.lower().str.count("yaml")

    return df
train=compute_feats(train)
train_inv=compute_feats(train_inv)

train=pd.concat([train,train_inv])
valid=compute_feats(valid)
test=compute_feats(test)


vectorizer_char = TfidfVectorizer(sublinear_tf=True, analyzer='char', ngram_range=(1,2), max_features=100_000)
vectorizer_word = TfidfVectorizer(sublinear_tf=True, analyzer='word', min_df=3)
preprocessor = ColumnTransformer(
    transformers=[
        ('prompt_feats', FeatureUnion([
            ('prompt_char', vectorizer_char),
            ('prompt_word', vectorizer_word)
        ]), 'prompt'),
        ('response_a_feats', FeatureUnion([
            ('response_a_char', vectorizer_char),
            ('response_a_word', vectorizer_word)
        ]), 'response_a'),
        ('response_b_feats', FeatureUnion([
            ('response_b_char', vectorizer_char),
            ('response_b_word', vectorizer_word)
        ]), 'response_b')
    ]
)
train_feats = preprocessor.fit_transform(train[["response_a","response_b","prompt"]])
test_feats = preprocessor.transform(test[["response_a","response_b","prompt"]])
valid_feats = preprocessor.transform(valid[["response_a","response_b","prompt"]])

model = LogisticRegression(C=0.1, solver='liblinear', dual=True, random_state=42)
model.fit(train_feats, train.winner)


model.predict_proba(test_feats)


train.columns


train.head()


feats=list(train.columns)[8:]
train["winner"]=(train["winner"]=="model_a").astype("int")
valid["winner"]=(valid["winner"]=="model_a").astype("int")

X=train[feats]
y=train["winner"]

X_val=valid[feats]
y_val=valid["winner"]


# Save data after feature engineering
train[feats + ['winner']].to_parquet("train_processed.parquet", index=False)
valid[feats + ['winner']].to_parquet("valid_processed.parquet", index=False)
test[feats].to_parquet("test_processed.parquet", index=False)

'''
# Load data of feature engineering

train_processed = pd.read_parquet("train_processed.parquet")
valid_processed = pd.read_parquet("valid_processed.parquet")
test_processed = pd.read_parquet("test_processed.parquet")

X_train = train_processed[feats]
y_train = train_processed['winner']
X_valid = valid_processed[feats]
y_valid = valid_processed['winner']
X_test = test_processed
'''


model = LGBMClassifier(
    n_estimators=1343,                  # 最佳树数量
    learning_rate=0.05721006996592793,  # 最佳学习率
    max_depth=9,                        # 最佳树深度
    num_leaves=44,                      # 最佳叶子节点数
    subsample=0.804822520556406,        # 最佳样本采样比例
    colsample_bytree=0.9973291744305393, # 最佳特征采样比例
    objective='binary',                 # 二分类任务
    boosting_type='gbdt',               # 梯度提升树
    random_state=42                     # 保持结果可复现
)

history=model.fit(
    X,y,
    eval_set=(X_val,y_val),
    eval_metric=["binary_error", "logloss", "auc"],
    callbacks=[early_stopping(100),log_evaluation(100)]
)


X_test=test[feats]
test["winner"]=model.predict(X_test)
test["winner"]=test["winner"].apply(lambda x: "model_a" if x==1 else "model_b")

sub=test[["id","winner"]]


sub.head()


sub.to_csv("submission.csv",index=False)

