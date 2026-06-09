import pandas as pd
import seaborn as sns
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk import tokenize
from matplotlib import pyplot as plt
import regex as re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, metrics
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



text_data = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
eval_data = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")


seven_text_data = text_data[text_data['RDizzl3_seven'] == True]


# Remove the " " " in the prompt names
seven_text_data['prompt_name'] = seven_text_data['prompt_name'].str.replace('''"''','')
# seven_text_data['prompt_name'].unique()


seven_text_data.groupby('label').count()


# seven_text_data['length'] = seven_text_data['text'].str.len()
# seven_text_data.head()


# mapping = dict(zip(seven_text_data['prompt_name'].unique(), range(7)))
# seven_text_data['prompt_id'] = seven_text_data.replace(mapping)['prompt_name']



mapping_source = dict(zip(seven_text_data['source'].unique(), range(len(seven_text_data['source'].unique()))))
seven_text_data['source_id'] = seven_text_data.replace(mapping_source)['source']
sorted_seven_text_data = seven_text_data.groupby('source').count().sort_values("text", ascending=False)
# sns.barplot(sorted_seven_text_data,x = 'text', y = 'source', orient = 'h')
# plt.title("Number of essays by source")


def word_count(text):
    text_list = text.split()
    return len(text_list)



seven_text_data['word_count'] = seven_text_data['text'].apply(word_count)
# seven_text_data.head()


seven_text_data['mean_word_length'] = seven_text_data['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()])
)


seven_text_data['mean_sent_length'] = seven_text_data['text'].apply(
    lambda x: np.mean([len(sent) for sent in tokenize.sent_tokenize(x)])
)



def normalize(text):
    # Replace with whitespace to separate 'ðŸ˜ƒ\n\nFor'
    text = text.replace(r"\n", r" ")
    text = text.replace(r"\r", r" ")
    # Drop punctuation
    text = re.sub(r"\p{P}", " ", text)
    # Remove extra spaces from 'ðŸ˜ƒ  For' to 'ðŸ˜ƒ For'
    text = re.sub(r"\s+", r" ", text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text

normalized_seven_text_data = seven_text_data.copy()
normalized_seven_text_data['text'] = seven_text_data['text'].apply(lambda x: normalize(x))



eval_data['text'] = eval_data['text'].apply(lambda x: normalize(x))


normalized_seven_text_data.sum(numeric_only=True)['label']/normalized_seven_text_data.shape[0]


dropped_persuade_index = normalized_seven_text_data[
    normalized_seven_text_data['source']=='persuade_corpus'
                                                    ].sample(n=8080, random_state=1).index


sub_sampled_text_data = normalized_seven_text_data.drop(dropped_persuade_index).reset_index(drop=True)


sub_sampled_text_data.sum(numeric_only=True)['label']/sub_sampled_text_data.shape[0]


corpus = sub_sampled_text_data['text']
vectorizer = TfidfVectorizer(max_features = 10000,
                            stop_words = 'english',
                            )

vectorizer2 = TfidfVectorizer(ngram_range=(1, 4),
                             tokenizer=lambda x: re.findall(r'[^\W]+', x),
                             stop_words='english',
                             token_pattern=None,
                             strip_accents='unicode',
                             sublinear_tf=True,
                             max_features=50000
                             )

X = vectorizer2.fit_transform(corpus,
                            )


X_eval = vectorizer2.transform(eval_data['text'])


features = vectorizer2.get_feature_names_out()


features


print("vectorized corpus dimensions : ", X.shape)
print("corpus dataset dimensions : ", seven_text_data.shape)


# X.mean(axis = 0)


y = sub_sampled_text_data['label'].values
y.shape


X_train, X_test, y_train, y_test = train_test_split(X,y)


reg = linear_model.LogisticRegression()


reg.fit(X_train,y_train)


def get_scores(classifier, X_test, y_test):
    metrics_dict = {}
    y_pred = classifier.predict(X_test)
    metrics_dict['f1'] = metrics.f1_score(y_test,y_pred)
    metrics_dict['auc'] = roc_auc_score(y_test,classifier.predict_proba(X_test)[:, 1])
    return metrics_dict


y_pred = reg.predict(X_test)


metrics.f1_score(y_test,y_pred)


roc_auc_score(y_test,reg.predict_proba(X_test)[:, 1])


# svc = SVC(gamma='auto',max_iter=1000, tol=1e-3, probability = True)
# svc.fit(X_train,y_train)


# svc.predict(X_test)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


get_scores(rf, X_test, y_test)


# boost = XGBClassifier()
# boost.fit(X_train, y_train)



# get_scores(boost, X_test, y_test)


rf = RandomForestClassifier()
rf.fit(X,y) # taking all data


print(X_eval)


y_eval_pred = rf.predict(X_eval)
y_eval_pred_proba = rf.predict_proba(X_eval)


y_eval_pred_proba[:,1]


eval_data['id']



d = {'id' : eval_data['id'], 'generated': y_eval_pred_proba[:,1]}
df_submission = pd.DataFrame(data=d)
df_submission.head()




df_submission.to_csv("submission.csv", index=False)

