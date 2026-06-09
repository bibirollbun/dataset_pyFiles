import pandas as pd
import seaborn as sns
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk import tokenize
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, metrics
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
import regex as re



text_data = pd.read_csv("/kaggle/input/train-v2-drcat-02/train_v2_drcat_02.csv")


csv_eval = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv") #ici on fait avec ce dataset pour essayer, on changera le path vers test_essay dans le notebook kaggle


text_data.head()


text_data.nunique()


seven_text_data = text_data[text_data['RDizzl3_seven'] == True]


seven_text_data.nunique()


seven_text_data=seven_text_data.reset_index()


seven_text_data['prompt_name'].unique()


# Remove the " " " in the prompt names
seven_text_data['prompt_name'] = seven_text_data['prompt_name'].str.replace('''"''','')
seven_text_data['prompt_name'].unique()


seven_text_data['source'].unique()


seven_text_data.head()


seven_text_counts = seven_text_data.groupby('label').count().reset_index()


sns.barplot(seven_text_counts, x = 'label', y = 'text')


seven_text_data['length'] = seven_text_data['text'].str.len()
seven_text_data.head()


#sns.histplot(seven_text_data, x = 'length', hue = 'label')


mapping = dict(zip(seven_text_data['prompt_name'].unique(), range(7)))
seven_text_data['prompt_id'] = seven_text_data.replace(mapping)['prompt_name']



#sns.barplot(seven_text_data.groupby('prompt_id').count(),x = 'prompt_id', y = 'text')


#mapping_source = dict(zip(seven_text_data['source'].unique(), range(len(seven_text_data['source'].unique()))))
#seven_text_data['source_id'] = seven_text_data.replace(mapping_source)['source']
#sorted_seven_text_data = seven_text_data.groupby('source').count().sort_values("text", ascending=False)
#sns.barplot(sorted_seven_text_data,x = 'text', y = 'source', orient = 'h')
#plt.title("Number of essays by source")


def word_count(text):
    text_list = text.split()
    return len(text_list)



seven_text_data['word_count'] = seven_text_data['text'].apply(word_count)
seven_text_data.head()


#sns.histplot(seven_text_data, x = 'word_count',kde=True, hue='label')


seven_text_data['mean_word_length'] = seven_text_data['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()])
)


#sns.histplot(seven_text_data, x = 'mean_word_length',kde=True, hue='label')


#seven_text_data['mean_sent_length'] = seven_text_data['text'].apply(
 #   lambda x: np.mean([len(sent) for sent in tokenize.sent_tokenize(x)])
#)



#sns.histplot(seven_text_data, x = 'mean_sent_length', hue='label')


random_idx = np.random.randint(seven_text_data.shape[0])
seven_text_data.iloc[random_idx]['text']


def normalize(texte):
    texte = texte.replace("\n", "")
    texte = texte.replace("\xa0", "")
    texte = texte.strip()
    texte = re.sub(r"\p{P}", " ", texte)
    return texte

seven_text_data['text'].apply(lambda x: normalize(x))
csv_eval['text'].apply(lambda x: normalize(x))



corpus = seven_text_data['text']
vectorizer = TfidfVectorizer(max_features = 10000,
                            stop_words = 'english',
                            )
X = vectorizer.fit_transform(corpus,
                            )


corpus_eval = csv_eval['text']
X_eval = vectorizer.transform(corpus_eval,
                            )


features = vectorizer.get_feature_names_out()


print(features)


print("vectorized corpus dimensions : ", X.shape)
print("corpus dataset dimensions : ", seven_text_data.shape)


y = seven_text_data['label'].values
y.shape


X_train, X_test, y_train, y_test = train_test_split(X,y)


print(X_train)


clf =  RandomForestClassifier(max_depth=10, random_state=12, class_weight='balanced')

clf.fit(X_train, y_train)

selector = SelectFromModel(clf, max_features=150, prefit=True)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
X_eval = selector.transform(X_eval)


reg = linear_model.LogisticRegression(class_weight='balanced')
xgb_model = XGBClassifier(scale_pos_weight=2)
clf =  RandomForestClassifier(max_depth=10, random_state=12, class_weight='balanced')


ensemble = VotingClassifier(estimators=[('lr', reg),('sgd', xgb_model), ('forest', clf) ],  voting='soft')
ensemble.fit(X_train, y_train)





#reg = linear_model.LogisticRegression()


#reg.fit(X_train,y_train)


y_pred = ensemble.predict(X_test)


metrics.f1_score(y_test,y_pred)


csv_eval.head()


print(X_eval)


y_pred_eval = ensemble.predict_proba(X_eval)


y_pred_no_prob_eval = ensemble.predict(X_eval)


print(len(y_pred_eval[:, 1]))


d = {'id' : csv_eval['id'], 'generated': y_pred_eval[:, 1]}
df_submission = pd.DataFrame(data=d)


df_submission.head()


df_submission.to_csv("submission.csv", index=False)

